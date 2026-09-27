from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_bundle import native_observations

from dfxlab.bundle import BundleError, write_bundle
from dfxlab.replay import ReplayError, replay
from dfxlab.timeline import (
    build_timeline,
    render_html,
    to_trace_events,
    write_timeline,
)
from dfxlab.timeline_bundle import build_bundle_timeline
from dfxlab.timeline_facts import TimelineError, to_trace, validate_timeline
from dfxlab.verify_bundle import verify_bundle

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "results" / "vllm-zmq-backpressure-stage1-r3-20260916"


class TimelineTests(unittest.TestCase):
    def test_published_case_draws_only_recorded_times(self) -> None:
        model = build_timeline(PUBLISHED)
        cells = {cell["label"]: cell for cell in model["cells"]}
        base = cells["base_pause"]
        self.assertAlmostEqual(base["gap"]["start"], 0.3097, places=3)
        self.assertAlmostEqual(base["gap"]["end"], 10.3828, places=3)
        self.assertEqual(
            {item["kind"] for item in base["bounded"]}, {"health", "stack"}
        )
        for item in base["bounded"]:
            self.assertEqual(item["start"], base["gap"]["start"])
            self.assertEqual(item["end"], base["release"])
        self.assertIsNone(cells["base_control"]["gap"])
        self.assertIsNone(cells["fix_pause"]["gap"])
        self.assertIn("4 KV-event batches dropped", cells["fix_pause"]["untimed"])

    def test_trace_has_no_causal_edges_or_private_fields(self) -> None:
        trace = to_trace_events(build_timeline(PUBLISHED))
        phases = {event["ph"] for event in trace["traceEvents"]}
        self.assertLessEqual(phases, {"M", "I", "X"})
        windows = [
            event
            for event in trace["traceEvents"]
            if event["name"].startswith("Possible time of one sample")
        ]
        self.assertEqual(len(windows), 2)
        self.assertEqual({event["tid"] for event in windows}, {3, 4})
        self.assertTrue(all(event["args"]["window_only"] for event in windows))
        self.assertIn("independent runs", trace["otherData"]["lane_identity"])
        process_names = [
            event["args"]["name"]
            for event in trace["traceEvents"]
            if event["ph"] == "M" and event["name"] == "process_name"
        ]
        self.assertEqual(len(process_names), 4)
        self.assertTrue(
            all(name.startswith("Independent run ") for name in process_names)
        )
        page = render_html(build_timeline(PUBLISHED))
        self.assertIn("possible time window", page)
        self.assertIn("not continuous states", page)
        self.assertIn("one\n<code>/health</code> probe", page)
        self.assertNotIn("/health</code> stayed", page)
        self.assertIn("Four independent runs; each", page)
        self.assertIn("0.31 → 10.38 s", page)
        self.assertIn("Untimed, Cell 4:", page)
        self.assertNotIn('class="untimed"', page)
        self.assertIn('class="bounded health"', page)
        self.assertIn('class="bounded stack"', page)
        empty_track_names = [
            event["args"]["name"]
            for event in trace["traceEvents"]
            if event["ph"] == "M" and event["name"] == "thread_name"
        ]
        self.assertEqual(empty_track_names.count("possible /health sample time"), 1)
        self.assertEqual(empty_track_names.count("possible stack sample time"), 1)
        rendered = json.dumps(trace) + page
        for field in (
            "private_server_log_sha256",
            "request_sha256",
            "server_command_sha256",
            "raw_sha256",
        ):
            self.assertNotIn(field, rendered)
        for event in trace["traceEvents"]:
            if event["ph"] != "M":
                self.assertGreaterEqual(event["ts"], 0)

    def test_tampered_bundle_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "case"
            shutil.copytree(PUBLISHED, copied)
            path = copied / "cell-3-base-pause.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            record["release_offset_seconds"] = 1.0
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaises(ReplayError):
                build_timeline(copied)
            output = Path(temporary) / "output"
            with self.assertRaises(ReplayError):
                write_timeline(copied, output)
            self.assertFalse(output.exists())

    def test_existing_output_directory_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            output.mkdir()
            old_page = output / "timeline.html"
            old_page.write_text("old output", encoding="utf-8")
            with self.assertRaisesRegex(ReplayError, "already exists"):
                write_timeline(PUBLISHED, output)
            self.assertEqual(old_page.read_text(encoding="utf-8"), "old output")

    def test_generated_files_have_platform_independent_lf_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "timeline"
            write_timeline(PUBLISHED, output)
            for name in ("timeline.html", "timeline.trace.json"):
                self.assertNotIn(b"\r", (output / name).read_bytes())

    def test_native_bundle_uses_same_trace_renderer_without_private_identity(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            write_bundle(bundle, native_observations())
            facts = build_bundle_timeline(bundle)
            trace = to_trace(facts)
            decision_facts = facts["lanes"][0]["tracks"][2]["facts"]
            self.assertIn("observed_gap", {fact["kind"] for fact in decision_facts})
            self.assertIn(
                "no request chunks observed (producer available)",
                [fact["name"] for fact in decision_facts],
            )
            self.assertIn(
                "1 request starts before collection window",
                [fact["name"] for fact in decision_facts],
            )
            self.assertEqual(
                facts["lanes"][0]["tracks"][3]["facts"][0]["kind"],
                "analysis_window",
            )
            self.assertEqual(
                trace["otherData"]["source_kind"], "verified_incident_bundle_v1"
            )
            self.assertEqual(
                [item["name"] for item in trace["otherData"]["untimed_facts"]],
                [
                    "1 request starts before collection window",
                    "decision producer state: flat (no_content_while_open)",
                    "verified verdict: alive_health_ok_no_progress",
                    "stack producer: disabled",
                ],
            )
            self.assertTrue(any(event["ph"] == "X" for event in trace["traceEvents"]))
            output = root / "output"
            write_timeline(bundle, output)
            page = (output / "timeline.html").read_text(encoding="utf-8")
            self.assertIn("Untimed claims", page)
            self.assertIn('class="tick"', page)
            self.assertIn('class="analysis-window"', page)
            published = (output / "timeline.trace.json").read_text(encoding="utf-8")
            for private in ("12345", "0123456789ab", "linux_start_ticks", "a" * 64):
                self.assertNotIn(private, published)
                self.assertNotIn(private, page)

    def test_native_bundle_tampering_does_not_write_a_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            write_bundle(bundle, native_observations())
            summary = bundle / "summary.json"
            payload = json.loads(summary.read_text(encoding="utf-8"))
            payload["verdict"]["verdict"] = "progress_observed"
            summary.write_text(json.dumps(payload), encoding="utf-8")
            output = root / "output"
            with self.assertRaises(ReplayError):
                write_timeline(bundle, output)
            self.assertFalse(output.exists())

    def test_unavailable_producer_is_not_an_empty_track(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            observations = native_observations()
            observations["producer_inputs"]["client_request"]["producer_available"] = (
                False
            )
            write_bundle(bundle, observations)
            facts = build_bundle_timeline(bundle)["lanes"][0]["tracks"][2]["facts"]
            self.assertEqual(
                [fact["name"] for fact in facts],
                [
                    "decision producer unavailable",
                    "1 request starts before collection window",
                    "decision producer state: producer_missing (unavailable)",
                ],
            )

    def test_non_content_chunk_does_not_hide_content_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            observations = native_observations()
            observations["producer_inputs"]["client_request"]["chunks"] = [
                {"monotonic_ns": 15_000_000_000, "kind": "role_only"}
            ]
            write_bundle(bundle, observations)
            facts = build_bundle_timeline(bundle)["lanes"][0]["tracks"][2]["facts"]
            self.assertIn(
                "no content chunks observed (producer available)",
                [fact["name"] for fact in facts],
            )

    def test_partial_request_window_never_looks_like_full_window_stall(self) -> None:
        for boundary in ("started_late", "completed_early"):
            with (
                self.subTest(boundary=boundary),
                tempfile.TemporaryDirectory() as temporary,
            ):
                bundle = Path(temporary) / "bundle"
                observations = native_observations()
                request = observations["producer_inputs"]["client_request"]
                demand = observations["demand_inputs"]["client_request"]
                if boundary == "started_late":
                    request["request_started_ns"] = 12_000_000_000
                    demand["request_started_ns"] = 12_000_000_000
                    expected = (2.0, 10.0)
                else:
                    request["request_completed_ns"] = 15_000_000_000
                    demand["request_completed_ns"] = 15_000_000_000
                    expected = (0.0, 5.0)
                write_bundle(bundle, observations)
                facts = build_bundle_timeline(bundle)["lanes"][0]["tracks"][2]["facts"]
                self.assertIn(
                    "request not open for the full evaluation window",
                    [fact["name"] for fact in facts],
                )
                gaps = [fact for fact in facts if fact["kind"] == "observed_gap"]
                self.assertEqual(len(gaps), 1)
                self.assertEqual((gaps[0]["start"], gaps[0]["end"]), expected)
                self.assertIn(
                    "decision producer state: insufficient_evidence "
                    "(request_not_open_for_interval)",
                    [fact["name"] for fact in facts],
                )

    def test_decision_state_matrix_has_visible_counterpart(self) -> None:
        for kind in ("client_request", "server_counter"):
            for state in (
                "progressing",
                "flat",
                "producer_missing",
                "insufficient_evidence",
            ):
                with (
                    self.subTest(kind=kind, state=state),
                    tempfile.TemporaryDirectory() as temporary,
                ):
                    bundle = Path(temporary) / "bundle"
                    observations = native_observations()
                    observations["producer_inputs"]["decision_source"] = kind
                    if kind == "client_request":
                        request = observations["producer_inputs"][kind]
                        if state == "progressing":
                            request["chunks"] = [
                                {"monotonic_ns": 15_000_000_000, "kind": "content"}
                            ]
                        elif state == "producer_missing":
                            request["producer_available"] = False
                        elif state == "insufficient_evidence":
                            request["request_started_ns"] = 12_000_000_000
                            observations["demand_inputs"][kind][
                                "request_started_ns"
                            ] = 12_000_000_000
                    else:
                        counter = observations["producer_inputs"][kind]
                        observations["demand_inputs"][kind]["samples"] = [
                            {
                                "monotonic_ns": time,
                                "fresh": True,
                                "running": 1,
                                "waiting": 0,
                            }
                            for time in (10_000_000_000, 20_000_000_000)
                        ]
                        counter["producer_available"] = state != "producer_missing"
                        if state != "producer_missing":
                            values = (
                                [4]
                                if state == "insufficient_evidence"
                                else [4, 5 if state == "progressing" else 4]
                            )
                            counter["samples"] = [
                                {"monotonic_ns": time, "fresh": True, "value": value}
                                for time, value in zip(
                                    (10_000_000_000, 20_000_000_000),
                                    values,
                                    strict=False,
                                )
                            ]
                    write_bundle(bundle, observations)
                    summary = verify_bundle(bundle)
                    self.assertEqual(summary["producers"]["decision"]["state"], state)
                    facts = build_bundle_timeline(bundle)["lanes"][0]["tracks"][2][
                        "facts"
                    ]
                    labels = [fact["name"] for fact in facts]
                    self.assertTrue(
                        any(
                            label.startswith(f"decision producer state: {state} ")
                            for label in labels
                        )
                    )
                    gaps = [fact for fact in facts if fact["kind"] == "observed_gap"]
                    if state == "flat":
                        self.assertEqual(len(gaps), 1)
                    if state == "producer_missing":
                        self.assertEqual(gaps, [])
                        self.assertIn("decision producer unavailable", labels)
                    if kind == "server_counter":
                        if state == "flat":
                            self.assertIn("counter change +0", labels)
                            self.assertEqual(
                                (gaps[0]["start"], gaps[0]["end"]), (0.0, 10.0)
                            )
                        elif state == "progressing":
                            self.assertIn("counter change +1", labels)
                            self.assertEqual(gaps, [])
                        elif state == "insufficient_evidence":
                            self.assertEqual(gaps, [])

    def test_out_of_collection_sample_is_untimed_not_negative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            observations = native_observations()
            observations["process"]["samples"].insert(
                0,
                {
                    "monotonic_ns": 9_000_000_000,
                    "fresh": True,
                    "alive": True,
                    "start_identity_value": "77",
                },
            )
            write_bundle(bundle, observations)
            process = build_bundle_timeline(bundle)["lanes"][0]["tracks"][0]["facts"]
            self.assertIn(
                "1 process samples before collection window",
                [fact["name"] for fact in process],
            )
            self.assertTrue(
                all(fact["start"] is None or fact["start"] >= 0 for fact in process)
            )

    def test_changed_bytes_after_verification_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            write_bundle(bundle, native_observations())

            def verify_then_change(path: Path) -> dict:
                summary = verify_bundle(path)
                observations = path / "observations.json"
                observations.write_text(
                    observations.read_text(encoding="utf-8") + " ", encoding="utf-8"
                )
                return summary

            with mock.patch(
                "dfxlab.timeline_bundle.verify_bundle", side_effect=verify_then_change
            ):
                with self.assertRaisesRegex(BundleError, "changed after verification"):
                    build_bundle_timeline(bundle)

    def test_r3_changed_bytes_after_replay_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "case"
            shutil.copytree(PUBLISHED, copied)

            def replay_then_change(path: Path) -> list[str]:
                lines = replay(path)
                record = path / "cell-3-base-pause.json"
                record.write_text(
                    record.read_text(encoding="utf-8") + " ", encoding="utf-8"
                )
                return lines

            with mock.patch("dfxlab.timeline.replay", side_effect=replay_then_change):
                with self.assertRaisesRegex(ReplayError, "changed after replay"):
                    build_timeline(copied)

    def test_failed_second_write_leaves_no_partial_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = Path.write_bytes

            def fail_html(path: Path, data: bytes) -> int:
                if path.name == "timeline.html":
                    raise OSError("simulated disk error")
                return original(path, data)

            with mock.patch.object(Path, "write_bytes", fail_html):
                with self.assertRaisesRegex(ReplayError, "complete timeline"):
                    write_timeline(PUBLISHED, root / "output")
            self.assertEqual(list(root.iterdir()), [])

    def test_fact_contract_rejects_fabricated_timing(self) -> None:
        model = build_timeline(PUBLISHED)
        facts = to_trace_events(model)
        self.assertIn("traceEvents", facts)
        from dfxlab.timeline import to_fact_timeline

        normalized = to_fact_timeline(model)
        sample = normalized["lanes"][2]["tracks"][2]["facts"][0]
        sample["kind"] = "instant"
        with self.assertRaises(TimelineError):
            validate_timeline(normalized)


if __name__ == "__main__":
    unittest.main()
