from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from dfxlab.bundle import OBSERVATION_SCHEMA, BundleError, write_bundle
from dfxlab.verify_bundle import project_legacy_r3, verify_bundle

ROOT = Path(__file__).resolve().parents[1]
R3 = ROOT / "results" / "vllm-zmq-backpressure-stage1-r3-20260916"


def native_observations(*, progressing: bool = False) -> dict:
    start = 10_000_000_000
    end = 20_000_000_000
    chunks = (
        [{"monotonic_ns": 15_000_000_000, "kind": "content"}] if progressing else []
    )
    return {
        "schema_version": OBSERVATION_SCHEMA,
        "targets": {
            "process": {
                "supplied": True,
                "pid": 12345,
                "start_identity": {"kind": "linux_start_ticks", "value": "77"},
                "identity_source": "operator_supplied",
            },
            "endpoint": {
                "supplied": True,
                "endpoint_id": "sha256:0123456789ab",
                "identity_source": "operator_supplied",
            },
            "relation": "operator_asserted_same_incident",
        },
        "intervals": {
            "collection": {"start_ns": start, "end_ns": end},
            "evaluation": {
                "start_ns": start,
                "end_ns": end,
                "minimum_duration_ns": end - start,
            },
        },
        "process": {
            "samples": [
                {
                    "monotonic_ns": start,
                    "fresh": True,
                    "alive": True,
                    "start_identity_value": "77",
                },
                {
                    "monotonic_ns": end,
                    "fresh": True,
                    "alive": True,
                    "start_identity_value": "77",
                },
            ]
        },
        "health": {
            "failure_threshold": 2,
            "samples": [
                {
                    "monotonic_ns": start,
                    "fresh": True,
                    "ok": True,
                    "status": 200,
                    "error_kind": None,
                },
                {
                    "monotonic_ns": end,
                    "fresh": True,
                    "ok": True,
                    "status": 200,
                    "error_kind": None,
                },
            ],
        },
        "producer_inputs": {
            "decision_source": "client_request",
            "corroborating_source": None,
            "client_request_sha256": "a" * 64,
            "server_counter": {
                "producer_available": False,
                "evaluation_start_ns": start,
                "evaluation_end_ns": end,
                "minimum_span_ns": end - start,
                "minimum_fresh_samples": 2,
                "samples": [],
            },
            "client_request": {
                "producer_available": True,
                "evaluation_start_ns": start,
                "evaluation_end_ns": end,
                "request_started_ns": 1,
                "request_completed_ns": None,
                "chunks": chunks,
            },
        },
        "demand_inputs": {
            "server_counter": {
                "evaluation_start_ns": start,
                "evaluation_end_ns": end,
                "minimum_fresh_samples": 2,
                "samples": [],
            },
            "client_request": {
                "evaluation_start_ns": start,
                "evaluation_end_ns": end,
                "request_started_ns": 1,
                "request_completed_ns": None,
            },
        },
        "stack": {"state": "disabled", "producer": None},
    }


class BundleTest(unittest.TestCase):
    def test_native_r3_facts_recompute_both_verdicts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base"
            fix = root / "fix"
            write_bundle(base, native_observations())
            write_bundle(fix, native_observations(progressing=True))
            self.assertEqual(
                "alive_health_ok_no_progress",
                verify_bundle(base)["verdict"]["verdict"],
            )
            self.assertEqual(
                "progress_observed", verify_bundle(fix)["verdict"]["verdict"]
            )

    def test_semantic_summary_mutation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_bundle(root, native_observations())
            path = root / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            summary["verdict"]["verdict"] = "undetermined"
            path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(BundleError, "recomputed evidence"):
                verify_bundle(root)

    def test_process_identity_loss_is_recomputed_above_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            observations = native_observations(progressing=True)
            final = observations["process"]["samples"][-1]
            final["alive"] = False
            final["start_identity_value"] = None
            write_bundle(root, observations)
            self.assertEqual(
                "process_missing", verify_bundle(root)["verdict"]["verdict"]
            )

    def test_health_loss_is_recomputed_above_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            observations = native_observations(progressing=True)
            start = observations["intervals"]["evaluation"]["start_ns"]
            end = observations["intervals"]["evaluation"]["end_ns"]
            observations["health"]["samples"] = [
                {
                    "monotonic_ns": start,
                    "fresh": True,
                    "ok": True,
                    "status": 200,
                    "error_kind": None,
                },
                {
                    "monotonic_ns": (start + end) // 2,
                    "fresh": True,
                    "ok": False,
                    "status": 503,
                    "error_kind": None,
                },
                {
                    "monotonic_ns": end,
                    "fresh": True,
                    "ok": False,
                    "status": None,
                    "error_kind": "connection",
                },
            ]
            write_bundle(root, observations)
            self.assertEqual("health_lost", verify_bundle(root)["verdict"]["verdict"])

    def test_never_healthy_endpoint_remains_undetermined(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            observations = native_observations()
            for sample in observations["health"]["samples"]:
                sample["ok"] = False
                sample["status"] = 503
            write_bundle(root, observations)
            self.assertEqual("undetermined", verify_bundle(root)["verdict"]["verdict"])

    def test_observation_mutation_fails_digest_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_bundle(root, native_observations())
            path = root / "observations.json"
            observations = json.loads(path.read_text(encoding="utf-8"))
            observations["process"]["samples"][1]["alive"] = False
            path.write_text(json.dumps(observations), encoding="utf-8")
            with self.assertRaisesRegex(BundleError, "digest mismatch"):
                verify_bundle(root)

    def test_unexpected_public_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_bundle(root, native_observations())
            (root / "notes.txt").write_text("extra", encoding="utf-8")
            with self.assertRaisesRegex(BundleError, "file set"):
                verify_bundle(root)

    def test_legacy_projection_reports_unavailable_evidence(self) -> None:
        projection = project_legacy_r3(R3)
        self.assertFalse(projection["native_bundle"])
        self.assertIsNone(projection["cells"]["base_pause"]["native_v0_2_verdict"])
        self.assertEqual(
            {
                "process_start_identity",
                "repeated_health_samples",
                "server_counter_samples",
            },
            set(projection["unavailable"]),
        )

    def test_legacy_projection_checks_immutable_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "r3"
            shutil.copytree(R3, copied)
            path = copied / "cell-3-base-pause.json"
            path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaisesRegex(BundleError, "digest mismatch"):
                project_legacy_r3(copied)


if __name__ == "__main__":
    unittest.main()
