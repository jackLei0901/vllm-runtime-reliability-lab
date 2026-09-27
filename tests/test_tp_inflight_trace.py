from __future__ import annotations

import importlib.util
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "vllm-tp-dfx"
    / "inflight_trace_v2.py"
)
spec = importlib.util.spec_from_file_location("llr_tp_inflight_trace_test", PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
Event = module.Event
COMM = "0" * 16


def event(kind: str, occurrence: int, value: int = 0) -> object:
    return Event(kind, COMM, occurrence, "AllReduce", 7, value)


def triplet(channels: int = 1) -> tuple[dict, dict, dict]:
    captured = (event("coll_start", 1, channels),) + tuple(
        item
        for channel in range(channels)
        for item in (
            event("kernel_ch_start", 1, channel),
            event("kernel_ch_stop", 1, channel),
        )
    )
    new_start = (event("coll_start", 2, channels),) + tuple(
        event("kernel_ch_start", 2, channel) for channel in range(channels)
    )
    new_stop = tuple(event("kernel_ch_stop", 2, channel) for channel in range(channels))
    before = {0: captured, 1: captured}
    during = {0: captured + new_start, 1: captured}
    after = {0: during[0] + new_stop, 1: captured + new_start + new_stop}
    return before, during, after


class TPInflightTraceTest(unittest.TestCase):
    def test_single_and_multi_channel_asymmetry(self) -> None:
        for channels in (1, 2):
            with self.subTest(channels=channels):
                before, during, after = triplet(channels)
                self.assertEqual(
                    module.score_triplet(
                        before,
                        during,
                        after,
                        held_during_snapshot=True,
                        request_completed_after_release=True,
                    ),
                    {
                        "result": "start_asymmetry_observed",
                        "func": "AllReduce",
                        "occurrences_after_observe": {0: 1, 1: 1},
                        "step_structure_verified": False,
                        "step_index": None,
                        "position_within_step": None,
                        "predicted_hold_match": None,
                        "rank_zero_channel_starts_during_hold": channels,
                        "rank_one_channel_starts_during_hold": 0,
                        "rank_one_caught_up_after_release": True,
                    },
                )

    def test_missing_hold_and_failed_release_do_not_score(self) -> None:
        before, during, after = triplet()
        for held, completed in ((False, True), (True, False)):
            with self.subTest(held=held, completed=completed):
                result = module.score_triplet(
                    before,
                    during,
                    after,
                    held_during_snapshot=held,
                    request_completed_after_release=completed,
                )
                self.assertEqual(result["reason"], "window_or_release_unverified")

    def test_missing_rank_and_discontinuous_log_do_not_score(self) -> None:
        before, during, after = triplet()
        missing = module.score_triplet(
            {0: before[0]},
            during,
            after,
            held_during_snapshot=True,
            request_completed_after_release=True,
        )
        self.assertEqual(missing["reason"], "rank_binding_unavailable")
        discontinuous = module.score_triplet(
            before,
            {0: during[0][1:], 1: during[1]},
            after,
            held_during_snapshot=True,
            request_completed_after_release=True,
        )
        self.assertEqual(discontinuous["reason"], "log_history_not_contiguous")

    def test_rank_one_must_catch_up(self) -> None:
        before, during, after = triplet()
        result = module.score_triplet(
            before,
            during,
            {0: after[0], 1: during[1]},
            held_during_snapshot=True,
            request_completed_after_release=True,
        )
        self.assertEqual(result["reason"], "rank_occurrence_mismatch")

    def test_malformed_channel_events_rejected(self) -> None:
        before, during, after = triplet()
        for bad_event in (
            event("kernel_ch_start", 3, 0),
            event("kernel_ch_start", 1, 1),
            event("kernel_ch_stop", 1, 0),
            event("unrecognized_kind", 1, 0),
        ):
            with self.subTest(bad_event=bad_event):
                bad_before = before[0] + (bad_event,)
                bad_during = bad_before + during[0][len(before[0]) :]
                bad_after = bad_during + after[0][len(during[0]) :]
                with self.assertRaises(ValueError):
                    module.score_triplet(
                        {0: bad_before, 1: before[1]},
                        {0: bad_during, 1: during[1]},
                        {0: bad_after, 1: after[1]},
                        held_during_snapshot=True,
                        request_completed_after_release=True,
                    )

    def test_private_log_parser_requires_both_ranks_and_complete_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "nccl.1.log"
            second = Path(directory) / "nccl.2.log"
            first.write_text(
                "host [0] NCCL INFO PROFILER/Plugin: init nranks: 2 rank: 0\n"
                "LLR_TP_EVT_V2 coll_start comm=0000000000000000 occurrence=1 func=AllReduce seq=7 channels=2\n"
                "LLR_TP_EVT_V2 kernel_ch_start comm=0000000000000000 occurrence=1 func=AllReduce seq=7 channel=0\n",
                encoding="utf-8",
            )
            second.write_text(
                "host [1] NCCL INFO PROFILER/Plugin: init nranks: 2 rank: 1\n"
                "host [0] NCCL INFO unrelated communicator status\n",
                encoding="utf-8",
            )
            observed = module.read_rank_logs(str(Path(directory) / "*.log"))
            self.assertEqual(len(observed[0]), 2)
            self.assertEqual(observed[1], ())
            second.write_text(
                "host [0] NCCL INFO PROFILER/Plugin: init nranks: 2 rank: 0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate rank"):
                module.read_rank_logs(str(Path(directory) / "*.log"))
            second.write_text(
                "host [1] NCCL INFO PROFILER/Plugin: init nranks: 2 rank: 1\n"
                "LLR_TP_EVT malformed\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "malformed"):
                module.read_rank_logs(str(Path(directory) / "*.log"))

    def test_replayed_sequence_does_not_alias_occurrences(self) -> None:
        observations = tuple(
            event(kind, occurrence, value)
            for occurrence in (1, 2)
            for kind, value in (
                ("coll_start", 1),
                ("kernel_ch_start", 0),
                ("kernel_ch_stop", 0),
            )
        )
        self.assertEqual(len(module.validate_history(observations)), 2)
        with self.assertRaisesRegex(ValueError, "duplicate collective occurrence"):
            module.validate_history(observations + (event("coll_start", 2, 2),))

    def test_control_rejects_the_historical_identity_collision(self) -> None:
        before, _, after = triplet()
        module.validate_control_history(before, after)
        collision = after[0] + (event("coll_start", 2, 2),)
        with self.assertRaisesRegex(ValueError, "duplicate collective occurrence"):
            module.validate_control_history(before, {0: collision, 1: after[1]})

    def test_cross_rank_ordinal_is_checked_not_assumed(self) -> None:
        before, _, after = triplet()
        changed = list(after[1])
        changed[len(before[1])] = Event(
            "coll_start", COMM, 2, "AllGather", 7, 1
        )
        changed[len(before[1]) + 1] = Event(
            "kernel_ch_start", COMM, 2, "AllGather", 7, 0
        )
        changed[len(before[1]) + 2] = Event(
            "kernel_ch_stop", COMM, 2, "AllGather", 7, 0
        )
        with self.assertRaisesRegex(ValueError, "rank descriptors disagree"):
            module.validate_control_history(before, {0: after[0], 1: tuple(changed)})

    def test_step_position_is_only_reported_for_complete_shape(self) -> None:
        events = tuple(
            Event(
                "coll_start",
                COMM,
                index + 1,
                "AllGather" if index % 74 == 73 else "AllReduce",
                7,
                1,
            )
            for index in range(16 * 74)
        )
        after = {0: events, 1: events}
        key = events[74].key  # First collective of the first decode step.
        descriptors = {event.key: (event.func, event.seq, event.value) for event in events}
        facts = module._request_step_metadata({0: (), 1: ()}, after, key, descriptors, 1)
        self.assertEqual(facts["occurrences_after_observe"], {0: 1184, 1: 1184})
        self.assertEqual(facts["step_index"], 2)
        self.assertEqual(facts["position_within_step"], 1)
        self.assertTrue(facts["predicted_hold_match"])
        without_all_gather = events[:73] + events[74:]
        bad = module._request_step_metadata(
            {0: (), 1: ()},
            {0: without_all_gather, 1: without_all_gather},
            key,
            descriptors,
            1,
        )
        self.assertFalse(bad["step_structure_verified"])
        self.assertIsNone(bad["predicted_hold_match"])

    def test_private_snapshots_replay_exact_events(self) -> None:
        before, during, after = triplet()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label, snapshot in (
                ("before", before),
                ("during", during),
                ("after", after),
            ):
                digest = module.save_private_snapshot(root, label, snapshot)
                path = root / f"callbacks-{label}.json"
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
                self.assertEqual(
                    module.read_private_snapshot(path, expected_sha256=digest), snapshot
                )
                with self.assertRaisesRegex(ValueError, "digest mismatch"):
                    module.read_private_snapshot(path, expected_sha256="0" * 64)
                with self.assertRaisesRegex(ValueError, "already exists"):
                    module.save_private_snapshot(root, label, snapshot)


if __name__ == "__main__":
    unittest.main()
