from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "vllm-tp-dfx"
    / "inflight_trace.py"
)
spec = importlib.util.spec_from_file_location("llr_tp_inflight_trace_test", PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
Event = module.Event
COMM = "0" * 16


def event(kind: str, seq: int, value: int = 0) -> object:
    return Event(kind, COMM, seq, value)


def triplet(channels: int = 1) -> tuple[dict, dict, dict]:
    captured = (event("coll_start", 7, channels),) + tuple(
        item
        for channel in range(channels)
        for item in (
            event("kernel_ch_start", 7, channel),
            event("kernel_ch_stop", 7, channel),
        )
    )
    new_start = (event("coll_start", 7, channels),) + tuple(
        event("kernel_ch_start", 7, channel) for channel in range(channels)
    )
    new_stop = tuple(event("kernel_ch_stop", 7, channel) for channel in range(channels))
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
                        "communicator_sequence_ordinal": 0,
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
        self.assertEqual(result["reason"], "rank_did_not_catch_up")

    def test_malformed_channel_events_rejected(self) -> None:
        before, during, after = triplet()
        for bad_event in (
            event("kernel_ch_start", 8, 0),
            event("kernel_ch_start", 7, 1),
            event("kernel_ch_stop", 7, 0),
            event("unrecognized_kind", 7, 0),
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
                "LLR_TP_EVT coll_start comm=0000000000000000 seq=7 channels=2\n"
                "LLR_TP_EVT kernel_ch_start comm=0000000000000000 seq=7 channel=0\n",
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


if __name__ == "__main__":
    unittest.main()
