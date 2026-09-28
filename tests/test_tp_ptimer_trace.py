from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "experiments" / "vllm-tp-dfx"
COMM = "0" * 16


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(ROOT))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(ROOT))
    return module


trace = load_module("llr_tp_ptimer_test", "ptimer_trace.py")
baseline = sys.modules["inflight_trace_v2"]
Event = baseline.Event


def event(kind: str, occurrence: int, value: int = 0) -> Event:
    return Event(kind, COMM, occurrence, "AllReduce", 7, value)


def lines(rank: int, *, clock: int = 111, missing_clock: bool = False) -> str:
    suffix = "" if missing_clock else f" ptimer={clock}"
    return (
        f"PROFILER/Plugin: init nranks: 2 rank: {rank}\n"
        f"LLR_TP_EVT_V2 coll_start comm={COMM} occurrence=1 "
        "func=AllReduce seq=7 channels=1\n"
        f"LLR_TP_EVT_V2 kernel_ch_start comm={COMM} occurrence=1 "
        "func=AllReduce seq=7 channel=0 ptimer=1\n"
        f"LLR_TP_EVT_V2 kernel_ch_stop comm={COMM} occurrence=1 "
        "func=AllReduce seq=7 channel=0 ptimer=2\n"
        f"LLR_TP_EVT_V2 coll_start comm={COMM} occurrence=2 "
        "func=AllReduce seq=7 channels=1\n"
        f"LLR_TP_EVT_V2 kernel_ch_start comm={COMM} occurrence=2 "
        f"func=AllReduce seq=7 channel=0{suffix}\n"
        f"LLR_TP_EVT_V2 kernel_ch_stop comm={COMM} occurrence=2 "
        "func=AllReduce seq=7 channel=0 ptimer=222\n"
    )


class TPPtimerTraceTest(unittest.TestCase):
    def test_existing_reader_accepts_trailing_clock_and_summary_is_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for rank in (0, 1):
                (root / f"nccl.{rank}.log").write_text(lines(rank), encoding="utf-8")
            pattern = str(root / "nccl.*.log")
            parsed = baseline.read_rank_logs(pattern)
            self.assertEqual(len(parsed[0]), 6)
            clocks, digests = trace.read_private_clocks(pattern)
            self.assertEqual(set(digests), {0, 1})
            self.assertEqual(
                trace.summarize_all_clocks(clocks)["0"]["by_kind"]["kernel_ch_start"][
                    "events"
                ],
                2,
            )
            before = {rank: parsed[rank][:3] for rank in (0, 1)}
            end = {rank: parsed[rank] for rank in (0, 1)}
            summary = trace.summarize_window(before, end, clocks)
            self.assertEqual(
                summary["0"]["by_kind"]["kernel_ch_start"],
                {
                    "events": 1,
                    "zero_clocks": 0,
                    "comm_channel_groups": 1,
                    "largest_group_event_count": 1,
                    "groups_with_at_least_six_events": 0,
                    "groups_with_repeated_clock": 0,
                    "largest_equal_value_class_within_group": 1,
                    "nonincreasing_adjacent_pairs": 0,
                },
            )
            self.assertEqual(summary["1"]["by_kind"]["kernel_ch_stop"]["events"], 1)
            self.assertNotIn("111", str(summary))
            self.assertNotIn(COMM, str(summary))

    def test_missing_clock_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nccl.0.log").write_text(
                lines(0, missing_clock=True), encoding="utf-8"
            )
            (root / "nccl.1.log").write_text(lines(1), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing or malformed pTimer"):
                trace.read_private_clocks(str(root / "nccl.*.log"))

    def test_clock_classes_are_scoped_to_communicator_and_channel(self) -> None:
        a = "a" * 16
        b = "b" * 16
        events = [
            trace.ClockEvent("kernel_ch_start", a, 1, "AllReduce", 7, 0, 10),
            trace.ClockEvent("kernel_ch_stop", a, 1, "AllReduce", 7, 0, 20),
            trace.ClockEvent("kernel_ch_start", a, 2, "AllReduce", 7, 0, 11),
            trace.ClockEvent("kernel_ch_stop", a, 2, "AllReduce", 7, 0, 30),
            trace.ClockEvent("kernel_ch_start", b, 1, "AllReduce", 7, 0, 10),
            trace.ClockEvent("kernel_ch_stop", b, 1, "AllReduce", 7, 0, 20),
        ]
        summary = trace._summarize_rank(events)
        self.assertEqual(summary["paired_channels_in_window"], 3)
        self.assertEqual(summary["largest_paired_group_event_count"], 2)
        self.assertEqual(summary["nonpositive_pair_duration"], 0)
        self.assertEqual(
            summary["by_kind"]["kernel_ch_start"][
                "largest_equal_value_class_within_group"
            ],
            1,
        )
        self.assertEqual(
            summary["by_kind"]["kernel_ch_start"]["comm_channel_groups"], 2
        )
        # A's second START precedes its first STOP; batching can make this real.
        events.append(trace.ClockEvent("kernel_ch_start", a, 3, "AllReduce", 7, 0, 30))
        events.append(trace.ClockEvent("kernel_ch_stop", a, 3, "AllReduce", 7, 0, 30))
        self.assertEqual(trace._summarize_rank(events)["nonpositive_pair_duration"], 1)

    def test_snapshot_prefix_and_missing_marker_fail_closed(self) -> None:
        captured = (
            event("coll_start", 1, 1),
            event("kernel_ch_start", 1),
            event("kernel_ch_stop", 1),
        )
        new = (
            event("coll_start", 2, 1),
            event("kernel_ch_start", 2),
        )
        before = {0: captured, 1: captured}
        end = {0: captured + new, 1: captured}
        with self.assertRaisesRegex(ValueError, "lacks a pTimer marker"):
            trace.summarize_window(before, end, {0: {}, 1: {}})
        changed = tuple(
            Event(item.kind, item.comm, item.occurrence, item.func, 8, item.value)
            for item in captured
        )
        with self.assertRaisesRegex(ValueError, "not a prefix"):
            trace.summarize_window(
                before, {0: changed + new, 1: captured}, {0: {}, 1: {}}
            )


if __name__ == "__main__":
    unittest.main()
