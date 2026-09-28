from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "experiments" / "vllm-tp-dfx"
COMM = "a" * 16


def load(name: str, filename: str):
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


audit = load("llr_ptimer_posthoc_test", "ptimer_posthoc_audit.py")
Event = sys.modules["inflight_trace_v2"].Event
ClockEvent = sys.modules["ptimer_trace"].ClockEvent


def occurrence(number: int, func: str = "AllReduce"):
    return (
        Event("coll_start", COMM, number, func, 7, 1),
        Event("kernel_ch_start", COMM, number, func, 7, 0),
        Event("kernel_ch_stop", COMM, number, func, 7, 0),
    )


def clock_map(*entries):
    result = {}
    for number, func, start, stop in entries:
        for kind, value in (("kernel_ch_start", start), ("kernel_ch_stop", stop)):
            item = ClockEvent(kind, COMM, number, func, 7, 0, value)
            result[item.key] = item
    return result


class PosthocPtimerTest(unittest.TestCase):
    def test_distinct_occurrence_reuse_and_history_are_counted(self) -> None:
        before = {rank: occurrence(1) for rank in (0, 1)}
        end = {rank: before[rank] + occurrence(2) + occurrence(3) for rank in (0, 1)}
        clocks = {
            rank: clock_map(
                (1, "AllReduce", 100, 200),
                (2, "AllReduce", 50, 150),
                (3, "AllReduce", 50, 150),
            )
            for rank in (0, 1)
        }
        result = audit.audit_windows(before, end, clocks)
        self.assertEqual(result["by_rank"]["0"]["clocks_below_pre_window_max"], 4)
        self.assertEqual(result["by_rank"]["0"]["history_comparable_events"], 4)
        self.assertEqual(
            result["by_rank"]["0"]["by_kind"]["kernel_ch_start"][
                "largest_equal_clock_distinct_occurrences"
            ],
            2,
        )
        self.assertEqual(
            result["by_rank"]["1"]["by_kind"]["kernel_ch_stop"][
                "occurrences_in_repeated_classes"
            ],
            2,
        )
        self.assertNotIn(COMM, str(result))
        self.assertNotIn("150", str(result))

    def test_function_stratification_prevents_spurious_reuse(self) -> None:
        before = {rank: occurrence(1) for rank in (0, 1)}
        end = {
            rank: before[rank] + occurrence(2) + occurrence(3, "AllGather")
            for rank in (0, 1)
        }
        clocks = {
            rank: clock_map(
                (1, "AllReduce", 10, 20),
                (2, "AllReduce", 30, 40),
                (3, "AllGather", 30, 40),
            )
            for rank in (0, 1)
        }
        result = audit.audit_windows(before, end, clocks)
        selected = result["by_rank"]["0"]["by_kind"]["kernel_ch_start"]
        self.assertEqual(selected["groups"], 2)
        self.assertEqual(selected["groups_with_reuse"], 0)

    def test_missing_clock_and_nonprefix_fail_closed(self) -> None:
        before = {rank: occurrence(1) for rank in (0, 1)}
        end = {rank: before[rank] + occurrence(2) for rank in (0, 1)}
        clocks = {rank: clock_map((1, "AllReduce", 10, 20)) for rank in (0, 1)}
        with self.assertRaisesRegex(ValueError, "lacks a pTimer"):
            audit.audit_windows(before, end, clocks)
        end[0] = occurrence(2) + occurrence(1)
        with self.assertRaises(ValueError):
            audit.audit_windows(before, end, clocks)


if __name__ == "__main__":
    unittest.main()
