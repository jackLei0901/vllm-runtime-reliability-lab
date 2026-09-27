from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "experiments" / "vllm-tp-dfx"
COMM = "b" * 16


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


scorer = load("llr_ptimer_standalone_score_test", "ptimer_standalone_score.py")
Event = sys.modules["inflight_trace_v2"].Event
ClockEvent = sys.modules["ptimer_trace"].ClockEvent


def occurrence(index: int):
    return (
        Event("coll_start", COMM, index, "AllReduce", 1, 1),
        Event("kernel_ch_start", COMM, index, "AllReduce", 1, 0),
        Event("kernel_ch_stop", COMM, index, "AllReduce", 1, 0),
    )


def fixtures(new_count: int, *, reused: bool = False):
    before = {rank: occurrence(1) for rank in (0, 1)}
    after = {
        rank: before[rank] + tuple(
            event for index in range(2, new_count + 2) for event in occurrence(index)
        )
        for rank in (0, 1)
    }
    clocks = {}
    for rank in (0, 1):
        markers = {}
        for index in range(1, new_count + 2):
            start = 10 if reused and index > 1 else index * 10
            stop = 15 if reused and index > 1 else index * 10 + 5
            for kind, clock in (("kernel_ch_start", start), ("kernel_ch_stop", stop)):
                event = ClockEvent(kind, COMM, index, "AllReduce", 1, 0, clock)
                markers[event.key] = event
        clocks[rank] = markers
    return before, after, clocks


class StandaloneScoreTest(unittest.TestCase):
    def test_eager_control_passes_only_for_ordered_distinct_clocks(self) -> None:
        before, after, clocks = fixtures(3)
        self.assertEqual(
            scorer.score(before, after, clocks, mode="eager")["outcome"],
            "eager_instrument_pass",
        )
        before, after, clocks = fixtures(3, reused=True)
        self.assertEqual(
            scorer.score(before, after, clocks, mode="eager")["outcome"],
            "eager_instrument_failed",
        )

    def test_graph_reuse_requires_both_ranks(self) -> None:
        before, after, clocks = fixtures(6, reused=True)
        result = scorer.score(before, after, clocks, mode="graph")
        self.assertEqual(result["outcome"], "graph_callback_clock_reuse_observed")
        self.assertNotIn(COMM, str(result))
        self.assertNotIn("15", str(result))
        _, _, distinct = fixtures(6)
        clocks[1] = distinct[1]
        self.assertEqual(
            scorer.score(before, after, clocks, mode="graph")["outcome"],
            "one_rank_only_unscored",
        )
        self.assertEqual(
            scorer.score(before, after, distinct, mode="graph")["outcome"],
            "graph_reuse_not_observed",
        )

    def test_missing_channel_clock_rejected(self) -> None:
        before, after, clocks = fixtures(3)
        one_key = next(key for key in clocks[0] if key[2] == 2 and key[0] == "kernel_ch_stop")
        del clocks[0][one_key]
        with self.assertRaisesRegex(ValueError, "lacks a pTimer marker"):
            scorer.score(before, after, clocks, mode="eager")


if __name__ == "__main__":
    unittest.main()
