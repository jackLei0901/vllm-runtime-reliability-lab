from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "experiments" / "vllm-tp-dfx"
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location(
    "llr_ptimer_call_history_score_test", ROOT / "ptimer_call_history_score.py"
)
assert SPEC is not None and SPEC.loader is not None
scorer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scorer)
from inflight_trace_v2 import Event  # noqa: E402
from ptimer_trace import ClockEvent  # noqa: E402

PRIMARY = "a" * 16
SECONDARY = "b" * 16


def occurrence(comm: str, index: int):
    return (
        Event("coll_start", comm, index, "AllReduce", 1, 1),
        Event("kernel_ch_start", comm, index, "AllReduce", 1, 0),
        Event("kernel_ch_stop", comm, index, "AllReduce", 1, 0),
    )


def fixture(variant: str, *, reuse: bool = False, wrong_comm: bool = False):
    initial = occurrence(PRIMARY, 1) + occurrence(SECONDARY, 1) + tuple(
        event for index in range(2, 5) for event in occurrence(PRIMARY, index)
    )
    eager_comm = SECONDARY if variant == "other_comm" else PRIMARY
    if wrong_comm:
        eager_comm = PRIMARY if eager_comm == SECONDARY else SECONDARY
    eager_index = 2 if eager_comm == SECONDARY else 5
    inserted = occurrence(eager_comm, eager_index) if variant != "no_eager" else ()
    first_replay = 6 if inserted and eager_comm == PRIMARY else 5
    replay = tuple(
        event
        for index in range(first_replay, first_replay + 6)
        for event in occurrence(PRIMARY, index)
    )
    before = {rank: initial for rank in (0, 1)}
    during = {rank: initial + inserted for rank in (0, 1)}
    after = {rank: initial + inserted + replay for rank in (0, 1)}
    clocks = {}
    for rank in (0, 1):
        markers = {}
        for comm, index in (
            [(PRIMARY, i) for i in range(1, 5)]
            + [(SECONDARY, 1)]
            + ([(eager_comm, eager_index)] if inserted else [])
            + [(PRIMARY, i) for i in range(first_replay, first_replay + 6)]
        ):
            start = 100 if reuse and index >= first_replay and comm == PRIMARY else index * 10
            for kind, clock in (
                ("kernel_ch_start", start),
                ("kernel_ch_stop", start + 5),
            ):
                event = ClockEvent(kind, comm, index, "AllReduce", 1, 0, clock)
                markers[event.key] = event
        clocks[rank] = markers
    return before, during, after, clocks


class CallHistoryScoreTest(unittest.TestCase):
    def test_same_comm_only_reuse_is_scored(self) -> None:
        for variant in scorer.VARIANTS:
            with self.subTest(variant=variant):
                result = scorer.score(
                    *fixture(variant, reuse=variant == "same_comm"), variant=variant
                )
                expected = (
                    "graph_callback_clock_reuse_observed"
                    if variant == "same_comm"
                    else "graph_reuse_not_observed"
                )
                self.assertEqual(result["outcome"], expected)
                self.assertNotIn(PRIMARY, str(result))
                self.assertNotIn(SECONDARY, str(result))

    def test_wrong_intervening_route_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "wrong communicator"):
            scorer.score(*fixture("other_comm", wrong_comm=True), variant="other_comm")

    def test_missing_eager_or_secondary_comm_rejected(self) -> None:
        before, during, after, clocks = fixture("no_eager")
        with self.assertRaisesRegex(ValueError, "intervening collective"):
            scorer.score(before, during, after, clocks, variant="same_comm")
        before, during, after, clocks = fixture("no_eager")
        before = {rank: before[rank][:3] + before[rank][6:] for rank in (0, 1)}
        with self.assertRaises(ValueError):
            scorer.score(before, during, after, clocks, variant="no_eager")


if __name__ == "__main__":
    unittest.main()
