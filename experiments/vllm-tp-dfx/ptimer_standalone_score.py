"""Closed-shape score for the two-rank pTimer standalone reproduction."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from inflight_trace_v2 import read_private_snapshot, validate_rank_pair
from ptimer_posthoc_audit import _window_clocks
from ptimer_trace import ClockEvent, read_private_clocks

EXPECTED = {"eager": 3, "graph": 6}


def score(
    before,
    after,
    clocks,
    *,
    mode: str,
) -> dict[str, object]:
    if mode not in EXPECTED:
        raise ValueError("unknown standalone mode")
    validate_rank_pair(before)
    validate_rank_pair(after)
    windows = _window_clocks(before, after, clocks)
    by_rank = {}
    for rank in (0, 1):
        new_descriptors = [
            item for item in after[rank][len(before[rank]) :]
            if item.kind == "coll_start"
        ]
        if len(new_descriptors) != EXPECTED[mode] or any(
            item.func != "AllReduce" for item in new_descriptors
        ):
            raise ValueError("unexpected standalone collective descriptors")
        events = windows[rank][1]
        pairs: dict[tuple[str, int, int], dict[str, ClockEvent]] = defaultdict(dict)
        for event in events:
            key = (event.comm, event.occurrence, event.channel)
            if event.kind in pairs[key]:
                raise ValueError("duplicate channel marker")
            pairs[key][event.kind] = event
        if not pairs or any(
            set(markers) != {"kernel_ch_start", "kernel_ch_stop"}
            for markers in pairs.values()
        ):
            raise ValueError("incomplete standalone channel clock pairs")
        expected_pairs = {
            (item.comm, item.occurrence, channel)
            for item in new_descriptors
            for channel in range(item.value)
        }
        if set(pairs) != expected_pairs:
            raise ValueError("missing standalone occurrence clock")
        by_group: dict[tuple[str, int], list[tuple[int, int, int]]] = defaultdict(list)
        for (comm, occurrence, channel), markers in pairs.items():
            by_group[(comm, channel)].append(
                (
                    occurrence,
                    markers["kernel_ch_start"].clock,
                    markers["kernel_ch_stop"].clock,
                )
            )
        largest_equal_start = 0
        largest_equal_stop = 0
        nonpositive_pairs = 0
        zero_clocks = 0
        eager_order_violations = 0
        for members in by_group.values():
            ordered = sorted(members)
            starts = Counter(start for _, start, _ in ordered)
            stops = Counter(stop for _, _, stop in ordered)
            largest_equal_start = max(largest_equal_start, max(starts.values()))
            largest_equal_stop = max(largest_equal_stop, max(stops.values()))
            nonpositive_pairs += sum(stop <= start for _, start, stop in ordered)
            zero_clocks += sum(start == 0 or stop == 0 for _, start, stop in ordered)
            if mode == "eager":
                eager_order_violations += sum(
                    next_start <= start or next_start < stop or next_stop <= stop
                    for (_, start, stop), (_, next_start, next_stop)
                    in zip(ordered, ordered[1:])
                )
        by_rank[str(rank)] = {
            "collective_occurrences": len(new_descriptors),
            "paired_channels": len(pairs),
            "comm_channel_groups": len(by_group),
            "largest_equal_start_class": largest_equal_start,
            "largest_equal_stop_class": largest_equal_stop,
            "nonpositive_pairs": nonpositive_pairs,
            "zero_clock_pairs": zero_clocks,
            "eager_order_violations": eager_order_violations if mode == "eager" else None,
        }
    if mode == "eager":
        passed = all(
            row["largest_equal_start_class"] == 1
            and row["largest_equal_stop_class"] == 1
            and row["nonpositive_pairs"] == 0
            and row["zero_clock_pairs"] == 0
            and row["eager_order_violations"] == 0
            for row in by_rank.values()
        )
        outcome = "eager_instrument_pass" if passed else "eager_instrument_failed"
    else:
        reused = [
            row["largest_equal_start_class"] >= 2
            or row["largest_equal_stop_class"] >= 2
            for row in by_rank.values()
        ]
        if all(reused):
            outcome = "graph_callback_clock_reuse_observed"
        elif any(reused):
            outcome = "one_rank_only_unscored"
        else:
            outcome = "graph_reuse_not_observed"
    return {
        "schema": "tp-ptimer-standalone-score-v1",
        "mode": mode,
        "outcome": outcome,
        "by_rank": by_rank,
        "mechanism_proven": False,
        "stock_metric_claim": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-dir", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=tuple(EXPECTED))
    parser.add_argument("--before-sha256", required=True)
    parser.add_argument("--after-sha256", required=True)
    args = parser.parse_args()
    for digest in (args.before_sha256, args.after_sha256):
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            parser.error("snapshot digests must be lowercase SHA-256")
    private = args.private_dir.resolve(strict=True)
    before = read_private_snapshot(
        private / "callbacks-before.json", expected_sha256=args.before_sha256
    )
    after = read_private_snapshot(
        private / "callbacks-after.json", expected_sha256=args.after_sha256
    )
    clocks, digests = read_private_clocks(str(private / "nccl.*.log"))
    result = score(before, after, clocks, mode=args.mode)
    result.update(
        {
            "before_sha256": args.before_sha256,
            "after_sha256": args.after_sha256,
            "rank_log_sha256": {str(rank): digest for rank, digest in digests.items()},
        }
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
