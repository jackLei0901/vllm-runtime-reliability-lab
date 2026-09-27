"""Closed, fail-closed score for the three call-history variants."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from inflight_trace_v2 import read_private_snapshot, validate_rank_pair
from ptimer_standalone_score import score as score_graph
from ptimer_trace import read_private_clocks

VARIANTS = ("no_eager", "same_comm", "other_comm")


def _starts(events):
    return [event for event in events if event.kind == "coll_start"]


def score(before, during, after, clocks, *, variant: str) -> dict[str, object]:
    if variant not in VARIANTS:
        raise ValueError("unknown call-history variant")
    for snapshot in (before, during, after):
        validate_rank_pair(snapshot)
    bindings = []
    for rank in (0, 1):
        if (
            during[rank][: len(before[rank])] != before[rank]
            or after[rank][: len(during[rank])] != during[rank]
        ):
            raise ValueError("callback history is not a prefix")
        initial = _starts(before[rank])
        inserted = _starts(during[rank][len(before[rank]) :])
        replay = _starts(after[rank][len(during[rank]) :])
        if (
            len(initial) != 5
            or len(replay) != 6
            or any(event.func != "AllReduce" for event in initial + replay)
        ):
            raise ValueError("unexpected warmup/capture/replay shape")
        primary = initial[0].comm
        secondary = initial[1].comm
        if primary == secondary or any(
            event.comm != primary for event in initial[2:] + replay
        ):
            raise ValueError("two distinct communicators or primary route unavailable")
        if variant == "no_eager":
            if inserted:
                raise ValueError("control unexpectedly inserted a collective")
        elif (
            len(inserted) != 1
            or inserted[0].func != "AllReduce"
            or inserted[0].comm
            != (primary if variant == "same_comm" else secondary)
        ):
            raise ValueError("intervening collective routed to wrong communicator")
        bindings.append((primary, secondary))
    if bindings[0] != bindings[1]:
        raise ValueError("communicator identities differ across ranks")
    graph = score_graph(during, after, clocks, mode="graph")
    return {
        "schema": "tp-ptimer-call-history-score-v1",
        "variant": variant,
        "outcome": graph["outcome"],
        "by_rank": graph["by_rank"],
        "distinct_communicators_verified": True,
        "intervening_route_verified": True,
        "numeric_result": "pass",
        "mechanism_proven": False,
        "stock_metric_claim": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-dir", required=True, type=Path)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--before-sha256", required=True)
    parser.add_argument("--during-sha256", required=True)
    parser.add_argument("--after-sha256", required=True)
    args = parser.parse_args()
    for digest in (args.before_sha256, args.during_sha256, args.after_sha256):
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            parser.error("snapshot digests must be lowercase SHA-256")
    private = args.private_dir.resolve(strict=True)
    snapshots = [
        read_private_snapshot(private / f"callbacks-{label}.json", expected_sha256=digest)
        for label, digest in (
            ("before", args.before_sha256),
            ("during", args.during_sha256),
            ("after", args.after_sha256),
        )
    ]
    clocks, log_digests = read_private_clocks(str(private / "nccl.*.log"))
    result = score(*snapshots, clocks, variant=args.variant)
    result.update(
        {
            "before_sha256": args.before_sha256,
            "during_sha256": args.during_sha256,
            "after_sha256": args.after_sha256,
            "rank_log_sha256": {str(rank): digest for rank, digest in log_digests.items()},
        }
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
