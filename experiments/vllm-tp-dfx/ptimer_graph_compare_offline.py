"""Compare two private pTimer cells without emitting raw identities or clocks.

This is post-hoc analysis, not a change to either frozen scorer. The caller
provides one serving healthy-control cell and one standalone graph cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from inflight_trace_v2 import read_private_snapshot, validate_rank_pair
from ptimer_posthoc_audit import _window_clocks
from ptimer_trace import ClockEvent, read_private_clocks

STEP_FUNCS = ("AllReduce",) * 73 + ("AllGather",)


def _load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("summary must be an object")
    return data


def _load_cell(directory: Path, summary_path: Path, *, serving: bool):
    summary = _load_json(summary_path)
    before_digest = summary["before_sha256"]
    after_digest = summary["end_sha256" if serving else "after_sha256"]
    before = read_private_snapshot(
        directory / "callbacks-before.json", expected_sha256=before_digest
    )
    after = read_private_snapshot(
        directory / "callbacks-after.json", expected_sha256=after_digest
    )
    validate_rank_pair(before)
    validate_rank_pair(after)
    clocks, log_digests = read_private_clocks(str(directory / "nccl.*.log"))
    if {str(rank): digest for rank, digest in log_digests.items()} != summary[
        "rank_log_sha256"
    ]:
        raise ValueError("rank logs do not match retained summary")
    windows = _window_clocks(before, after, clocks)
    return before, after, windows


def _rank_summary(before, after, windows, rank: int, *, serving: bool) -> dict:
    prior_descriptors = [event for event in before[rank] if event.kind == "coll_start"]
    descriptors = [
        event
        for event in after[rank][len(before[rank]) :]
        if event.kind == "coll_start"
    ]
    _, clock_events = windows[rank]
    channel_counts = Counter((event.func, event.value) for event in descriptors)
    by_clock_group: dict[tuple[str, int, str, str], list[ClockEvent]] = defaultdict(
        list
    )
    clock_by_occurrence = {}
    for event in clock_events:
        by_clock_group[(event.comm, event.channel, event.func, event.kind)].append(
            event
        )
        key = (event.comm, event.occurrence, event.channel, event.kind)
        clock_by_occurrence[key] = event.clock
    reuse_by_func: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    for func in sorted({key[2] for key in by_clock_group}):
        for kind in ("kernel_ch_start", "kernel_ch_stop"):
            groups = [
                events
                for key, events in by_clock_group.items()
                if key[2] == func and key[3] == kind
            ]
            class_sizes = [
                len({event.occurrence for event in events if event.clock == clock})
                for events in groups
                for clock in {event.clock for event in events}
            ]
            reuse_by_func[func][kind] = {
                "events": sum(map(len, groups)),
                "groups": len(groups),
                "max_distinct_occurrences_same_clock": max(class_sizes, default=0),
                "occurrences_in_repeated_classes": sum(
                    size for size in class_sizes if size > 1
                ),
            }
    per_pass = []
    width = 74 if serving else 3
    if len(descriptors) % width:
        raise ValueError("new descriptor count does not fit the declared pass width")
    for offset in range(0, len(descriptors), width):
        batch = descriptors[offset : offset + width]
        starts = []
        all_reduce_starts = []
        for desc in batch:
            key = (desc.comm, desc.occurrence, 0, "kernel_ch_start")
            if key not in clock_by_occurrence:
                raise ValueError("pass descriptor lacks channel-zero START clock")
            starts.append(clock_by_occurrence[key])
            if desc.func == "AllReduce":
                all_reduce_starts.append(clock_by_occurrence[key])
        per_pass.append(
            {
                "functions_match_expected": tuple(item.func for item in batch)
                == (STEP_FUNCS if serving else ("AllReduce",) * 3),
                "channel_zero_unique_start_clocks": len(set(starts)),
                "channel_zero_all_equal": len(set(starts)) == 1,
                "allreduce_channel_zero_unique_start_clocks": len(
                    set(all_reduce_starts)
                ),
            }
        )
    allreduce_comms = {d.comm for d in descriptors if d.func == "AllReduce"}
    allgather_comms = {d.comm for d in descriptors if d.func == "AllGather"}
    last_prior = prior_descriptors[-1] if prior_descriptors else None
    return {
        "collective_occurrences": len(descriptors),
        "function_counts": dict(sorted(Counter(d.func for d in descriptors).items())),
        "last_prior_collective_function": last_prior.func if last_prior else None,
        "last_prior_collective_shares_window_allreduce_communicator": (
            last_prior is not None and last_prior.comm in allreduce_comms
        ),
        "last_window_collective_function": (
            descriptors[-1].func if descriptors else None
        ),
        "allgather_and_allreduce_share_communicator": (
            bool(allgather_comms) and allgather_comms <= allreduce_comms
        ),
        "channel_counts_by_function": [
            {"func": func, "channels": channels, "occurrences": count}
            for (func, channels), count in sorted(channel_counts.items())
        ],
        "start_stop_events": dict(
            sorted(Counter(event.kind for event in clock_events).items())
        ),
        "reuse_by_function": reuse_by_func,
        "pass_count": len(per_pass),
        "passes_matching_expected_function_sequence": sum(
            item["functions_match_expected"] for item in per_pass
        ),
        "passes_with_all_equal_channel_zero_starts": sum(
            item["channel_zero_all_equal"] for item in per_pass
        ),
        "channel_zero_unique_start_clocks_per_pass": [
            item["channel_zero_unique_start_clocks"] for item in per_pass
        ],
        "allreduce_channel_zero_unique_start_clocks_per_pass": [
            item["allreduce_channel_zero_unique_start_clocks"]
            for item in per_pass
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-cell", type=Path, required=True)
    parser.add_argument("--standalone-cell", type=Path, required=True)
    args = parser.parse_args()
    serving = args.serving_cell.resolve(strict=True)
    standalone = args.standalone_cell.resolve(strict=True)
    serving_summary = serving / "ptimer-summary.json"
    standalone_summary = standalone / "score.json"
    outputs = {}
    for label, directory, summary_path, is_serving in (
        ("serving", serving, serving_summary, True),
        ("standalone", standalone, standalone_summary, False),
    ):
        before, after, windows = _load_cell(
            directory, summary_path, serving=is_serving
        )
        outputs[label] = {
            "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            "by_rank": {
                str(rank): _rank_summary(
                    before, after, windows, rank, serving=is_serving
                )
                for rank in (0, 1)
            },
        }
    print(
        json.dumps(
            {
                "schema": "tp-ptimer-offline-graph-compare-v1",
                "role": "post_hoc_non_decisional",
                "raw_identities_published": False,
                "cells": outputs,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
