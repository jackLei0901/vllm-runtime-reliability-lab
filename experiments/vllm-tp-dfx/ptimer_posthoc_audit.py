"""Post-hoc, privacy-safe clock audit of a frozen Inspector cell.

This module does not revise the preregistered occurrence score. Raw clocks,
communicator hashes, file paths and PIDs remain in the private directory.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from inflight_trace_v2 import Event, read_private_snapshot, validate_history
from ptimer_trace import ClockEvent, read_private_clocks


def _marker_key(event: Event) -> tuple[str, str, int, str, int, int]:
    return (
        event.kind,
        event.comm,
        event.occurrence,
        event.func,
        event.seq,
        event.value,
    )


def _window_clocks(
    before: dict[int, tuple[Event, ...]],
    end: dict[int, tuple[Event, ...]],
    clocks: dict[int, dict[tuple, ClockEvent]],
) -> dict[int, tuple[list[ClockEvent], list[ClockEvent]]]:
    if set(before) != {0, 1} or set(end) != {0, 1} or set(clocks) != {0, 1}:
        raise ValueError("both ranks are required")
    selected = {}
    for rank in (0, 1):
        validate_history(before[rank])
        validate_history(end[rank])
        if end[rank][: len(before[rank])] != before[rank]:
            raise ValueError("snapshot history is not a prefix")
        portions = []
        for events in (before[rank], end[rank][len(before[rank]) :]):
            markers = []
            for event in events:
                if event.kind not in {"kernel_ch_start", "kernel_ch_stop"}:
                    continue
                marker = clocks[rank].get(_marker_key(event))
                if marker is None:
                    raise ValueError("snapshot event lacks a pTimer marker")
                markers.append(marker)
            portions.append(markers)
        selected[rank] = (portions[0], portions[1])
    return selected


def audit_windows(
    before: dict[int, tuple[Event, ...]],
    end: dict[int, tuple[Event, ...]],
    clocks: dict[int, dict[tuple, ClockEvent]],
) -> dict[str, object]:
    """Count distinct-occurrence clock reuse and lower-than-history values."""
    windows = _window_clocks(before, end, clocks)
    by_rank = {}
    for rank, (history, current) in windows.items():
        # Keep function in the key so an unrelated collective cannot explain
        # an equal-clock class attributed to ordinary AllReduce calls.
        groups: dict[tuple[str, int, str, str], list[ClockEvent]] = defaultdict(list)
        history_max: dict[tuple[str, int, str], int] = {}
        for event in history:
            group = (event.comm, event.channel, event.kind)
            history_max[group] = max(history_max.get(group, 0), event.clock)
        lower_than_history = 0
        history_comparable = 0
        for event in current:
            groups[(event.comm, event.channel, event.func, event.kind)].append(event)
            prior = history_max.get((event.comm, event.channel, event.kind))
            if prior is not None:
                history_comparable += 1
                lower_than_history += event.clock < prior
        by_kind: dict[str, dict[str, int]] = {}
        for kind in ("kernel_ch_start", "kernel_ch_stop"):
            selected = [members for key, members in groups.items() if key[3] == kind]
            largest_equal_class = 0
            repeated_occurrences = 0
            groups_with_reuse = 0
            for members in selected:
                clock_to_occurrences: dict[int, set[int]] = defaultdict(set)
                for event in members:
                    clock_to_occurrences[event.clock].add(event.occurrence)
                classes = [len(ids) for ids in clock_to_occurrences.values()]
                largest_equal_class = max(largest_equal_class, max(classes, default=0))
                repeated_occurrences += sum(size for size in classes if size > 1)
                groups_with_reuse += any(size > 1 for size in classes)
            by_kind[kind] = {
                "events": sum(map(len, selected)),
                "groups": len(selected),
                "largest_equal_clock_distinct_occurrences": largest_equal_class,
                "occurrences_in_repeated_classes": repeated_occurrences,
                "groups_with_reuse": groups_with_reuse,
            }
        by_rank[str(rank)] = {
            "by_kind": by_kind,
            "history_comparable_events": history_comparable,
            "clocks_below_pre_window_max": lower_than_history,
        }
    return {
        "schema": "tp-ptimer-posthoc-v1",
        "analysis_role": "post_hoc_not_preregistered",
        "by_rank": by_rank,
        "adjacent_inversions_scored": False,
        "device_work_identity_proven": False,
        "verdict_changed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-dir", required=True, type=Path)
    parser.add_argument("--end-label", required=True, choices=("during", "after"))
    parser.add_argument("--before-sha256", required=True)
    parser.add_argument("--end-sha256", required=True)
    args = parser.parse_args()
    for digest in (args.before_sha256, args.end_sha256):
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            parser.error("both snapshot digests must be lowercase SHA-256")
    private = args.private_dir.resolve(strict=True)
    before = read_private_snapshot(
        private / "callbacks-before.json", expected_sha256=args.before_sha256
    )
    end = read_private_snapshot(
        private / f"callbacks-{args.end_label}.json", expected_sha256=args.end_sha256
    )
    clocks, digests = read_private_clocks(str(private / "nccl.*.log"))
    result = audit_windows(before, end, clocks)
    result.update(
        {
            "end_label": args.end_label,
            "before_sha256": args.before_sha256,
            "end_sha256": args.end_sha256,
            "rank_log_sha256": {str(rank): digest for rank, digest in digests.items()},
        }
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
