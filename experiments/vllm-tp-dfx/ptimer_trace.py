"""Private, post-run pTimer audit for the v2 Inspector occurrence experiment.

The frozen serving runner and its score are not inputs to this module's decision.
Raw logs, communicator identifiers and GPU clock values stay in the private cell.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from inflight_trace_v2 import Event, read_private_snapshot, validate_history

MAX_LOG_BYTES = 32 * 1024 * 1024
RANK_LINE = re.compile(r"PROFILER/Plugin: init .*\bnranks: 2 rank: (\d+)(?:\s|$)")
CLOCK_LINE = re.compile(
    r"LLR_TP_EVT_V2 (?P<kind>kernel_ch_start|kernel_ch_stop) "
    r"comm=(?P<comm>[0-9a-f]{16}) occurrence=(?P<occurrence>\d+) "
    r"func=(?P<func>[A-Za-z][A-Za-z0-9_]*) seq=(?P<seq>\d+) "
    r"channel=(?P<channel>\d+) ptimer=(?P<clock>\d+)(?:\s|$)"
)


@dataclass(frozen=True)
class ClockEvent:
    kind: str
    comm: str
    occurrence: int
    func: str
    seq: int
    channel: int
    clock: int

    @property
    def key(self) -> tuple[str, str, int, str, int, int]:
        return (
            self.kind,
            self.comm,
            self.occurrence,
            self.func,
            self.seq,
            self.channel,
        )


def read_private_clocks(pattern: str) -> tuple[dict[int, dict[tuple, ClockEvent]], dict[int, str]]:
    """Require one complete rank-bound log per rank and unique clock markers."""
    paths = sorted(glob.glob(pattern))
    if len(paths) != 2:
        raise ValueError("expected exactly two rank logs")
    result: dict[int, dict[tuple, ClockEvent]] = {}
    digests: dict[int, str] = {}
    for name in paths:
        path = Path(name)
        if path.stat().st_size > MAX_LOG_BYTES:
            raise ValueError("private NCCL log exceeds budget")
        payload = path.read_bytes()
        if len(payload) > MAX_LOG_BYTES:
            raise ValueError("private NCCL log grew beyond budget")
        text = payload.decode("utf-8", errors="replace")
        ranks = set(RANK_LINE.findall(text))
        if len(ranks) != 1:
            raise ValueError("rank binding unavailable")
        rank = int(next(iter(ranks)))
        if rank not in (0, 1) or rank in result:
            raise ValueError("duplicate or unexpected rank")
        events: dict[tuple, ClockEvent] = {}
        for line in text.splitlines(keepends=True):
            if not line.endswith("\n"):
                continue
            if "LLR_TP_EVT_V2 kernel_ch_" not in line:
                continue
            match = CLOCK_LINE.search(line)
            if match is None:
                raise ValueError("missing or malformed pTimer marker")
            event = ClockEvent(
                match["kind"],
                match["comm"],
                int(match["occurrence"]),
                match["func"],
                int(match["seq"]),
                int(match["channel"]),
                int(match["clock"]),
            )
            if event.key in events:
                raise ValueError("duplicate pTimer marker")
            events[event.key] = event
        result[rank] = events
        digests[rank] = hashlib.sha256(payload).hexdigest()
    if set(result) != {0, 1}:
        raise ValueError("both ranks are required")
    return result, digests


def _marker_key(event: Event) -> tuple[str, str, int, str, int, int]:
    return (
        event.kind,
        event.comm,
        event.occurrence,
        event.func,
        event.seq,
        event.value,
    )


def _summarize_rank(events: list[ClockEvent]) -> dict[str, object]:
    """Keep clock comparisons inside a communicator/channel; publish no keys."""
    by_kind = {}
    for kind in ("kernel_ch_start", "kernel_ch_stop"):
        selected = [event for event in events if event.kind == kind]
        groups: dict[tuple[str, int], list[ClockEvent]] = {}
        for event in selected:
            groups.setdefault((event.comm, event.channel), []).append(event)
        largest_class = 0
        repeated_groups = 0
        nonincreasing_pairs = 0
        for members in groups.values():
            classes = Counter(event.clock for event in members)
            largest_class = max(largest_class, max(classes.values(), default=0))
            repeated_groups += max(classes.values(), default=0) > 1
            ordered = sorted(members, key=lambda event: event.occurrence)
            nonincreasing_pairs += sum(
                later.clock <= earlier.clock
                for earlier, later in zip(ordered, ordered[1:])
            )
        by_kind[kind] = {
            "events": len(selected),
            "zero_clocks": sum(event.clock == 0 for event in selected),
            "comm_channel_groups": len(groups),
            "largest_group_event_count": max(map(len, groups.values()), default=0),
            "groups_with_at_least_six_events": sum(
                len(members) >= 6 for members in groups.values()
            ),
            "groups_with_repeated_clock": repeated_groups,
            "largest_equal_value_class_within_group": largest_class,
            "nonincreasing_adjacent_pairs": nonincreasing_pairs,
        }
    starts = {
        (event.comm, event.occurrence, event.channel): event.clock
        for event in events
        if event.kind == "kernel_ch_start"
    }
    stops = {
        (event.comm, event.occurrence, event.channel): event.clock
        for event in events
        if event.kind == "kernel_ch_stop"
    }
    paired = starts.keys() & stops.keys()
    paired_by_group = Counter((comm, channel) for comm, _, channel in paired)
    return {
        "by_kind": by_kind,
        "paired_channels_in_window": len(paired),
        "largest_paired_group_event_count": max(paired_by_group.values(), default=0),
        "nonpositive_pair_duration": sum(stops[key] <= starts[key] for key in paired),
    }


def summarize_window(
    before: dict[int, tuple[Event, ...]],
    end: dict[int, tuple[Event, ...]],
    clocks: dict[int, dict[tuple, ClockEvent]],
) -> dict[str, object]:
    """Summarize exact snapshot-delta markers; never emit clock values or keys."""
    if set(before) != {0, 1} or set(end) != {0, 1} or set(clocks) != {0, 1}:
        raise ValueError("both ranks are required")
    result: dict[str, object] = {}
    for rank in (0, 1):
        validate_history(before[rank])
        validate_history(end[rank])
        if end[rank][: len(before[rank])] != before[rank]:
            raise ValueError("snapshot history is not a prefix")
        new_events = [
            event
            for event in end[rank][len(before[rank]) :]
            if event.kind in {"kernel_ch_start", "kernel_ch_stop"}
        ]
        selected: list[ClockEvent] = []
        for event in new_events:
            marker = clocks[rank].get(_marker_key(event))
            if marker is None:
                raise ValueError("snapshot event lacks a pTimer marker")
            selected.append(marker)
        result[str(rank)] = _summarize_rank(selected)
    return result


def summarize_all_clocks(clocks: dict[int, dict[tuple, ClockEvent]]) -> dict[str, object]:
    """Instrument sanity check; this does not identify a serving window."""
    if set(clocks) != {0, 1}:
        raise ValueError("both ranks are required")
    result: dict[str, object] = {}
    for rank in (0, 1):
        result[str(rank)] = _summarize_rank(list(clocks[rank].values()))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-dir", required=True, type=Path)
    parser.add_argument("--end-label", required=True, choices=("all", "during", "after"))
    parser.add_argument("--before-sha256")
    parser.add_argument("--end-sha256")
    args = parser.parse_args()
    if args.end_label == "all":
        if args.before_sha256 is not None or args.end_sha256 is not None:
            parser.error("all-logs mode does not take snapshot digests")
    else:
        for value in (args.before_sha256, args.end_sha256):
            if value is None or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                parser.error("window mode requires both snapshot SHA-256 digests")
    private = args.private_dir.resolve(strict=True)
    clocks, digests = read_private_clocks(str(private / "nccl.*.log"))
    if args.end_label == "all":
        summary = summarize_all_clocks(clocks)
    else:
        before = read_private_snapshot(
            private / "callbacks-before.json", expected_sha256=args.before_sha256
        )
        end = read_private_snapshot(
            private / f"callbacks-{args.end_label}.json", expected_sha256=args.end_sha256
        )
        summary = summarize_window(before, end, clocks)
    print(
        json.dumps(
            {
                "schema": "tp-ptimer-equality-v1",
                "end_label": args.end_label,
                "before_sha256": args.before_sha256,
                "end_sha256": args.end_sha256,
                "rank_log_sha256": {str(k): v for k, v in digests.items()},
                "by_rank": summary,
                "device_work_identity_proven": False,
                "verdict_changed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
