"""Experiment-only reader for private Inspector callback logs.

Raw communicator hashes, PIDs, paths and NCCL lines never enter the returned
scored record. This evaluates a bounded acquisition experiment, not a fault
verdict or a production export format.
"""

from __future__ import annotations

import glob
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

RANK_LINE = re.compile(
    r"PROFILER/Plugin: init .*\bnranks: 2 rank: (\d+)(?:\s|$)"
)
EVENT_LINE = re.compile(
    r"LLR_TP_EVT (?P<kind>coll_start|kernel_ch_start|kernel_ch_stop) "
    r"comm=(?P<comm>[0-9a-f]{16}) seq=(?P<seq>\d+) "
    r"(?P<field>channels|channel)=(?P<value>\d+)(?:\s|$)"
)
MAX_LOG_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class Event:
    kind: str
    comm: str
    seq: int
    value: int

    @property
    def key(self) -> tuple[str, int]:
        return self.comm, self.seq


def read_rank_logs(pattern: str) -> dict[int, tuple[Event, ...]]:
    """Read only complete debug lines from exactly two uniquely bound logs."""
    paths = sorted(glob.glob(pattern))
    if len(paths) != 2:
        raise ValueError("expected exactly two NCCL debug logs")
    result: dict[int, tuple[Event, ...]] = {}
    for name in paths:
        path = Path(name)
        if path.stat().st_size > MAX_LOG_BYTES:
            raise ValueError("NCCL debug log exceeded private budget")
        content = path.read_text(encoding="utf-8", errors="replace")
        ranks = set(RANK_LINE.findall(content))
        if len(ranks) != 1:
            raise ValueError("NCCL log rank binding unavailable")
        rank = int(next(iter(ranks)))
        if rank in result:
            raise ValueError("duplicate rank-bound NCCL log")
        events = []
        for line in content.splitlines(keepends=True):
            if not line.endswith("\n") or "LLR_TP_EVT" not in line:
                continue  # Ignore an actively written, incomplete final line.
            match = EVENT_LINE.search(line)
            if match is None:
                raise ValueError("malformed Inspector event marker")
            kind, field = match["kind"], match["field"]
            if (kind == "coll_start") != (field == "channels"):
                raise ValueError("Inspector event field mismatch")
            events.append(
                Event(kind, match["comm"], int(match["seq"]), int(match["value"]))
            )
        result[rank] = tuple(events)
    if set(result) != {0, 1}:
        raise ValueError("expected NCCL ranks zero and one")
    return result


def _validate(events: tuple[Event, ...]) -> None:
    channels: dict[tuple[str, int], int] = {}
    started: Counter[tuple[tuple[str, int], int]] = Counter()
    stopped: Counter[tuple[tuple[str, int], int]] = Counter()
    for event in events:
        if event.kind not in {"coll_start", "kernel_ch_start", "kernel_ch_stop"}:
            raise ValueError("unknown Inspector event kind")
        if event.kind == "coll_start":
            if not 1 <= event.value <= 256:
                raise ValueError("invalid channel count")
            previous = channels.setdefault(event.key, event.value)
            if previous != event.value:
                raise ValueError("channel count changed for one collective key")
        else:
            if event.key not in channels or event.value >= channels[event.key]:
                raise ValueError("channel marker lacks its collective descriptor")
            index = (event.key, event.value)
            if event.kind == "kernel_ch_start":
                started[index] += 1
            else:
                stopped[index] += 1
                if stopped[index] > started[index]:
                    raise ValueError("channel stopped before starting")


def _channel_starts(events: tuple[Event, ...], key: tuple[str, int]) -> int:
    return sum(event.kind == "kernel_ch_start" and event.key == key for event in events)


def score_triplet(
    before: dict[int, tuple[Event, ...]],
    during: dict[int, tuple[Event, ...]],
    after: dict[int, tuple[Event, ...]],
    *,
    held_during_snapshot: bool,
    request_completed_after_release: bool,
) -> dict[str, object]:
    """Score one rank-asymmetric start window; never infer a fault cause."""
    if not held_during_snapshot or not request_completed_after_release:
        return {"result": "unscored", "reason": "window_or_release_unverified"}
    if any(set(snapshot) != {0, 1} for snapshot in (before, during, after)):
        return {"result": "unscored", "reason": "rank_binding_unavailable"}
    for rank in (0, 1):
        prefix, middle, last = before[rank], during[rank], after[rank]
        if middle[: len(prefix)] != prefix or last[: len(middle)] != middle:
            return {"result": "unscored", "reason": "log_history_not_contiguous"}
        for events in (prefix, middle, last):
            _validate(events)

    keys = sorted({event.key for events in after.values() for event in events})
    candidates = []
    for ordinal, key in enumerate(keys):
        baseline = [_channel_starts(before[rank], key) for rank in (0, 1)]
        middle = [_channel_starts(during[rank], key) for rank in (0, 1)]
        final = [_channel_starts(after[rank], key) for rank in (0, 1)]
        if baseline[0] != baseline[1]:
            continue  # This key was already asymmetric before the trigger.
        if middle[0] > baseline[0] and middle[1] == baseline[1]:
            if final[0] != final[1] or final[1] < middle[0]:
                return {"result": "unscored", "reason": "rank_did_not_catch_up"}
            candidates.append((ordinal, middle[0] - baseline[0]))
    if len(candidates) != 1:
        return {"result": "unscored", "reason": "target_collective_ambiguous"}
    ordinal, rank_zero_channel_starts = candidates[0]
    return {
        "result": "start_asymmetry_observed",
        "communicator_sequence_ordinal": ordinal,
        "rank_zero_channel_starts_during_hold": rank_zero_channel_starts,
        "rank_one_channel_starts_during_hold": 0,
        "rank_one_caught_up_after_release": True,
    }
