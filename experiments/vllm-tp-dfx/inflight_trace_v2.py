"""Experiment-only reader for private Inspector callback logs.

Raw communicator hashes, PIDs, paths and NCCL lines never enter the returned
scored record. This evaluates a bounded acquisition experiment, not a fault
verdict or a production export format.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

RANK_LINE = re.compile(
    r"PROFILER/Plugin: init .*\bnranks: 2 rank: (\d+)(?:\s|$)"
)
EVENT_LINE = re.compile(
    r"LLR_TP_EVT_V2 (?P<kind>coll_start|kernel_ch_start|kernel_ch_stop) "
    r"comm=(?P<comm>[0-9a-f]{16}) occurrence=(?P<occurrence>\d+) "
    r"func=(?P<func>[A-Za-z][A-Za-z0-9_]*) seq=(?P<seq>\d+) "
    r"(?P<field>channels|channel)=(?P<value>\d+)(?:\s|$)"
)
MAX_LOG_BYTES = 32 * 1024 * 1024
MAX_SNAPSHOT_BYTES = 64 * 1024 * 1024
NCCL_FUNCS = frozenset(
    {"Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllReduce", "SendRecv", "Send", "Recv"}
)
SERVING_STEP_FUNCS = ("AllReduce",) * 73 + ("AllGather",)


@dataclass(frozen=True)
class Event:
    kind: str
    comm: str
    occurrence: int
    func: str
    seq: int
    value: int

    @property
    def key(self) -> tuple[str, int]:
        return self.comm, self.occurrence


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
                Event(
                    kind,
                    match["comm"],
                    int(match["occurrence"]),
                    match["func"],
                    int(match["seq"]),
                    int(match["value"]),
                )
            )
        result[rank] = tuple(events)
    if set(result) != {0, 1}:
        raise ValueError("expected NCCL ranks zero and one")
    return result


def validate_history(
    events: tuple[Event, ...],
) -> dict[tuple[str, int], tuple[str, int, int]]:
    """Require each plugin occurrence to bind exactly one descriptor."""
    descriptors: dict[tuple[str, int], tuple[str, int, int]] = {}
    last_occurrence: dict[str, int] = {}
    started: Counter[tuple[tuple[str, int], int]] = Counter()
    stopped: Counter[tuple[tuple[str, int], int]] = Counter()
    for event in events:
        if event.kind not in {"coll_start", "kernel_ch_start", "kernel_ch_stop"}:
            raise ValueError("unknown Inspector event kind")
        if event.func not in NCCL_FUNCS:
            raise ValueError("unknown collective function")
        if event.kind == "coll_start":
            if not 1 <= event.value <= 256:
                raise ValueError("invalid channel count")
            if event.key in descriptors:
                raise ValueError("duplicate collective occurrence")
            if event.occurrence != last_occurrence.get(event.comm, 0) + 1:
                raise ValueError("collective occurrence is not contiguous")
            last_occurrence[event.comm] = event.occurrence
            descriptors[event.key] = (event.func, event.seq, event.value)
        else:
            descriptor = descriptors.get(event.key)
            if (
                descriptor is None
                or (event.func, event.seq) != descriptor[:2]
                or event.value >= descriptor[2]
            ):
                raise ValueError("channel marker lacks its collective descriptor")
            index = (event.key, event.value)
            if event.kind == "kernel_ch_start":
                started[index] += 1
                if started[index] > 1:
                    raise ValueError("duplicate channel start")
            else:
                stopped[index] += 1
                if stopped[index] > started[index]:
                    raise ValueError("channel stopped before starting")
    return descriptors


def validate_rank_pair(snapshot: dict[int, tuple[Event, ...]]) -> None:
    if set(snapshot) != {0, 1}:
        raise ValueError("rank binding unavailable")
    descriptors = {rank: validate_history(snapshot[rank]) for rank in (0, 1)}
    if descriptors[0].keys() != descriptors[1].keys():
        raise ValueError("rank occurrence sets disagree")
    for key in descriptors[0]:
        if descriptors[0][key] != descriptors[1][key]:
            raise ValueError("rank descriptors disagree")


def validate_control_history(
    before: dict[int, tuple[Event, ...]],
    after: dict[int, tuple[Event, ...]],
) -> None:
    validate_rank_pair(before)
    validate_rank_pair(after)
    for rank in (0, 1):
        if after[rank][: len(before[rank])] != before[rank]:
            raise ValueError("control callback history is not contiguous")


def save_private_snapshot(
    directory: Path, label: str, snapshot: dict[int, tuple[Event, ...]]
) -> str:
    """Persist one exact, private callback snapshot and return only its digest."""
    if label not in {"before", "during", "after"} or set(snapshot) != {0, 1}:
        raise ValueError("invalid private snapshot")
    destination = directory / f"callbacks-{label}.json"
    if destination.exists():
        raise ValueError("private snapshot already exists")
    payload = json.dumps(
        {
            "schema": "tp-inflight-callbacks-v2",
            "ranks": {
                str(rank): [asdict(event) for event in snapshot[rank]]
                for rank in (0, 1)
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(payload) > MAX_SNAPSHOT_BYTES:
        raise ValueError("private snapshot exceeded budget")
    with tempfile.NamedTemporaryFile(
        dir=directory, prefix=".callbacks-", delete=False
    ) as handle:
        temporary = Path(handle.name)
        os.chmod(temporary, 0o600)
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, destination)
    return hashlib.sha256(payload).hexdigest()


def read_private_snapshot(
    path: Path, *, expected_sha256: str
) -> dict[int, tuple[Event, ...]]:
    if path.stat().st_size > MAX_SNAPSHOT_BYTES:
        raise ValueError("private snapshot exceeded budget")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("private snapshot digest mismatch")
    data = json.loads(payload)
    if data.get("schema") != "tp-inflight-callbacks-v2" or set(
        data.get("ranks", {})
    ) != {"0", "1"}:
        raise ValueError("private snapshot schema mismatch")
    result = {
        rank: tuple(Event(**item) for item in data["ranks"][str(rank)])
        for rank in (0, 1)
    }
    return result


def _channel_starts(events: tuple[Event, ...], key: tuple[str, int]) -> int:
    return sum(event.kind == "kernel_ch_start" and event.key == key for event in events)


def _request_step_metadata(
    before: dict[int, tuple[Event, ...]],
    after: dict[int, tuple[Event, ...]],
    candidate_key: tuple[str, int],
    descriptors: dict[tuple[str, int], tuple[str, int, int]],
    rank_zero_channel_starts: int,
) -> dict[str, object]:
    request = {
        rank: tuple(
            event
            for event in after[rank][len(before[rank]) :]
            if event.kind == "coll_start"
        )
        for rank in (0, 1)
    }
    rank_zero = request[0]
    fields: dict[str, object] = {
        "occurrences_after_observe": {rank: len(request[rank]) for rank in (0, 1)},
        "step_structure_verified": False,
        "step_index": None,
        "position_within_step": None,
        "predicted_hold_match": None,
    }
    if (
        len(rank_zero) != 16 * len(SERVING_STEP_FUNCS)
        or tuple(event.key for event in rank_zero)
        != tuple(event.key for event in request[1])
        or any(
            tuple(event.func for event in rank_zero[offset : offset + 74])
            != SERVING_STEP_FUNCS
            for offset in range(0, len(rank_zero), 74)
        )
    ):
        return fields
    positions = [index for index, event in enumerate(rank_zero) if event.key == candidate_key]
    if len(positions) != 1:
        return fields
    index = positions[0]
    decode_channels = {
        descriptors[rank_zero[step * 74].key][2] for step in range(1, 16)
    }
    fields["step_structure_verified"] = True
    fields["step_index"] = index // 74 + 1
    fields["position_within_step"] = index % 74 + 1
    if len(decode_channels) == 1:
        fields["predicted_hold_match"] = (
            fields["step_index"] == 2
            and fields["position_within_step"] == 1
            and descriptors[candidate_key][0] == "AllReduce"
            and rank_zero_channel_starts == next(iter(decode_channels))
        )
    return fields


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
            validate_history(events)

    validate_rank_pair(before)
    final_descriptors = {rank: validate_history(after[rank]) for rank in (0, 1)}
    if final_descriptors[0].keys() != final_descriptors[1].keys():
        return {"result": "unscored", "reason": "rank_occurrence_mismatch"}
    if final_descriptors[0] != final_descriptors[1]:
        return {"result": "unscored", "reason": "rank_descriptor_mismatch"}

    keys = {event.key for events in after.values() for event in events}
    candidates = []
    for key in keys:
        baseline = [_channel_starts(before[rank], key) for rank in (0, 1)]
        middle = [_channel_starts(during[rank], key) for rank in (0, 1)]
        final = [_channel_starts(after[rank], key) for rank in (0, 1)]
        if baseline[0] != baseline[1]:
            continue  # This key was already asymmetric before the trigger.
        if middle[0] > baseline[0] and middle[1] == baseline[1]:
            if final[0] != final[1] or final[1] < middle[0]:
                return {"result": "unscored", "reason": "rank_did_not_catch_up"}
            candidates.append((key, middle[0] - baseline[0]))
    if len(candidates) != 1:
        return {"result": "unscored", "reason": "target_collective_ambiguous"}
    candidate_key, rank_zero_channel_starts = candidates[0]
    return {
        "result": "start_asymmetry_observed",
        "func": final_descriptors[0][candidate_key][0],
        **_request_step_metadata(
            before,
            after,
            candidate_key,
            final_descriptors[0],
            rank_zero_channel_starts,
        ),
        "rank_zero_channel_starts_during_hold": rank_zero_channel_starts,
        "rank_one_channel_starts_during_hold": 0,
        "rank_one_caught_up_after_release": True,
    }
