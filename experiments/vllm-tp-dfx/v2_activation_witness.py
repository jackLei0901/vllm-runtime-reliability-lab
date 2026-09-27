"""Fail-closed reader for private, bounded V2 replay activation witnesses."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

FIELDS = frozenset(
    {
        "schema",
        "install_seen",
        "manager_instances",
        "manager_kind",
        "runner_v2",
        "tp_rank",
        "tp_world_size",
        "configured_graph_mode",
        "breakable_enabled",
        "replay_calls",
        "armed_replay_calls",
        "full_cached_calls",
        "eligible_calls",
        "hold_entered",
    }
)
COUNTS = (
    "manager_instances",
    "replay_calls",
    "armed_replay_calls",
    "full_cached_calls",
    "eligible_calls",
)
MAX_BYTES = 4096


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate activation witness field")
        result[key] = value
    return result


def read_witness(directory: Path, pid: int, expected_rank: int) -> dict[str, object]:
    """Return only closed facts; caller must separately bind PID start ticks."""
    if expected_rank not in (0, 1) or pid <= 0:
        raise ValueError("invalid expected identity")
    folder = directory.stat()
    if not stat.S_ISDIR(folder.st_mode):
        raise ValueError("witness directory is unavailable")
    if hasattr(os, "geteuid") and (
        folder.st_uid != os.geteuid() or stat.S_IMODE(folder.st_mode) != 0o700
    ):
        raise ValueError("witness directory is not owner-only")
    path = directory / f"witness.{pid}.json"
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_BYTES:
        raise ValueError("witness file is unavailable or oversized")
    if hasattr(os, "geteuid") and (
        metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise ValueError("witness file is not owner-only")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode) or opened.st_ino != metadata.st_ino:
            raise ValueError("witness file identity changed")
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("witness file exceeded private budget")
    observed = json.loads(raw.decode("ascii"), object_pairs_hook=_unique_pairs)
    if not isinstance(observed, dict) or set(observed) != FIELDS:
        raise ValueError("activation witness shape mismatch")
    if (
        observed["schema"] != "tp-v2-activation-v1"
        or observed["install_seen"] is not True
        or observed["manager_kind"] != "ModelCudaGraphManager"
        or observed["runner_v2"] is not True
        or type(observed["tp_rank"]) is not int
        or observed["tp_rank"] != expected_rank
        or type(observed["tp_world_size"]) is not int
        or observed["tp_world_size"] != 2
        or observed["configured_graph_mode"]
        not in ("FULL", "FULL_AND_PIECEWISE")
        or observed["breakable_enabled"] is not False
        or type(observed["hold_entered"]) is not bool
    ):
        raise ValueError("activation witness precondition mismatch")
    if any(type(observed[name]) is not int or not 0 <= observed[name] <= 2 for name in COUNTS):
        raise ValueError("activation witness counter invalid")
    if observed["manager_instances"] != 1:
        raise ValueError("activation manager identity ambiguous")
    if observed["hold_entered"] and observed["eligible_calls"] == 0:
        raise ValueError("hold lacks eligible replay")
    if observed["eligible_calls"] and (
        observed["armed_replay_calls"] == 0 or observed["full_cached_calls"] == 0
    ):
        raise ValueError("eligible replay lacks required facts")
    return observed
