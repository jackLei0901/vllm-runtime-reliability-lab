#!/usr/bin/env python3
"""Pure-Python contract helpers for the vLLM #53859 Stage 1 campaign."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STALL_SECONDS = 10.0
CONTROL_MAX_GAP_SECONDS = 2.0
RECOVERY_SECONDS = 30.0
STACK_TOKENS_IN_SAMPLE_ORDER = (
    "wait (threading.py",
    "put (queue.py",
    "publish (vllm/distributed/kv_events.py",
)


@dataclass
class ProgressTrace:
    """Retain counts and monotonic offsets without retaining response content."""

    started_at: float
    offsets: list[float] = field(default_factory=list)
    completed_at: float | None = None

    def observe(self, now: float) -> None:
        if now < self.started_at:
            raise ValueError("progress timestamp precedes request start")
        offset = now - self.started_at
        if self.offsets and offset < self.offsets[-1]:
            raise ValueError("progress timestamps must be monotonic")
        self.offsets.append(offset)

    def complete(self, now: float) -> None:
        if now < self.started_at:
            raise ValueError("completion timestamp precedes request start")
        self.completed_at = now - self.started_at

    def is_open(self) -> bool:
        return self.completed_at is None

    def stalled(self, now: float) -> bool:
        if not self.is_open():
            return False
        if len(self.offsets) < 2:
            return False
        last = self.offsets[-1] if self.offsets else 0.0
        return now - self.started_at - last >= STALL_SECONDS

    def maximum_gap(self) -> float | None:
        if len(self.offsets) < 2:
            return None
        return max(b - a for a, b in zip(self.offsets, self.offsets[1:], strict=False))

    def maximum_no_progress_gap(self) -> float | None:
        """Include request start and completion in the no-progress bound."""
        if self.completed_at is None or not self.offsets:
            return None
        boundaries = [0.0, *self.offsets, self.completed_at]
        return max(b - a for a, b in zip(boundaries, boundaries[1:], strict=False))

    def public_record(self) -> dict[str, Any]:
        return {
            "completed_offset_seconds": self.completed_at,
            "progress_count": len(self.offsets),
            "progress_offsets_seconds": self.offsets,
        }


def load_ready(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if set(record) != {
        "authorization",
        "pid",
        "plugin_sha256",
        "plugin_version",
        "kv_events_sha256",
        "start_time_ticks",
        "state",
        "yama_ptrace_scope",
    }:
        raise ValueError("unexpected EngineCore ready fields")
    if not isinstance(record["pid"], int) or record["pid"] <= 1:
        raise ValueError("invalid EngineCore PID")
    if record["plugin_version"] != 1 or record["state"] != "paused":
        raise ValueError("unexpected hook identity or state")
    for name in ("plugin_sha256", "kv_events_sha256"):
        value = record[name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"invalid {name}")
    if (
        not isinstance(record["start_time_ticks"], int)
        or isinstance(record["start_time_ticks"], bool)
        or record["start_time_ticks"] <= 0
    ):
        raise ValueError("invalid EngineCore start time")
    if record["authorization"] not in {
        "pr_set_ptracer_observer",
        "yama_absent",
    }:
        raise ValueError("unexpected observer authorization")
    scope = record["yama_ptrace_scope"]
    if record["authorization"] == "yama_absent":
        if scope is not None:
            raise ValueError("Yama-absent authorization has a scope")
    elif not isinstance(scope, int) or isinstance(scope, bool) or not 0 <= scope <= 3:
        raise ValueError("invalid Yama ptrace scope")
    return record


def load_queue_counts(path: Path) -> dict[str, int]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if set(record) != {"accepted_batch_count", "dropped_batch_count"}:
        raise ValueError("unexpected queue-count fields")
    for value in record.values():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("invalid queue count")
    return record


def load_hook_error(path: Path) -> str:
    record = json.loads(path.read_text(encoding="utf-8"))
    if set(record) != {"error_kind", "pid", "plugin_version"}:
        raise ValueError("unexpected hook-error fields")
    if not isinstance(record["pid"], int) or record["pid"] <= 1:
        raise ValueError("invalid hook-error PID")
    if record["plugin_version"] != 1:
        raise ValueError("unexpected hook-error plugin version")
    kind = record["error_kind"]
    allowed = {
        "contract_error",
        "duplicate_engine_core",
        "missing_file",
        "os_error",
        "permission_error",
        "unexpected_error",
    }
    if kind not in allowed:
        raise ValueError("unexpected hook error kind")
    return kind


def stack_matches(raw_stack: str) -> bool:
    """Require the current frame and callers in py-spy dump order."""
    cursor = 0
    for token in STACK_TOKENS_IN_SAMPLE_ORDER:
        position = raw_stack.find(token, cursor)
        if position < 0:
            return False
        cursor = position + len(token)
    return True


def classify_cell(
    *,
    source_arm: str,
    trigger: str,
    trace: ProgressTrace,
    stalled_before_release: bool,
    stack_match: bool | None,
    recovered_after_release: bool | None,
    accepted_batch_count: int,
    dropped_batch_count: int,
) -> str:
    """Score only the four pre-registered Stage 1 cells."""
    if source_arm not in {"base", "fix"} or trigger not in {"control", "pause"}:
        raise ValueError("unknown matrix cell")
    if trigger == "control":
        maximum_gap = trace.maximum_gap()
        eligible = maximum_gap is not None and maximum_gap < CONTROL_MAX_GAP_SECONDS
        return (
            "pass"
            if trace.completed_at is not None
            and eligible
            and not stalled_before_release
            and accepted_batch_count >= 1
            and dropped_batch_count == 0
            else "control_failed"
        )
    if source_arm == "base":
        return (
            "pass"
            if stalled_before_release
            and stack_match is True
            and recovered_after_release is True
            and trace.completed_at is not None
            and accepted_batch_count >= 1
            and dropped_batch_count == 0
            else "mechanism_mismatch"
        )
    return (
        "pass"
        if trace.completed_at is not None
        and not stalled_before_release
        and trace.maximum_no_progress_gap() is not None
        and trace.maximum_no_progress_gap() < STALL_SECONDS
        and accepted_batch_count >= 1
        and dropped_batch_count > 0
        else "mechanism_mismatch"
    )
