"""Experiment-only, one-shot host hold in GPU Worker.execute_model.

The hold is entered only after an external arm marker exists. It changes no
GPU or collective implementation and must not be installed in production.
"""

from __future__ import annotations

import json
import os
import stat
import threading
import time
from pathlib import Path
from typing import Any

_lock = threading.Lock()
_installed = False
_claimed = False


def eligible(*, armed: bool, claimed: bool, rank: int, tp: int,
             expected_tp: int, scheduled_tokens: int) -> bool:
    return (
        armed
        and not claimed
        and rank == 0
        and tp == expected_tp
        and scheduled_tokens > 0
    )


def _start_ticks() -> int:
    fields = Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()
    return int(fields[19])  # field 22, after pid and parenthesized comm


def _write_marker(directory: Path, name: str, payload: dict[str, Any]) -> None:
    target = directory / name
    temporary = directory / f".{name}.{os.getpid()}.tmp"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="ascii") as stream:
        json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if target.exists():
        raise RuntimeError(f"K5 marker already exists: {name}")
    os.replace(temporary, target)


def install() -> None:
    global _installed
    if _installed:
        return
    raw_dir = os.environ.get("LLR_K5_PRIVATE_DIR")
    if not raw_dir or not os.path.isabs(raw_dir):
        raise RuntimeError("LLR_K5_PRIVATE_DIR must be absolute")
    directory = Path(raw_dir)
    metadata = directory.stat()
    if (not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700):
        raise RuntimeError("K5 private directory must be owner-only")
    expected_tp = int(os.environ["LLR_K5_TP_SIZE"])
    hold_seconds = float(os.environ["LLR_K5_HOLD_SECONDS"])
    if expected_tp not in (1, 2) or not 40 <= hold_seconds <= 50:
        raise RuntimeError("K5 TP or hold duration is outside the protocol")

    from vllm.v1.worker.gpu_worker import Worker

    original = Worker.execute_model

    def reset_after_fork() -> None:
        global _lock, _claimed
        _lock = threading.Lock()
        _claimed = False

    def wrapped(self: Any, scheduler_output: Any) -> Any:
        global _claimed
        rank = int(self.rank)
        tp = int(self.vllm_config.parallel_config.tensor_parallel_size)
        scheduled = int(scheduler_output.total_num_scheduled_tokens)
        with _lock:
            should_hold = eligible(
                armed=(directory / "arm").is_file(),
                claimed=_claimed,
                rank=rank,
                tp=tp,
                expected_tp=expected_tp,
                scheduled_tokens=scheduled,
            )
            if should_hold:
                _claimed = True
        if should_hold:
            identity = {
                "schema": "llr-k5-worker-hold-v1",
                "pid": os.getpid(),
                "start_ticks": _start_ticks(),
                "rank": rank,
                "tp": tp,
                "scheduled_tokens_positive": True,
                "monotonic_ns": time.monotonic_ns(),
            }
            _write_marker(directory, "entered.json", identity)
            time.sleep(hold_seconds)
            _write_marker(directory, "released.json", {
                **identity, "monotonic_ns": time.monotonic_ns()
            })
        return original(self, scheduler_output)

    Worker.execute_model = wrapped
    if hasattr(os, "register_at_fork"):
        os.register_at_fork(after_in_child=reset_after_fork)
    _installed = True
