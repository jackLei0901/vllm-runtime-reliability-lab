"""Test-only EngineCore hook for the vLLM #53859 Stage 1 experiment."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import queue
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

ENABLE_ENV = "DFX_STAGE1_ENABLE"
CONTROL_DIR_ENV = "DFX_STAGE1_CONTROL_DIR"
OBSERVER_PID_ENV = "DFX_STAGE1_OBSERVER_PID"
PR_SET_PTRACER = 0x59616D61
YAMA_SCOPE = Path("/proc/sys/kernel/yama/ptrace_scope")
POLL_SECONDS = 0.05
PLUGIN_VERSION = 1


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _plugin_sha256() -> str:
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def _kv_events_sha256() -> str:
    import vllm.distributed.kv_events as kv_events

    source = Path(kv_events.__file__).resolve()
    return hashlib.sha256(source.read_bytes()).hexdigest()


def _mapped_worktree_binaries() -> dict[str, str]:
    import vllm

    worktree = Path(vllm.__file__).resolve().parent.parent
    records: dict[str, str] = {}
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return records
    for line in maps.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        mapped = Path(fields[5].removesuffix(" (deleted)")).resolve()
        if ".so" not in mapped.name or not mapped.is_relative_to(worktree):
            continue
        relative = mapped.relative_to(worktree).as_posix()
        records[relative] = hashlib.sha256(mapped.read_bytes()).hexdigest()
    return dict(sorted(records.items()))


def _authorize_observer() -> tuple[str, int | None]:
    if not YAMA_SCOPE.exists():
        return "yama_absent", None
    scope = int(YAMA_SCOPE.read_text(encoding="utf-8").strip())
    raw_pid = os.environ.get(OBSERVER_PID_ENV)
    if raw_pid is None:
        raise RuntimeError(f"{OBSERVER_PID_ENV} is required when Yama is present")
    observer_pid = int(raw_pid)
    if observer_pid <= 0:
        raise RuntimeError(f"{OBSERVER_PID_ENV} must be a positive PID")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_PTRACER, observer_pid, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    return "pr_set_ptracer_observer", scope


def _process_start_time_ticks(pid: int) -> int:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    closing = raw.rfind(")")
    if closing < 0:
        raise RuntimeError("malformed /proc process identity")
    fields_after_comm = raw[closing + 2 :].split()
    return int(fields_after_comm[19])


def _install_queue_counters(publisher: Any, control_dir: Path, pid: int) -> None:
    real_put = publisher._event_queue.put
    state = {"accepted_batch_count": 0, "dropped_batch_count": 0}
    lock = threading.Lock()
    counter_path = control_dir / f"queue-counts-{pid}.json"

    def observed_put(
        item: Any, block: bool = True, timeout: float | None = None
    ) -> None:
        try:
            real_put(item, block=block, timeout=timeout)
        except queue.Full:
            if item is not None and block is False:
                with lock:
                    state["dropped_batch_count"] += 1
                    _atomic_json(counter_path, state)
            raise
        else:
            if item is not None:
                with lock:
                    state["accepted_batch_count"] += 1
                    _atomic_json(counter_path, state)

    publisher._event_queue.put = observed_put
    _atomic_json(counter_path, state)


def _wait_for_release(control_dir: Path) -> None:
    release = control_dir / "release"
    while not release.exists():
        time.sleep(POLL_SECONDS)


def _bounded_hook_error(error: Exception) -> str:
    if isinstance(error, FileExistsError):
        return "duplicate_engine_core"
    if isinstance(error, FileNotFoundError):
        return "missing_file"
    if isinstance(error, PermissionError):
        return "permission_error"
    if isinstance(error, (ValueError, RuntimeError)):
        return "contract_error"
    if isinstance(error, OSError):
        return "os_error"
    return "unexpected_error"


def register() -> None:
    """Patch the publisher thread only when the frozen experiment opts in."""
    if os.environ.get(ENABLE_ENV) != "1":
        return
    raw_control_dir = os.environ.get(CONTROL_DIR_ENV)
    if raw_control_dir is None:
        raise RuntimeError(f"{CONTROL_DIR_ENV} is required")
    control_dir = Path(raw_control_dir).resolve()

    from vllm.distributed.kv_events import ZmqEventPublisher

    if getattr(ZmqEventPublisher, "_dfx_stage1_wrapped", False):
        return

    production_thread = ZmqEventPublisher._publisher_thread

    def controlled_publisher_thread(publisher: Any) -> None:
        pid = os.getpid()
        try:
            start_time_ticks = _process_start_time_ticks(pid)
            authorization, yama_ptrace_scope = _authorize_observer()
            _install_queue_counters(publisher, control_dir, pid)
            _exclusive_json(
                control_dir / f"ready-{pid}.json",
                {
                    "authorization": authorization,
                    "kv_events_sha256": _kv_events_sha256(),
                    "mapped_worktree_binaries": _mapped_worktree_binaries(),
                    "pid": pid,
                    "plugin_sha256": _plugin_sha256(),
                    "plugin_version": PLUGIN_VERSION,
                    "state": "paused",
                    "start_time_ticks": start_time_ticks,
                    "yama_ptrace_scope": yama_ptrace_scope,
                },
            )
            _wait_for_release(control_dir)
            _atomic_json(
                control_dir / f"released-{pid}.json",
                {"pid": pid, "plugin_version": PLUGIN_VERSION},
            )
        except Exception as error:
            try:
                _exclusive_json(
                    control_dir / f"hook-error-{pid}.json",
                    {
                        "error_kind": _bounded_hook_error(error),
                        "pid": pid,
                        "plugin_version": PLUGIN_VERSION,
                    },
                )
            except OSError:
                pass
            return
        production_thread(publisher)

    ZmqEventPublisher._publisher_thread = controlled_publisher_thread
    ZmqEventPublisher._dfx_stage1_wrapped = True
