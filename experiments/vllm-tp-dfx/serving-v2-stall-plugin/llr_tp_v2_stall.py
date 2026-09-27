"""Experiment-only one-shot hold before V2's cached FULL graph replay.

This is not a vLLM/NCCL product probe. It emits only bounded, private
activation facts so a missing hold marker has a typed explanation.
"""

from __future__ import annotations

import json
import os
import stat
import threading
import time
from typing import Any

_lock = threading.Lock()
_installed = False
_held = False
_manager_identity: int | None = None
_witness_path: str | None = None
_state: dict[str, Any] = {
    "schema": "tp-v2-activation-v1",
    "install_seen": False,
    "manager_instances": 0,
    "manager_kind": None,
    "runner_v2": None,
    "tp_rank": None,
    "tp_world_size": None,
    "configured_graph_mode": None,
    "breakable_enabled": None,
    "replay_calls": 0,
    "armed_replay_calls": 0,
    "full_cached_calls": 0,
    "eligible_calls": 0,
    "hold_entered": False,
}


def eligible_v2(
    *,
    armed: bool,
    held: bool,
    manager_unique: bool,
    runner_v2: bool,
    tp_rank: int,
    tp_world_size: int,
    mode: str,
    graph_cached: bool,
    breakable_enabled: bool,
) -> bool:
    """Closed precondition for the experiment's single host-side delay."""
    return (
        armed
        and not held
        and manager_unique
        and runner_v2
        and tp_rank == 1
        and tp_world_size == 2
        and mode == "FULL"
        and graph_cached
        and not breakable_enabled
    )


def _bump(name: str) -> bool:
    """Saturate at 2: the persisted value 2 means two or more calls."""
    old = _state[name]
    _state[name] = min(2, old + 1)
    return _state[name] != old


def _write_witness() -> None:
    assert _witness_path is not None
    temporary = f"{_witness_path}.tmp"
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="ascii") as stream:
        json.dump(_state, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
    os.replace(temporary, _witness_path)


def _validate_paths(arm: str | None, entered: str | None, witness_dir: str | None) -> None:
    if not arm or not entered or not witness_dir:
        raise RuntimeError("arm, entered and witness paths are required")
    if not all(os.path.isabs(path) for path in (arm, entered, witness_dir)):
        raise RuntimeError("experiment paths must be absolute")
    if os.path.dirname(arm) != os.path.dirname(entered):
        raise RuntimeError("arm and entered markers must share a private directory")
    if os.path.dirname(witness_dir) != os.path.dirname(arm):
        raise RuntimeError("witness directory must be under the same private root")
    metadata = os.stat(witness_dir)
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError("witness target is not a directory")
    if hasattr(os, "geteuid") and (
        metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise RuntimeError("witness directory must be owner-only")


def install() -> None:
    """Install in all vLLM processes; only V2 managers can arm the hold."""
    global _installed, _held, _manager_identity, _witness_path
    if _installed:
        return

    arm = os.environ.get("LLR_TP_ARM_FILE")
    entered = os.environ.get("LLR_TP_ENTER_FILE")
    witness_dir = os.environ.get("LLR_TP_WITNESS_DIR")
    _validate_paths(arm, entered, witness_dir)
    assert arm is not None and entered is not None and witness_dir is not None
    hold_seconds = float(os.environ.get("LLR_TP_HOLD_SECONDS", "3"))
    if not 2 <= hold_seconds <= 5:
        raise RuntimeError("hold duration must be between 2 and 5 seconds")
    if os.environ.get("VLLM_USE_BREAKABLE_CUDAGRAPH") != "0":
        raise RuntimeError("breakable CUDA graphs must be explicitly disabled")

    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
    from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager

    original_init = ModelCudaGraphManager.__init__
    original_replay = ModelCudaGraphManager.run_fullgraph
    _witness_path = os.path.join(witness_dir, f"witness.{os.getpid()}.json")
    if os.path.exists(_witness_path):
        raise RuntimeError("witness path must be fresh")
    _state["install_seen"] = True
    _write_witness()

    def init_wrapped(self: Any, *args: Any, **kwargs: Any) -> None:
        global _manager_identity
        original_init(self, *args, **kwargs)
        with _lock:
            _bump("manager_instances")
            _manager_identity = id(self) if _state["manager_instances"] == 1 else None
            _state.update(
                manager_kind="ModelCudaGraphManager",
                runner_v2=bool(self.vllm_config.use_v2_model_runner),
                tp_rank=get_tensor_model_parallel_rank(),
                tp_world_size=get_tensor_model_parallel_world_size(),
                configured_graph_mode=self.cudagraph_mode.name,
                breakable_enabled=bool(self.use_breakable_cg),
            )
            _write_witness()

    def replay_wrapped(self: Any, desc: Any) -> Any:
        global _held
        mode = desc.cg_mode.name
        cached = desc in self.graphs and self.graphs[desc] is not None
        armed = os.path.isfile(arm)
        with _lock:
            changed = _bump("replay_calls")
            if armed:
                changed = _bump("armed_replay_calls") or changed
            if mode == "FULL" and cached:
                changed = _bump("full_cached_calls") or changed
            allowed = eligible_v2(
                armed=armed,
                held=_held,
                manager_unique=_state["manager_instances"] == 1
                and _manager_identity == id(self),
                runner_v2=_state["runner_v2"] is True,
                tp_rank=_state["tp_rank"],
                tp_world_size=_state["tp_world_size"],
                mode=mode,
                graph_cached=cached,
                breakable_enabled=_state["breakable_enabled"] is True,
            )
            if allowed:
                changed = _bump("eligible_calls") or changed
                _held = True
            if changed:
                _write_witness()
        if allowed:
            temporary = f"{entered}.tmp.{os.getpid()}"
            fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "w", encoding="ascii") as marker:
                marker.write(str(time.monotonic_ns()))
            os.replace(temporary, entered)
            with _lock:
                _state["hold_entered"] = True
                _write_witness()
            time.sleep(hold_seconds)
        return original_replay(self, desc)

    ModelCudaGraphManager.__init__ = init_wrapped
    ModelCudaGraphManager.run_fullgraph = replay_wrapped
    _installed = True
