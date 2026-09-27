"""Experiment-only bounded host hold before one rank-1 FULL graph replay.

The module imports no vLLM code until its general-plugin entry point runs in
each process. It changes no CUDA or NCCL implementation and is never a product
dependency. The launcher owns the private arm and entered-marker paths.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

_lock = threading.Lock()
_held = False
_installed = False


def eligible(
    *,
    armed: bool,
    held: bool,
    tp_rank: int,
    tp_world_size: int,
    wrapper_mode: str,
    runtime_mode: str,
    graph_cached: bool,
) -> bool:
    """Pure precondition, separately mutation-tested on the CPU."""
    return (
        armed
        and not held
        and tp_rank == 1
        and tp_world_size == 2
        and wrapper_mode == "FULL"
        and runtime_mode == "FULL"
        and graph_cached
    )


def install() -> None:
    """Patch only the test process's graph wrapper; fail closed on bad setup."""
    global _installed
    if _installed:
        return

    arm_path = os.environ.get("LLR_TP_ARM_FILE")
    entered_path = os.environ.get("LLR_TP_ENTER_FILE")
    hold_raw = os.environ.get("LLR_TP_HOLD_SECONDS", "3")
    if not arm_path or not entered_path:
        raise RuntimeError("LLR_TP_ARM_FILE and LLR_TP_ENTER_FILE are required")
    if not os.path.isabs(arm_path) or not os.path.isabs(entered_path):
        raise RuntimeError("hold marker paths must be absolute")
    hold_seconds = float(hold_raw)
    if not 2 <= hold_seconds <= 5:
        raise RuntimeError("hold duration must be between 2 and 5 seconds")

    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
    from vllm.forward_context import get_forward_context, is_forward_context_available

    original = CUDAGraphWrapper.__call__

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        global _held
        if os.path.exists(arm_path) and is_forward_context_available():
            context = get_forward_context()
            descriptor = context.batch_descriptor
            entry = self.concrete_cudagraph_entries.get(descriptor)
            if eligible(
                armed=True,
                held=_held,
                tp_rank=get_tensor_model_parallel_rank(),
                tp_world_size=get_tensor_model_parallel_world_size(),
                wrapper_mode=self.runtime_mode.name,
                runtime_mode=context.cudagraph_runtime_mode.name,
                graph_cached=entry is not None and entry.cudagraph is not None,
            ):
                do_hold = False
                with _lock:
                    if not _held:
                        _held = True
                        do_hold = True
                if do_hold:
                    temporary = f"{entered_path}.tmp.{os.getpid()}"
                    fd = os.open(
                        temporary,
                        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                        0o600,
                    )
                    with os.fdopen(fd, "w", encoding="ascii") as marker:
                        marker.write(str(time.monotonic_ns()))
                    os.replace(temporary, entered_path)
                    time.sleep(hold_seconds)
        return original(self, *args, **kwargs)

    CUDAGraphWrapper.__call__ = wrapped
    _installed = True
