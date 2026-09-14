"""Minimal two-rank ProcessGroupNCCL shutdown/dump reproducer."""

from __future__ import annotations

import argparse
import ctypes
import datetime
import json
import os
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist

PR_SET_PTRACER = 0x59616D61
YAMA_SCOPE = Path("/proc/sys/kernel/yama/ptrace_scope")


class InjectedRankFailure(RuntimeError):
    pass


def emit(prefix: str, payload: dict) -> None:
    print(prefix + json.dumps(payload, sort_keys=True), flush=True)


def elapsed(started: float) -> float:
    return round(time.monotonic() - started, 6)


def authorize_observer_from_env() -> str:
    if not YAMA_SCOPE.exists():
        return "yama_absent"
    observer_pid = int(os.environ.get("DFX_OBSERVER_PID", "0"))
    if observer_pid <= 0:
        raise RuntimeError("DFX_OBSERVER_PID must identify the campaign process")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_PTRACER, observer_pid, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    return "pr_set_ptracer_parent"


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def wait_for_path(path: Path, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.is_file():
            return
        time.sleep(0.01)
    raise TimeoutError(f"coordination marker did not appear: {path.name}")


def run_all_reduce(
    tensor: torch.Tensor,
    *,
    rank: int,
    started: float,
    ready_path: Path,
    publish_ready: bool,
) -> None:
    work = dist.all_reduce(tensor, async_op=True)
    emit(
        "DFX_G1G_ALLREDUCE_ENQUEUED=",
        {"rank": rank, "elapsed_seconds": elapsed(started)},
    )
    if publish_ready:
        atomic_write(ready_path, "rank0_all_reduce_enqueued\n")
    work.wait(timeout=datetime.timedelta(seconds=180))
    emit(
        "DFX_G1G_ALLREDUCE_RETURN=",
        {"rank": rank, "elapsed_seconds": elapsed(started)},
    )


def warm_up_communicator(tensor: torch.Tensor, *, rank: int, started: float) -> None:
    emit(
        "DFX_G1G_WARMUP_ENTER=",
        {"rank": rank, "elapsed_seconds": elapsed(started)},
    )
    dist.all_reduce(tensor)
    torch.cuda.synchronize()
    emit(
        "DFX_G1G_WARMUP_RETURN=",
        {"rank": rank, "elapsed_seconds": elapsed(started)},
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("control", "affected"), required=True)
    args = parser.parse_args()

    started = time.monotonic()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    state_dir = Path(os.environ["DFX_STATE_DIR"])
    ready_path = state_dir / "rank0-all-reduce-enqueued"
    ptrace_mode = authorize_observer_from_env()
    emit("DFX_G1G_PTRACE=", {"rank": rank, "mode": ptrace_mode})
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=30))
    atomic_write(state_dir / f"rank-{rank}.pid", f"{os.getpid()}\n")
    tensor = torch.ones(16, device=torch.device("cuda", local_rank))
    warm_up_communicator(tensor, rank=rank, started=started)

    try:
        if args.arm == "control" or rank == 0:
            run_all_reduce(
                tensor,
                rank=rank,
                started=started,
                ready_path=ready_path,
                publish_ready=args.arm == "affected" and rank == 0,
            )
            emit(
                "DFX_G1G_OUTCOME=",
                {
                    "rank": rank,
                    "outcome": "completed",
                    "elapsed_seconds": elapsed(started),
                },
            )
            return 0

        wait_for_path(ready_path, timeout_seconds=10)
        emit(
            "DFX_G1G_READY_OBSERVED=",
            {"rank": rank, "elapsed_seconds": elapsed(started)},
        )
        emit(
            "DFX_G1G_OUTCOME=",
            {
                "rank": rank,
                "outcome": "injected_rank_failure",
                "elapsed_seconds": elapsed(started),
            },
        )
        raise InjectedRankFailure("intentional Gate 1g rank-local failure")
    except InjectedRankFailure:
        raise
    except BaseException as exc:
        emit(
            "DFX_G1G_OUTCOME=",
            {
                "rank": rank,
                "outcome": "unexpected_exception",
                "exception_type": type(exc).__name__,
                "elapsed_seconds": elapsed(started),
            },
        )
        raise
    finally:
        emit(
            "DFX_G1G_TEARDOWN_ENTER=",
            {"rank": rank, "elapsed_seconds": elapsed(started)},
        )
        dist.destroy_process_group()
        emit(
            "DFX_G1G_TEARDOWN_RETURN=",
            {"rank": rank, "elapsed_seconds": elapsed(started)},
        )


if __name__ == "__main__":
    raise SystemExit(main())
