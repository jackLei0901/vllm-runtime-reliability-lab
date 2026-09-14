"""Reproduce a rank becoming unavailable to Flight Recorder during teardown."""

from __future__ import annotations

import datetime
import os
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist


class InjectedRankFailure(RuntimeError):
    pass


def atomic_write(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(b"rank 0 enqueued\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def wait_for(path: Path, timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            return
        time.sleep(0.01)
    raise TimeoutError(f"marker not observed: {path}")


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    marker = Path(os.environ["REPRO_READY_FILE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=30))

    tensor = torch.ones(16, device=f"cuda:{local_rank}")
    dist.all_reduce(tensor)
    torch.cuda.synchronize()
    print(f"rank {rank}: warm-up complete", flush=True)

    try:
        if rank == 0:
            work = dist.all_reduce(tensor, async_op=True)
            atomic_write(marker)
            print("rank 0: all-reduce enqueued; waiting", flush=True)
            work.wait(timeout=datetime.timedelta(seconds=180))
            print("rank 0: all-reduce returned unexpectedly", flush=True)
        else:
            wait_for(marker)
            print("rank 1: injecting local failure", flush=True)
            raise InjectedRankFailure("intentional rank-local failure")
    finally:
        print(f"rank {rank}: destroy_process_group entered", flush=True)
        dist.destroy_process_group()
        print(f"rank {rank}: destroy_process_group returned", flush=True)


if __name__ == "__main__":
    main()
