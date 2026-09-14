# Flight Recorder misses a rank blocked in ProcessGroupNCCL teardown

> **Draft status:** hold. The committed reproducer confirms the behavior on
> PyTorch 2.13.0's legacy `ProcessGroupNCCL`. The first nightly run selected the
> new default `nccl2` backend and is not a legacy-backend comparison. Run nightly
> once with `TORCH_DIST_USE_NCCL2=0` before deciding whether to file.

## Describe the bug

When one rank raises a local exception while a peer is waiting on a NCCL
collective, the failing rank can enter `destroy_process_group()` and become
unable to answer the peer's later Flight Recorder dump request. Only the peer
writes a dump, so the rank that raised the original exception is missing from
the dump set.

In PyTorch v2.13.0, `shutdown()` stops the heartbeat monitor
([source](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L1582))
before destroying NCCL communicators
([source](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L1585-L1590)).
That monitor is also the thread that polls the store for `exception_dump`
([source](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L1855-L1919)).

This report concerns the missing diagnostic artifact. It does not identify the
exact internal blocking call or propose a shutdown-ordering fix.

## Minimal reproducer

```python
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

    if rank == 0:
        marker.unlink(missing_ok=True)
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
```

Run it under an external hard bound after removing old dump files:

```bash
rm -f /tmp/pgnccl-trace_*
export REPRO_READY_FILE=/tmp/pgnccl-repro-$RANDOM.ready
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=3
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/tmp/pgnccl-trace_
timeout -k 10s 60s torchrun --standalone --nproc-per-node=2 reproducer.py \
  2>&1 | tee reproducer.log
ls -l /tmp/pgnccl-trace_*
```

## Observed behavior

The exact committed script (`e8af605`) reproduced the behavior on PyTorch
`2.13.0+cu130` with NCCL `2.29.7`. Relevant rank-prefixed output:

```text
rank 0: all-reduce enqueued; waiting
rank 1: injecting local failure
rank 1: destroy_process_group entered
[rank1] ... Rank 1] Watchdog joined, destroying NCCL communicators.
[rank0] ... [Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=2, OpType=ALLREDUCE, NumelIn=16, NumelOut=16, Timeout(ms)=30000) ran for 30005 milliseconds before timing out.
[rank0] ... Rank 0] Broadcasting signal exception_dump to other ranks via TCPStore.
[rank0] ... Rank 0] Flight Recorder trace successfully dumped.
```

There was no rank-1 `Destroy complete.`, no rank-1 `Observed flight recorder
dump signal from another rank via TCPStore.`, and no rank-1 dump. After the
external bound, the only dump file was `/tmp/pgnccl-trace_0` (1,428 bytes).
The command exited with `timeout` status 124; no claim is made about behavior
after that bound.

### Nightly backend note

The first nightly run did **not** reproduce the teardown stall on
`2.15.0.dev20260913+cu130` (git `13376c2070a764e25f67b2385c31358b325e8a1c`)
with NCCL `2.30.7`, but `backend="nccl"` selected the new default `nccl2`
implementation rather than legacy `ProcessGroupNCCL`:

```text
rank 1: destroy_process_group entered
[rank0] ... Operation timed out after 30794 ms
[rank0] ... Finished writing Flight Recorder debug info to /tmp/pgnccl-nightly-trace_0
rank 1: destroy_process_group returned
rank 0: destroy_process_group returned
```

The nightly job exited by itself with status 1 in about 35 seconds. This result
does not establish a version boundary: `nccl2` uses different timeout, shutdown
and Flight Recorder paths, and its rank-1 return neither proves nor disproves
the legacy missing-responder gap. Nightly's legacy backend remains to be tested
with `TORCH_DIST_USE_NCCL2=0`.

## Expected behavior

A rank that is still alive but blocked in the documented process-group cleanup
path should remain able to respond to a peer's timeout dump request, or PyTorch
should make the diagnostic limitation explicit.

## Environment

- PyTorch: `2.13.0+cu130`
- PyTorch git revision: `cf30153c4c131c8164ee7798e5022d810682e2cb`
- PyTorch CUDA build: 13.0
- system CUDA runtime reported by `collect_env`: 12.4.131
- NCCL: 2.29.7
- GPUs: 2 x NVIDIA GeForce RTX 4090
- driver: 580.105.08
- OS: Ubuntu 22.04.4, Linux 5.15.0-78-generic
- Python: 3.12.3
- nightly default-backend observation: `2.15.0.dev20260913+cu130`, NCCL 2.30.7,
  `nccl2`; not a legacy-backend test

### `python -m torch.utils.collect_env`

```text
The complete output is retained in the private validation record and must be
pasted here if this draft is converted into an upstream report.
```

## Related issues

- #117883 notes that `TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC` had to be raised from
  2 seconds to 60 seconds to obtain dumps from all ranks. That concerns the dump
  wait timeout, not a rank already blocked in teardown.
- #169943 covers a heartbeat-monitor hang after a dump; here the affected rank
  does not dump.
- #132696 covers a different `destroy_process_group()` hang during active
  reconfiguration.
- #122694 asks how to retain evidence before the elastic agent kills a hung
  process.

A search of open `oncall: distributed` issues for `destroy_process_group` and
dump-related terms did not find this missing-monitor sequence. Keyword search is
not proof that no duplicate exists.
