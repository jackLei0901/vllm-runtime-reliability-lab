# Flight Recorder misses a rank blocked in ProcessGroupNCCL teardown

> **Draft status:** do not file yet. The inline reproducer has not been run as
> committed. Replace the pending validation sections with its actual output and
> `collect_env` output first.

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

**Pending:** paste the actual rank-prefixed output from the standalone validation
run. Confirm that only `/tmp/pgnccl-trace_0` exists; do not infer behavior after
the 60-second external bound.

Expected lines to check, not yet reported as observations:

- rank 1: `Watchdog joined, destroying NCCL communicators.`, without
  `Destroy complete.` or `Observed flight recorder dump signal from another
  rank via TCPStore.`;
- rank 0: `Broadcasting signal exception_dump to other ranks via TCPStore.` and
  `Flight Recorder trace successfully dumped.`

## Expected behavior

A rank that is still alive but blocked in the documented process-group cleanup
path should remain able to respond to a peer's timeout dump request, or PyTorch
should make the diagnostic limitation explicit.

## Environment

- PyTorch: `2.13.0+cu130`
- PyTorch git revision: `cf30153c4c131c8164ee7798e5022d810682e2cb`
- CUDA runtime: 13.0
- NCCL: 2.29.7
- GPUs: 2 x NVIDIA GeForce RTX 4090
- nightly: not tested; the bundled NCCL version may affect shutdown behavior

### `python -m torch.utils.collect_env`

```text
PENDING: paste output from the standalone validation environment.
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
