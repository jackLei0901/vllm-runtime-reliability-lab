# Flight Recorder responder stops before a rank blocks in ProcessGroupNCCL teardown

## Describe the bug

When one rank raises a local exception while a peer is waiting on a NCCL
collective, the failing rank can enter `destroy_process_group()`, stop its
Flight Recorder heartbeat responder, and then remain blocked during communicator
destruction. When the peer later times out and broadcasts `exception_dump`, only
the peer writes a dump. The rank that raised the original exception is missing
from the dump set.

This report is about the missing diagnostic artifact, not the underlying NCCL
shutdown stall or a proposed shutdown-ordering fix.

## Minimal reproducer

Use `standalone_reproducer.py` from this directory. Choose a fresh marker path,
then run under an external bound:

```bash
export REPRO_READY_FILE=/tmp/pgnccl-repro-$RANDOM.ready
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=3
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/tmp/pgnccl-trace_
timeout 60s torchrun --standalone --nproc-per-node=2 standalone_reproducer.py
```

## Observed behavior

On PyTorch `2.13.0+cu130` with NCCL `2.29.7`:

- both ranks complete a warm-up all-reduce;
- rank 0 enqueues a second all-reduce and waits;
- rank 1 observes that enqueue marker, raises locally, and enters
  `destroy_process_group()`;
- rank 1 logs shutdown start, operation flush, and
  `Watchdog joined, destroying NCCL communicators.`, but not `Destroy complete.`;
- after the collective timeout, rank 0 logs a successful `exception_dump`
  broadcast and writes a decodable Flight Recorder dump;
- rank 1 neither logs receipt of that signal nor writes a dump.

The rank-0 dump contains one non-completed `ALL_REDUCE` for group `[0, 1]`.
It does not establish whether rank 1 entered that collective.

| observation | rank 0 | rank 1 |
| --- | --- | --- |
| dump signal broadcast | yes | n/a |
| dump signal observed from peer | n/a | no |
| dump written | yes | no |
| communicator destruction started | no | yes |
| communicator destruction completed | no | no |

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

## Related issues

- #117883 tracks Flight Recorder feature requests, including obtaining dumps
  from all ranks.
- #169943 covers a heartbeat-monitor hang after a dump; here the affected rank
  does not dump.
- #132696 covers a different `destroy_process_group()` hang during active
  reconfiguration.
- #122694 asks how to retain evidence before the elastic agent kills a hung
  process.

A search of open `oncall: distributed` issues for `destroy_process_group` and
dump-related terms did not find this missing-responder sequence. Keyword search
is not a proof that no duplicate exists.
