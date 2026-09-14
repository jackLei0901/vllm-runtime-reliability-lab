# Standalone ProcessGroupNCCL validation

Status: **PASS: the selectable legacy ProcessGroupNCCL gap reproduces on current
nightly**

The standalone script from commit
`e8af6054fb519d12b353399efdab3b6b6e1a7af0` was run unchanged on the same two
RTX 4090 host used for Gate 1g. This run validates the script intended for an
upstream report; it is separate from the Gate 1g harness result.

## PyTorch 2.13 result

Environment:

- PyTorch `2.13.0+cu130`, git
  `cf30153c4c131c8164ee7798e5022d810682e2cb`;
- legacy `ProcessGroupNCCL` selected by `backend="nccl"`;
- PyTorch CUDA build 13.0, system CUDA runtime 12.4.131, NCCL 2.29.7;
- Ubuntu 22.04.4, Linux 5.15.0-78-generic;
- NVIDIA driver 580.105.08;
- two NVIDIA GeForce RTX 4090 GPUs.

Observed sequence:

```text
rank 0: all-reduce enqueued; waiting
rank 1: injecting local failure
rank 1: destroy_process_group entered
[rank1] ... Rank 1] Watchdog joined, destroying NCCL communicators.
[rank0] ... [Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=2, OpType=ALLREDUCE, NumelIn=16, NumelOut=16, Timeout(ms)=30000) ran for 30005 milliseconds before timing out.
[rank0] ... Rank 0] Broadcasting signal exception_dump to other ranks via TCPStore.
[rank0] ... Rank 0] Flight Recorder trace successfully dumped.
```

The run reached the declared 60-second external bound and returned status 124.
Rank 1 did not log `Destroy complete.` or observation of the peer dump signal.
After stale files were removed before launch, exactly one dump existed:
`/tmp/pgnccl-trace_0` (1,428 bytes). No behavior after the external bound is
inferred.

Private audit hashes:

- raw launcher log:
  `0a1a15797b8ccc1a4d3c10aead1adf56c791027a2d7004edc7668b2e8cdfa879`;
- rank-0 dump:
  `4f182da9c5840891f325ebaa726bc7deb312b887a90339f3cfdbd8f14ccdcd00`;
- `python -m torch.utils.collect_env` output:
  `d7b1ce47e6207700ca326013ecd093f568cbd897634a628716ce1c46431d071d`.

## Nightly default-backend observation

The same script and environment variables were then run with:

- PyTorch `2.15.0.dev20260913+cu130`, git
  `13376c2070a764e25f67b2385c31358b325e8a1c`;
- NCCL 2.30.7;
- default `nccl2` backend, not legacy `ProcessGroupNCCL`;
- the same host and GPUs.

Relevant output:

```text
rank 1: destroy_process_group entered
[rank0] ... Operation timed out after 30794 ms
[rank0] ... Finished writing Flight Recorder debug info to /tmp/pgnccl-nightly-trace_0
rank 1: destroy_process_group returned
rank 0: destroy_process_group returned
```

The job exited without the external bound, with status 1 after about 35 seconds.
Only the rank-0 dump was present, but rank 1 had completed teardown and exited;
therefore this is not the Gate 1g missing-responder stall. It also cannot show
whether nightly's separately selectable legacy backend has the gap, because
`nccl2` has a different timeout, shutdown and Flight Recorder implementation.

Private audit hashes:

- raw launcher log:
  `1ee3c206fb0a2c4d18ec4a71e7c96f6c272b417c0c0341bbca27a5dbd5b7e1e1`;
- rank-0 dump:
  `231cb39042b74e6cd73b9af77b80e83d5bbfecb5dad906183fd2b94dbd8a9b1f`.

## Nightly legacy-backend result

The exact script was run a third time with the same nightly wheel and NCCL
version, plus the pre-registered backend selector
`TORCH_DIST_USE_NCCL2=0`. The legacy backend reproduced the gap:

```text
rank 1: destroy_process_group entered
[rank1] ... Rank 1] Watchdog joined, destroying NCCL communicators.
[rank0] ... Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=2, OpType=ALLREDUCE, NumelIn=16, NumelOut=16, Timeout(ms)=30000) ran for 30028 milliseconds before timing out.
[rank0] ... Rank 0] Broadcasting signal exception_dump to other ranks via TCPStore.
[rank0] ... Rank 0] Flight Recorder trace successfully dumped.
```

Rank 1 logged neither `Destroy complete.` nor `Observed flight recorder dump
signal`. The run reached the 60-second external bound with status 124. After
stale files were removed before launch, only
`/tmp/pgnccl-legacy-trace_0` existed (1,447 bytes).

Private audit hashes:

- raw launcher log:
  `ede66b3dc52ad4e218b3100f7d14acddf47806a0cf353d9f22cd73553bf23b36`;
- rank-0 dump:
  `131c1c000a02d62fe4b34a0b29f404fd9aa8c3f507e779b2697fa7e902fd7901`.

## Interpretation

The standalone validation closes the script-provenance gap and the backend
confound. The exact script reproduces on both PyTorch 2.13 and the selectable
legacy backend in current nightly. The default `nccl2` run is a separate
observation and is not used to claim that the gap is fixed or present there.

The pre-registered positive outcome was met, so the result now supports filing
an issue against current nightly's legacy backend. The exact NCCL blocking call
and a safe shutdown-ordering fix remain outside the evidence.

Raw logs, dumps and environment output are retained outside the public repository
and are not committed.
