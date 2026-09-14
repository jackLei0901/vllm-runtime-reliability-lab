# Flight Recorder misses a rank blocked in ProcessGroupNCCL teardown

> **Draft status:** technically ready for final human review. The committed
> reproducer confirms the behavior on PyTorch 2.13.0 and on current nightly's
> selectable legacy `ProcessGroupNCCL` with `TORCH_DIST_USE_NCCL2=0`.

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
export TORCH_DIST_USE_NCCL2=0
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

### Nightly backend results

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
the legacy missing-responder gap.

With `TORCH_DIST_USE_NCCL2=0`, the same nightly wheel selected legacy
`ProcessGroupNCCL` and reproduced the missing-rank behavior:

```text
rank 1: destroy_process_group entered
[rank1] ... Rank 1] Watchdog joined, destroying NCCL communicators.
[rank0] ... Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=2, OpType=ALLREDUCE, NumelIn=16, NumelOut=16, Timeout(ms)=30000) ran for 30028 milliseconds before timing out.
[rank0] ... Rank 0] Broadcasting signal exception_dump to other ranks via TCPStore.
[rank0] ... Rank 0] Flight Recorder trace successfully dumped.
```

Rank 1 logged neither `Destroy complete.` nor `Observed flight recorder dump
signal`, and only `/tmp/pgnccl-legacy-trace_0` existed at the external bound.
The default `nccl2` observation is not used to claim that its different Flight
Recorder implementation has or fixes this gap.

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
- nightly legacy reproduction: `2.15.0.dev20260913+cu130`, git
  `13376c2070a764e25f67b2385c31358b325e8a1c`, NCCL 2.30.7,
  `TORCH_DIST_USE_NCCL2=0`
- nightly default-backend observation: same wheel and NCCL, `nccl2`; separate
  backend, not used as a legacy comparison

### `python -m torch.utils.collect_env`

```text
Collecting environment information...
PyTorch version: 2.13.0+cu130
Is debug build: False
CUDA used to build PyTorch: 13.0
ROCM used to build PyTorch: N/A

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: Could not collect
CMake version: version 3.22.1
Libc version: glibc-2.35

Python version: 3.12.3 | packaged by Anaconda, Inc. | (main, May  6 2024, 19:46:43) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-5.15.0-78-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: 12.4.131
CUDA_MODULE_LOADING set to:
GPU models and configuration:
GPU 0: NVIDIA GeForce RTX 4090
GPU 1: NVIDIA GeForce RTX 4090

Nvidia driver version: 580.105.08
cuDNN version: Probably one of the following:
/usr/lib/x86_64-linux-gnu/libcudnn.so.9.1.0
/usr/lib/x86_64-linux-gnu/libcudnn_adv.so.9.1.0
/usr/lib/x86_64-linux-gnu/libcudnn_cnn.so.9.1.0
/usr/lib/x86_64-linux-gnu/libcudnn_engines_precompiled.so.9.1.0
/usr/lib/x86_64-linux-gnu/libcudnn_engines_runtime_compiled.so.9.1.0
/usr/lib/x86_64-linux-gnu/libcudnn_graph.so.9.1.0
/usr/lib/x86_64-linux-gnu/libcudnn_heuristic.so.9.1.0
/usr/lib/x86_64-linux-gnu/libcudnn_ops.so.9.1.0
Is XPU available: False
HIP runtime version: N/A
MIOpen runtime version: N/A
Is XNNPACK available: False
Caching allocator config: N/A

CPU:
Architecture:                    x86_64
CPU op-mode(s):                  32-bit, 64-bit
Address sizes:                   52 bits physical, 57 bits virtual
Byte Order:                      Little Endian
CPU(s):                          128
On-line CPU(s) list:             0-127
Vendor ID:                       GenuineIntel
Model name:                      Intel(R) Xeon(R) Platinum 8352V CPU @ 2.10GHz
CPU family:                      6
Model:                           106
Thread(s) per core:              2
Core(s) per socket:              32
Socket(s):                       2
Stepping:                        6
CPU max MHz:                     3500.0000
CPU min MHz:                     800.0000
BogoMIPS:                        4000.00
Flags:                           fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 invpcid_single intel_ppin ssbd mba ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb intel_pt avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local split_lock_detect wbnoinvd dtherm ida arat pln pts hwp hwp_act_window hwp_epp hwp_pkg_req avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg tme avx512_vpopcntdq la57 rdpid fsrm md_clear pconfig flush_l1d arch_capabilities
Virtualization:                  VT-x
L1d cache:                       3 MiB (64 instances)
L1i cache:                       2 MiB (64 instances)
L2 cache:                        80 MiB (64 instances)
L3 cache:                        108 MiB (2 instances)
NUMA node(s):                    2
NUMA node0 CPU(s):               0-31,64-95
NUMA node1 CPU(s):               32-63,96-127
Vulnerability Itlb multihit:     Not affected
Vulnerability L1tf:              Not affected
Vulnerability Mds:               Not affected
Vulnerability Meltdown:          Not affected
Vulnerability Mmio stale data:   Mitigation; Clear CPU buffers; SMT vulnerable
Vulnerability Retbleed:          Not affected
Vulnerability Spec store bypass: Mitigation; Speculative Store Bypass disabled via prctl and seccomp
Vulnerability Spectre v1:        Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:        Mitigation; Enhanced IBRS, IBPB conditional, RSB filling, PBRSB-eIBRS SW sequence
Vulnerability Srbds:             Not affected
Vulnerability Tsx async abort:   Not affected

Versions of relevant libraries:
[pip3] numpy==2.2.6
[pip3] nvidia-cublas==13.1.1.3
[pip3] nvidia-cuda-cupti==13.0.85
[pip3] nvidia-cuda-nvrtc==13.0.88
[pip3] nvidia-cuda-runtime==13.0.96
[pip3] nvidia-cudnn-cu13==9.20.0.48
[pip3] nvidia-cufft==12.0.0.61
[pip3] nvidia-curand==10.4.0.35
[pip3] nvidia-cusolver==12.0.4.66
[pip3] nvidia-cusparse==12.6.3.3
[pip3] nvidia-cusparselt-cu13==0.8.1
[pip3] nvidia-nccl-cu13==2.29.7
[pip3] nvidia-nvjitlink==13.3.33
[pip3] nvidia-nvtx==13.0.85
[pip3] torch==2.13.0+cu130
[pip3] triton==3.7.1
[conda] numpy                     2.1.3                    pypi_0    pypi
[conda] nvidia-cublas-cu12        12.4.5.8                 pypi_0    pypi
[conda] nvidia-cuda-cupti-cu12    12.4.127                 pypi_0    pypi
[conda] nvidia-cuda-nvrtc-cu12    12.4.127                 pypi_0    pypi
[conda] nvidia-cuda-runtime-cu12  12.4.127                 pypi_0    pypi
[conda] nvidia-cudnn-cu12         9.1.0.70                 pypi_0    pypi
[conda] nvidia-cufft-cu12         11.2.1.3                 pypi_0    pypi
[conda] nvidia-curand-cu12        10.3.5.147               pypi_0    pypi
[conda] nvidia-cusolver-cu12      11.6.1.9                 pypi_0    pypi
[conda] nvidia-cusparse-cu12      12.3.1.170               pypi_0    pypi
[conda] nvidia-nccl-cu12          2.21.5                   pypi_0    pypi
[conda] nvidia-nvjitlink-cu12     12.4.127                 pypi_0    pypi
[conda] nvidia-nvtx-cu12          12.4.127                 pypi_0    pypi
[conda] torch                     2.5.1+cu124              pypi_0    pypi
[conda] torchvision               0.20.1+cu124             pypi_0    pypi
[conda] triton                    3.1.0                    pypi_0    pypi
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
