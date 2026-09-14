## 🐛 Describe the bug

FSDP2 gradient accumulation with
`MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)`
can present both bf16 and fp32 gradients to one `foreach_reduce()` call.
`foreach_reduce()` rejects the mixed list before copying it into the
`reduce_dtype` buffer:

```text
FSDP reduce-scatter expects uniform gradient dtype but got
{torch.bfloat16, torch.float32}
```

There are two triggers for the same underlying condition:

1. A parameter is first used on the final, synchronized microbatch. Its fresh
   gradient remains bf16 while gradients accumulated from earlier microbatches
   are fp32. This is the case previously reported in #160279.
2. With `set_reduce_scatter_unused_params(True)`, a parameter that is never
   used contributes a bf16 zero placeholder while another parameter contributes
   an accumulated fp32 gradient.

The assertion reproduces with one GPU.

### Minimal reproducer

Save as `reproducer.py`:

```python
"""Minimal FSDP2 reproducer for mixed fresh/accumulated gradient dtypes.

Examples:
  torchrun --standalone --nproc-per-node=1 reproducer.py --case control
  torchrun --standalone --nproc-per-node=1 reproducer.py --case last-microbatch
  torchrun --standalone --nproc-per-node=1 reproducer.py --case unused-placeholder
  torchrun --standalone --nproc-per-node=2 reproducer.py --case rank-divergent-unused
"""

from __future__ import annotations

import argparse
import datetime
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard


CASES = (
    "control",
    "last-microbatch",
    "unused-placeholder",
    "rank-divergent-unused",
)


class Model(nn.Module):
    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.always = nn.Linear(width, width)
        self.conditional = nn.Linear(width, width)

    def forward(self, x: torch.Tensor, use_conditional: bool) -> torch.Tensor:
        output = self.always(x)
        if use_conditional:
            output = output + self.conditional(x)
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--microbatches", type=int, default=4)
    args = parser.parse_args()
    if args.microbatches < 2:
        parser.error("--microbatches must be at least 2")
    return args


def uses_conditional(case: str, rank: int, microbatch: int, last: int) -> bool:
    if case == "control":
        return True
    if case == "last-microbatch":
        return microbatch == last
    if case == "unused-placeholder":
        return False
    if case == "rank-divergent-unused":
        return rank == 0
    raise AssertionError(f"unhandled case: {case}")


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if args.case == "rank-divergent-unused" and world_size < 2:
        raise ValueError("rank-divergent-unused requires at least 2 ranks")
    if not hasattr(FSDPModule, "set_reduce_scatter_unused_params"):
        raise RuntimeError("this PyTorch build lacks the required FSDP2 API")

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=30))
    device = torch.device("cuda", local_rank)

    print(
        f"REPRO_START rank={rank} world_size={world_size} case={args.case} "
        f"torch={torch.__version__}",
        flush=True,
    )

    try:
        torch.manual_seed(20260914)
        model = Model().to(device)
        fully_shard(
            model,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
            ),
        )
        if args.case in ("unused-placeholder", "rank-divergent-unused"):
            model.set_reduce_scatter_unused_params(True)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)

        last = args.microbatches - 1
        for microbatch in range(args.microbatches):
            is_last = microbatch == last
            model.set_is_last_backward(is_last)
            model.set_requires_gradient_sync(is_last)
            model.set_reshard_after_backward(is_last)
            x = torch.randn(4, 16, device=device, dtype=torch.bfloat16)
            use_conditional = uses_conditional(
                args.case, rank, microbatch, last
            )
            model(x, use_conditional).float().sum().backward()

        optimizer.step()
        dist.barrier()
        print(f"REPRO_RESULT rank={rank} status=completed", flush=True)
    except BaseException as error:
        print(
            f"REPRO_RESULT rank={rank} status=exception "
            f"type={type(error).__name__} message={error}",
            flush=True,
        )
        raise
    finally:
        print(f"REPRO_CLEANUP rank={rank} stage=enter", flush=True)
        dist.destroy_process_group()
        print(f"REPRO_CLEANUP rank={rank} stage=return", flush=True)


if __name__ == "__main__":
    main()
```

### Commands and observed results

Tested on `torch==2.15.0.dev20260914+cu130`:

The final script enables `set_reduce_scatter_unused_params()` only for the
`unused-placeholder` and `rank-divergent-unused` cases. The final flag-off run
confirmed that `last-microbatch` reproduces independently of that API.

```bash
torchrun --standalone --nproc-per-node=1 reproducer.py --case control
torchrun --standalone --nproc-per-node=1 reproducer.py --case last-microbatch
torchrun --standalone --nproc-per-node=1 reproducer.py --case unused-placeholder

timeout -k 10s 70s torchrun --standalone --nproc-per-node=2 \
  reproducer.py --case rank-divergent-unused
```

| ranks | case | observed result |
| ---: | --- | --- |
| 1 | `control` | exits 0; `status=completed` |
| 1 | `last-microbatch` | exits 1 with the mixed-dtype assertion |
| 1 | `unused-placeholder` | exits 1 with the same assertion |

The relevant single-rank traceback is:

```text
[Rank 0] FSDP reduce-scatter expects uniform gradient dtype but got
{torch.bfloat16, torch.float32}
  File "torch/distributed/fsdp/_fully_shard/_fsdp_state.py", line 421,
    in _root_post_backward_final_callback
    fsdp_param_group.post_backward()
  File "torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py", line 712,
    in post_backward
    ) = foreach_reduce(
  File "torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py", line 556,
    in foreach_reduce
    _raise_assert_with_print(
AssertionError: FSDP reduce-scatter expects uniform gradient dtype but got
{torch.bfloat16, torch.float32}
```

**Impact in multi-rank jobs.** `set_reduce_scatter_unused_params()` targets
rank-divergent parameter usage. In that setting only some ranks may hit the
assertion: with `--case rank-divergent-unused` on two GPUs, rank 1 raised while
rank 0 did not complete (its reduce-scatter had no matching call from rank 1),
and the job did not finish within a 70-second external bound using nightly's
default `nccl2` backend. In practice this can appear as a stalled job rather
than an obvious dtype error.

### Expected behavior

The `last-microbatch` and `unused-placeholder` cases should complete like
`control`. Each parameter's gradients should be accumulated over microbatches
in `reduce_dtype` (contributing zero for a parameter that was never used) and
then reduced as usual, with gradients accumulated in earlier microbatches
keeping their `reduce_dtype` precision. The existing uniform-dtype check should continue
to catch the mixtures it was written for, such as fp8 weights that do not
produce higher-precision gradients.

### Source-level explanation on the tested nightly

At commit
[`7d5f0216450b8253e62d88284d49de725d88bd42`](https://github.com/pytorch/pytorch/commit/7d5f0216450b8253e62d88284d49de725d88bd42):

- [`to_accumulated_grad_if_needed()` converts an unsynchronized gradient to
  `reduce_dtype`](https://github.com/pytorch/pytorch/blob/7d5f0216450b8253e62d88284d49de725d88bd42/torch/distributed/fsdp/_fully_shard/_fsdp_param.py#L1060-L1077).
- On the final synchronized backward,
  [`post_backward()` collects accumulated, fresh, or zero-placeholder gradients
  into one list](https://github.com/pytorch/pytorch/blob/7d5f0216450b8253e62d88284d49de725d88bd42/torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py#L638-L658).
- [`unsharded_zero_grad_data` uses
  `torch.zeros_like(self.unsharded_param)`](https://github.com/pytorch/pytorch/blob/7d5f0216450b8253e62d88284d49de725d88bd42/torch/distributed/fsdp/_fully_shard/_fsdp_param.py#L1197-L1199),
  so it follows `param_dtype` here.
- [`foreach_reduce()` rejects a non-uniform dtype set before selecting or using
  `reduce_dtype`](https://github.com/pytorch/pytorch/blob/7d5f0216450b8253e62d88284d49de725d88bd42/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py#L553-L560).

This report does not prescribe where the reconciliation should occur.

### Related work and duplicate search

- #160279 reported the same assertion when a parameter is first used on the
  final microbatch. It was closed as stale and was not merged.
- #170667 introduced `set_reduce_scatter_unused_params()` and zero placeholders
  to keep collective shapes consistent across divergent parameter usage. The
  placeholder creates the second trigger described here.
- #183040 fixed unused DTensor parameter handling in the same path, not the
  accumulated/fresh dtype mismatch.
- #174862 changes unused-gradient handling but still builds the placeholder in
  the parameter/original dtype; it does not reconcile it with an already
  accumulated `reduce_dtype` gradient.
- #194546 supports mixed gradients when non-uniform per-parameter `grad_dtype`
  settings explain the mixture. Its uniform-configuration branch deliberately
  retains this assertion, so it does not cover this case.
- The investigation began from pytorch/torchtitan#2747. This minimal reproducer
  removes torchtitan, pipeline parallelism, model code, and multi-node setup.

I searched open and closed PyTorch issues and pull requests using the assertion
text, `set_reduce_scatter_unused_params`, unused-gradient, accumulation, and
reduce-scatter dtype terms. I did not find a merged fix or an open report that
covers both triggers above.

cc @weifengpy

## Versions

The reproducer ran from a dedicated pip virtual environment. Packages shown by
`collect_env` under the unrelated Conda base environment were not used.

```text
PyTorch version: 2.15.0.dev20260914+cu130
PyTorch git revision: 7d5f0216450b8253e62d88284d49de725d88bd42
Is debug build: False
CUDA used to build PyTorch: 13.0
Loaded NCCL (`torch.cuda.nccl.version()`): 2.30.7
OS: Ubuntu 22.04.4 LTS (x86_64)
Python: 3.12.3
GPU 0: NVIDIA GeForce RTX 4090
GPU 1: NVIDIA GeForce RTX 4090
NVIDIA driver: 580.105.08
cuDNN: 9.1.0
```

<details>
<summary>Complete collect_env output</summary>

```text
Collecting environment information...
PyTorch version: 2.15.0.dev20260914+cu130
Is debug build: False
CUDA used to build PyTorch: 13.0
ROCm SDK used to build PyTorch: N/A
HIP used to build PyTorch: N/A

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
[pip3] nvidia-cublas==13.1.1.3
[pip3] nvidia-cuda-cupti==13.0.85
[pip3] nvidia-cuda-nvrtc==13.0.88
[pip3] nvidia-cuda-runtime==13.0.96
[pip3] nvidia-cudnn-cu13==9.25.1.1
[pip3] nvidia-cufft==12.0.0.61
[pip3] nvidia-cusolver==12.0.4.66
[pip3] nvidia-cusparse==12.6.3.3
[pip3] nvidia-cusparselt-cu13==0.8.1
[pip3] nvidia-nccl-cu13==2.30.7
[pip3] nvidia-nvjitlink==13.4.46rc1
[pip3] nvidia-nvtx==13.0.85
[pip3] torch==2.15.0.dev20260914+cu130
[pip3] triton==3.8.0+gitc01b6774
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

</details>

> **AI assistance disclosure:** AI tools helped organize and review this report
> and prepare the reproducer. I ran the experiments, reviewed the reproducer and
> source links, verified the reported outputs, and take responsibility for the
> final report.
