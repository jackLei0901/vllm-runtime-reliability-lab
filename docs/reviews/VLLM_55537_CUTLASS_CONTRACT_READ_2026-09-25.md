# #55537 CUTLASS C3x contract read — independent kernel track

Status: **read-only source/trial-design note**, 2026-09-25. This is not a Lab
DFX finding, a kernel implementation, a GPU validation, or a new upstream
proposal. The inspected checkout is
`vllm-55537-reject-only` at `7b054aca96cea8be1369d651c3434ad140580b92`;
its working tree was clean. The candidate commit is a reject-only dispatch
guard plus tests, not a change to CUTLASS arithmetic.

## Contract boundary seen in source

`csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_entry.cu:228-243`
checks two-dimensional conformability, unit inner stride for row-major A
and output / column-major B, and a 16-*element* multiple for B and output
leading strides. Those checks do not require a packed leading dimension:
a slice of a larger allocation can pass while its row/column pitch exceeds
the visible extent.

One downstream C3x caller in
`csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/cutlass_gemm_caller.cuh:85-100`
constructs **packed strides from M/N/K**, then passes raw A/B/output pointers.
The SM90 and SM100 FP8 dispatchers likewise construct packed strides from
shape rather than each tensor's leading stride
(`scaled_mm_sm90_fp8_dispatch.cuh:258-267` and
`scaled_mm_sm100_fp8_dispatch.cuh:215-224`). Thus, for a padded view, the
metadata consumed by the kernel can
describe a different layout from the caller's tensor view. This is the
source-level reason for a layout guard; it is stronger than merely naming
“non-contiguous input.” It is not by itself an executed wrong-answer or
memory-safety demonstration on a given SM/runtime.

The candidate adds `has_packed_leading_dimension()` and a 16-byte pointer
check at `scaled_mm_entry.cu:200-223`, called only before compiled C3x
`scaled_mm` SM90/100/120 dispatch and the SM90+ AZP path. Older C2x
branches are left under their previous acceptance rules. `size(leading_dim)
<= 1` exempts the degenerate leading dimension because its stride is not
observed by traversal of a second row/column; the existing unit inner-stride
check still applies. Pointer alignment and packed pitch are separate
predicates; a view can satisfy either one without satisfying the other.

| Operand | Existing inner-stride precondition | New leading-pitch condition | New pointer condition |
| --- | --- | --- | --- |
| A, row-major M x K | `stride(1)==1` | M<=1 or `stride(0)==K` | address divisible by 16 bytes |
| B, column-major K x N | `stride(0)==1` | N<=1 or `stride(1)==K` | address divisible by 16 bytes |
| output, row-major M x N | `stride(1)==1` | M<=1 or `stride(0)==N` | address divisible by 16 bytes |

The helper does not validate scale tensor layouts or the separate grouped
entry. It **does** reach blockwise FP8 A/B/output: the SM90/100/120 wrappers
call `dispatch_scaled_mm()` only *after* this guard, and
`c3x/scaled_mm_helper.hpp:22-55` selects blockwise by scale `numel`.
The blockwise dispatchers also synthesize packed strides from shape. This
widens the reject-only behavior beyond the current test's ordinary FP8
cases; do not generalize it to every vLLM quantization kernel.

## Existing test intent and remaining validation

`tests/kernels/quantization/test_cutlass_scaled_mm.py` on this branch has:

- a previous subset test changed to expect rejection on C3x, while retaining
  the C2x numerical comparison;
- A/B/output padded-view rejection for FP8 and supported INT8 cases;
- a positive M=1 case preserving a legal noncanonical leading stride;
- A padded-view rejection on AZP; and
- A/B/output misaligned-pointer rejection.

These are candidate test *definitions*, not results from this review. They
exercise the dispatch guard rather than a newly written kernel. A reviewer
should ask whether the relevant CUDA build actually enables each intended
SM branch and whether the exception arises before the CUTLASS call. On the
AZP path, only padded A is explicitly tested by the added AZP-specific
case; B/output share the same helper by source, but a complete behavioral
matrix would test them if that distinction affects acceptance. There is
also no explicit **blockwise FP8 padded-view** case, despite the guard's
source-level reach. That negative control is worth considering before
claiming validated blockwise behavior, particularly for sliced weights in
DeepSeek-style block-FP8 use. No new tests are added here.

The next kernel-capability step is **not** another guard. With maintainer
direction and a suitable GPU/build, choose one adjacent kernel path and
write a version-pinned contract for numerical reference, shapes/dtypes,
alignment, correctness tolerances, negative layout controls, and a measured
performance/resource trade-off. Until those execute, #55537 remains useful
CUDA/C++ *dispatch-contract* work and not evidence of kernel implementation
or performance engineering. It stays separate from Lab DFX metrics.

## 中文审阅摘要

这个分支的价值在于源码级别识别了“张量真实 pitch”与 C3x caller 构造的 packed
stride 之间的合同不一致，并在进入已有 kernel 前拒绝不受支持的布局。它不是新
kernel，也尚未由本轮审阅运行 GPU 测试。后续若要证明 kernel 能力，需要在
固定架构与环境下完成数值正确性、负对照和性能/资源取舍，而不是继续增加
dispatch 检查。
