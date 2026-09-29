# Quantized GEMM operand-contract inventory: `cutlass_scaled_mm` and `cutlass_moe_mm`

Status: revised 2026-09-29. Kernel-correctness track, not the Lab runtime model. Chinese text not yet written.

**Evidence levels.**

- **B1 (blockwise scale layout) is observed**: one preregistered H800 run, predictions committed first in Lab `8ab51bb`, all eight cases as predicted ([result](../../experiments/kernel-operand-contracts/B1_H800_RESULT_2026-09-29.md)). It is latent in vLLM's current callers, and was reported on #55534 in [issuecomment-5893471238](https://github.com/vllm-project/vllm/issues/55534#issuecomment-5893471238).
- **B2 is observed** in one preregistered H800 run: unsupported operand dtypes produced incorrect blockwise output without an error; the standard branch rejected e5m2 ([result](../../experiments/kernel-operand-contracts/B2_B0_H800_RESULT_2026-09-29.md)). Whether to reject these inputs remains an API-contract question.
- **B0** is the already-reported #55534 class on the blockwise path. The same run supplies padded-view regression coverage (the Q6 vector); it is not a new finding.
- **MoE rows M3–M11 remain source reading only.** "Gap candidate" there means a question for a negative-control test, not an observed defect.

**Pin and anchors.** vLLM `main` at `91d7324cb19d301c72d849e457221ee8dd645024`. Anchors are checked mechanically: [`check_anchors.py`](../../experiments/kernel-operand-contracts/check_anchors.py) fetches the 19 cited files at the pin, verifies their git blob ids (appendix), and asserts the expected text on each cited line. On 2026-09-29 it reported 19 blobs, 106 anchors, 0 failures. Ranges below are checked at their key lines, not every line. The B1 run's three source files (entry, helper, SM90 blockwise dispatcher) have the same blobs at `main` `741edeebeec3cedbe938d831b6d87641ed6191ef`. Paths are under `csrc/libtorch_stable/quantization/w8a8/cutlass/` unless stated.

**Reachability.** vLLM's production callers satisfy every requirement below by convention. No production-reachable silent-wrong-result bug has been found. The claim is narrower: some requirements live only in caller convention, and host-side checks would make a future caller's mistake loud instead of silent.

## `cutlass_scaled_mm`

Entry: `scaled_mm_entry.cu` L197–270, then `c3x/scaled_mm_helper.hpp` `dispatch_scaled_mm` for SM90/100/120.

| # | Requirement | Where it is checked (pin) | Existing test coverage | Status |
| --- | --- | --- | --- | --- |
| S1 | `a`, `b`, `c` are 2-D and shape-conformal | entry L203–205 | all GEMM tests | checked |
| S2 | `a` and `c` row-major inner dimension (`stride(1) == 1`), `b` column-major (`stride(0) == 1`) | entry L208–209 | implicit in all tests | checked |
| S3 | `c.stride(0)` and `b.stride(1)` multiples of 16 | entry L210–211 | padded-B test | checked; the comment says "16 Byte", but the check is 16 *elements*, stricter than 16 bytes for 16-bit outputs (not a bug) |
| S4 | `a.stride(0)` (A's leading stride) honored or rejected | **not checked at entry** | `test_cutlass_subset` (`test_cutlass_scaled_mm.py` L576–591, int8, `whole_a[0:m, 0:k]` at L582) exercises it only on the architecture CI runs; `test_cutlass_fp8_gemm_padded` (L179) pads B through the Python linear path | open; the #55534 question, addressed by #55537 (reject) vs #56248/#56480/#56659 (honor) |
| S5 | Operand base pointers 16-byte aligned (sliced views with a storage offset) | **not checked at entry on `main`**; #55537 adds it | none | open, same decision as S4 |
| S6 | `a` dtype FP8 e4m3 or int8 | helper L26–29, **standard branch only** | dtype tests | checked on the standard branch |
| S7 | `a` **and** `b` dtypes on the SM90 standard path | FP8: `a` and `b` each required to be `Float8_e4m3fn` in `c3x/scaled_mm_sm90_fp8_dispatch.cuh` (epilogue L388–391, dispatcher L291–294). int8: `a` and `b` each required to be `Char` in `c3x/scaled_mm_sm90_int8_dispatch.cuh` (epilogue L151–152, dispatcher L97–98) | dtype tests | checked on the SM90 standard path; the blockwise branch is B2 |
| S8 | Output dtype bf16/fp16 | SM90 FP8 epilogue L393–399; SM90 blockwise `c3x/scaled_mm_blockwise_sm90_fp8.cu` L12–17 | output-dtype tests | checked on SM90 |
| S9 | Scale dtype float32 | helper L15–18 (both branches) | yes | checked |
| S10 | Per-tensor/token/channel scale counts; scales contiguous | helper L22–25 (by `numel`, not shape); `c3x/scaled_mm_sm90_fp8.cu` L12 | CUDA-graph and scale-shape tests | checked by count; `[M]` vs `[M,1]` not distinguished |
| S11 | Blockwise scales 2-D with group shapes `[1,128]`/`[128,128]` | helper L40–41 (dims), L43–51 (shapes, only when SM ≥ 90) | blockwise tests (SM ≥ 90) | checked on SM ≥ 90, shape only (see B1) |
| S12 | Bias length N, contiguous, 1-D, dtype = output dtype | entry L213–216 (length, contiguity, 1-D); dtype in SM90 FP8 `c3x/scaled_mm_sm90_fp8.cu` L14–16 and in `_azp` (entry L417) | bias tests | checked on SM90 standard; the blockwise branch rejects bias (helper L54) |
| S13 | All tensors on one device | not checked | device tests use one device | likely a loud fault, low priority |

### SM90 blockwise path

`c3x/scaled_mm_blockwise_sm90_fp8_dispatch.cuh`, reached from the helper's blockwise branch (helper L39–56). The dispatcher chooses the A/B-swapped kernel when `a.size(0) % 4 != 0` (L210).

| # | Requirement | Where it is checked (pin) | Existing test coverage | Status |
| --- | --- | --- | --- | --- |
| B0 | Operand leading strides | **ignored**: A, B and C strides are rebuilt packed from the shape (`make_cute_packed_stride`, L155–162) | packed operands only | known #55534 class on this path; #56248 targets it. Q6 regression vector, not a new finding |
| B1 | Scale memory layout: the config requires A's scales MN-major (column-major `[M, K/128]`) and B's scales K-major (L58–64, roles swapped with `swap_ab`) | **not checked**: layouts are derived from shape alone (`tile_atom_to_shape_SFA/SFB`, L164–169); the helper checks scale dims and shape only (helper L40–51) | tests build the required layout by hand (`test_cutlass_scaled_mm.py` L103–106; `test_block_fp8.py` L200–202) | **observed** in one preregistered H800 run: wrong-layout scales gave `rel_diff` 0.90–1.12 with no error, required layouts 0.0014 |
| B2 | Operand dtypes on the blockwise branch | **no `a`/`b` dtype check found**: the helper's dtype checks are in the standard branch only (L26–29); the blockwise branch checks scale dims, shapes and bias (L39–56); the SM90 blockwise file checks only the output dtype (L12–17); the dispatcher casts both operand pointers to its e4m3 element type (L171–172) | FP8 e4m3 only | observed in one H800 run: e5m2 or int8 operands yielded wrong output without an error ([result](../../experiments/kernel-operand-contracts/B2_B0_H800_RESULT_2026-09-29.md)). Whether the op should reject them remains an API-contract question |

Where the B1 convention lives: `per_token_group_quant_fp8` defaults to `column_major_scales=False` (`vllm/model_executor/layers/quantization/utils/fp8_utils.py` L556); the CUTLASS, FlashInfer and DeepGEMM linear kernels each pass `column_major_scales=True` (`vllm/model_executor/kernels/linear/scaled_mm/cutlass.py` L285, `flashinfer.py` L282, `deep_gemm.py` L43), and the PyTorch kernel passes it only on CUDA-like platforms (`pytorch.py` L288). The CUTLASS blockwise kernel passes `Bs.T` as a transposed view (`cutlass.py` L321–327). Each caller supplies the layout; the op does not enforce it.

## `cutlass_moe_mm` (SM90 path read; SM100 shares the pointer and stride code). All rows untested.

Entry: `scaled_mm_entry.cu` L272–304 dispatches by SM with no input checks. SM90: `moe/grouped_mm_c3x_sm90.cu` → `moe/grouped_mm_c3x.cuh` `cutlass_group_gemm_caller` → `moe/get_group_starts.cuh`.

| # | Requirement | Where it is checked (pin) | Existing test coverage | Status |
| --- | --- | --- | --- | --- |
| M1 | `a`, `b` FP8 e4m3; scales float32; `expert_offsets` int64 | `grouped_mm_c3x_sm90.cu` L122–127; `get_group_starts.cuh` L61–71 | yes | checked |
| M2 | Output bf16 or fp16 | `get_group_starts.cuh` L83–88 (raises) | fp16 only | checked (loud) |
| M3 | A packed: each expert's A pointer is `base + expert_offset * k` with `k = a.size(1)` (`get_group_starts.cuh` L24), while the kernel reads per-expert strides from `a_strides` | not checked that `a.stride(0) == k` or that `a_strides[e]` matches | `test_cutlass_fp8_group_gemm` (`test_cutlass_scaled_mm.py` L655) uses packed A only | gap candidate |
| M4 | B packed per expert: pointer `base + e * k * n` (L25), strides from `b_strides` | not checked | packed only | gap candidate |
| M5 | Output packed: pointer `base + expert_offset * n` (L26), strides from `c_strides` | not checked | packed only | gap candidate |
| M6 | `a/b/c_strides` int64 with ≥ `num_experts` entries: the kernel reinterprets `data_ptr()` as one `int64_t` stride per expert (`grouped_mm_c3x.cuh` L129–131, L142–150, L167–169) | not checked (dtype, length, contiguity) | int64 only | gap candidate |
| M7 | `problem_sizes` int32 `[E, 3]`, contiguous, consistent with `expert_offsets` | not checked; reinterpreted as `UnderlyingProblemShape*` (L133–136) | consistent inputs only | gap candidate |
| M8 | `per_act_token` / `per_out_ch` agree with the scale shapes | pointer offsets use flags recomputed from `numel` (`get_group_starts.cuh` L74–75); the epilogue uses the caller's flags (`grouped_mm_c3x.cuh` L165–166); no agreement check | tests pass consistent flags | gap candidate; reachability unverified |
| M9 | `num_experts` ≤ 1024 | not checked; the starts kernel launches `<<<1, num_experts>>>` without a launch-error check (`get_group_starts.cuh` L36) | 8 and 64 experts | low priority |
| M10 | Per-group alignment of M/N/K and pointers | `can_implement` is called (`grouped_mm_c3x.cuh` L182), but group sizes live on the device | aligned sizes only | unverified what CUTLASS validates for grouped problems |
| M11 | Supported-capability report matches dispatch | `cutlass_group_gemm_supported` reports the SM100 path for 100–119 (entry L183); `cutlass_moe_mm` dispatches SM100 only for 100–109 (L285) | none | inconsistency that fails loudly; verify the SM100 MoE build's architectures before calling it a bug. The error text says `cutlass_scaled_mm` (L302) |

Reachability of M3–M8: the production caller (`vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py` L223 and L249) passes buffers from `_resize_cache` or `moe_permute` (packed), int64 strides built with `torch.full((e,), k|n|2n)` (L298–306, L1283–1287), and flags from the quantization config. These rows are reachable only by other callers of the public `ops.cutlass_moe_mm` or by future layout changes.

Schema: `cutlass_scaled_mm` and `cutlass_moe_mm` declare their output mutable (`Tensor!`, `csrc/libtorch_stable/torch_bindings.cpp` L114, L136). `cutlass_scaled_mm` has `opcheck` coverage (`test_cutlass_scaled_mm.py` L115, L146); `cutlass_moe_mm` does not. Low priority.

## Testing

- B1: done (see result above).
- B2 and B0: one bounded H800 session completed under [the frozen protocol](../../experiments/kernel-operand-contracts/B2_B0_H800_PROTOCOL_2026-09-29.md); see the [result](../../experiments/kernel-operand-contracts/B2_B0_H800_RESULT_2026-09-29.md).
- M3–M11: not booked. Revisit only if a maintainer engages with the operand-layout decision.
- Upstream use: re-pin to current `main`, duplicate-check each confirmed row, and report blockwise findings on #55534 as short comments; nothing goes into the #55537 diff without maintainer agreement.

## Appendix: pinned blob SHAs (`91d7324c`)

| Path | Blob |
| --- | --- |
| `csrc/…/cutlass/scaled_mm_entry.cu` | `51f84d2ffd95d1a2dd5d1d4273c52de6fca93ec1` |
| `csrc/…/cutlass/c3x/scaled_mm_helper.hpp` | `913436186c3a4abcb9f68c050553e278a1ae6370` |
| `csrc/…/cutlass/c3x/scaled_mm_blockwise_sm90_fp8_dispatch.cuh` | `529b28ceece5d4a50a30970e03689a6cb244fc1a` |
| `csrc/…/cutlass/c3x/scaled_mm_blockwise_sm90_fp8.cu` | `a97909d37ae8f699075ee6c31497681e8bea3a83` |
| `csrc/…/cutlass/c3x/scaled_mm_sm90_fp8_dispatch.cuh` | `2fae3016c3092a7e8ed8e43519ee32183ad57490` |
| `csrc/…/cutlass/c3x/scaled_mm_sm90_int8_dispatch.cuh` | `a2ec816c9c937016fccbfbf729ed91b203261a52` |
| `csrc/…/cutlass/c3x/scaled_mm_sm90_fp8.cu` | `e86c9bd48d3fc40512c080477907902bb27ac3ac` |
| `csrc/…/cutlass/moe/grouped_mm_c3x_sm90.cu` | `d494bfb9d08cb5708b0d1ee877acf1175fb73ef2` |
| `csrc/…/cutlass/moe/grouped_mm_c3x.cuh` | `b4cd520e9679792f26179730499cea7c46dd41e8` |
| `csrc/…/cutlass/moe/get_group_starts.cuh` | `6942c4b48ccbca248d6fa734f96f36632ad08b8d` |
| `csrc/libtorch_stable/torch_bindings.cpp` | `a76852df89e5329c839b8fc734be86d55d942723` |
| `vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py` | `cd7da3e1e3a0c702d7ece0b6eb52683d92efcdf4` |
| `vllm/model_executor/kernels/linear/scaled_mm/cutlass.py` | `da8c69eed0a2ce640a358ac89497bb1e3631384d` |
| `vllm/model_executor/layers/quantization/utils/fp8_utils.py` | `79859a69d44055c1bfc512ee67b33ebde7b2f738` |
| `tests/kernels/quantization/test_cutlass_scaled_mm.py` | `9e1d6d2cc76cb0ca2740a29b6c9a9b0e69b7feb4` |
| `tests/kernels/quantization/test_block_fp8.py` | `038b94bd2d741f995a3857c2608f1bf36f2d0ace` |
