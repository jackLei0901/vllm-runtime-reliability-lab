# Quantized GEMM operand-contract inventory: `cutlass_scaled_mm` and `cutlass_moe_mm`

Status: first-pass draft, 2026-09-29. **Source reading only**: nothing here has run on a GPU, and every "gap" is a hypothesis for a negative-control test, not an observed defect. Kernel-correctness track, not the Lab runtime model. Chinese text not yet written.

Pin: vLLM `main` at `91d7324cb19d301c72d849e457221ee8dd645024`, read from a local checkout whose only extra commit changes a test file. The tracked-tree cleanliness of that checkout was not re-verified; re-pin to current `main` before any upstream use. Paths are under `csrc/libtorch_stable/quantization/w8a8/cutlass/` unless stated.

Purpose: evidence for a possible follow-up on operand contracts for quantized GEMM custom ops, and input to the Q6 blockwise-FP8 negative-control vector. Scope of this pass: the two entry points, their shared helpers, the SM90 grouped and blockwise paths, the Python callers and the existing tests. The per-architecture standard FP8/int8 functions are not yet read (rows marked *unverified*).

**Reachability summary:** every gap below is *latent* at this pin. vLLM's production callers satisfy each requirement by convention, so no production-reachable silent-wrong-result bug was found in this pass. The value claimed is narrower: several requirements live only in caller convention, and cheap host-side checks would turn a future caller's mistake from silent into loud.

## `cutlass_scaled_mm`

Entry: `scaled_mm_entry.cu` L197–270, then `c3x/scaled_mm_helper.hpp` `dispatch_scaled_mm` for SM90/100/120.

| # | Requirement | Where it is checked | Existing test coverage | Status |
| --- | --- | --- | --- | --- |
| S1 | `a`, `b`, `c` are 2-D and shape-conformal | entry L203–205 | all GEMM tests | checked |
| S2 | `a` and `c` row-major inner dimension (`stride(1) == 1`), `b` column-major (`stride(0) == 1`) | entry L208–209 | implicit in all tests | checked |
| S3 | `c.stride(0)` and `b.stride(1)` multiples of 16 | entry L210–211 | padded-B test | checked; the comment says "16 Byte", but the check is 16 *elements*, stricter than 16 bytes for 16-bit outputs (not a bug) |
| S4 | `a.stride(0)` (A's leading stride) honored or rejected | **not checked at entry** | `test_cutlass_subset` (int8, `whole_a[0:512, 0:512]` of 1024) exercises it only on the architecture CI runs; `test_cutlass_fp8_gemm_padded` pads B through the Python linear path | open; the #55534 question, addressed by #55537 (reject) vs #56248/#56480/#56659 (honor) |
| S5 | Operand base pointers 16-byte aligned (sliced views with a storage offset) | **not checked at entry on `main`**; #55537 adds it | none | open, same decision as S4 |
| S6 | `a` dtype FP8 e4m3 or int8 | helper L26–29 (`a` only) | dtype tests | checked for `a` |
| S7 | `b` dtype matches `a` | **not checked in the entry or helper** | none | *unverified*: read the per-arch FP8/int8 functions |
| S8 | Output dtype supported (bf16/fp16) | not in entry or helper; SM90 blockwise checks it (`scaled_mm_blockwise_sm90_fp8.cu` L17) | output-dtype tests | *unverified* for the standard per-arch paths |
| S9 | Scale dtype float32 | helper L15–18 | yes | checked |
| S10 | Per-tensor/token/channel scale counts; scales contiguous | helper L22–25 (by `numel`, not shape) | CUDA-graph and scale-shape tests | checked by count; `[M]` vs `[M,1]` not distinguished |
| S11 | Blockwise scales 2-D with group shapes `[1,128]`/`[128,128]` | helper L40–51, shape check only when SM ≥ 90 | blockwise tests (SM ≥ 90) | checked on SM ≥ 90 (shape only; see B1) |
| S12 | Bias length N, contiguous, 1-D | entry L213–216 | bias tests | checked; bias dtype vs output dtype is checked in `_azp` (L417) but not in `cutlass_scaled_mm`, *unverified* in kernels |
| S13 | All tensors on one device | not checked | device tests use one device | likely a loud fault, low priority |

### SM90 blockwise path

`c3x/scaled_mm_blockwise_sm90_fp8_dispatch.cuh`, reached from the helper's blockwise branch.

| # | Requirement | Where it is checked | Existing test coverage | Status |
| --- | --- | --- | --- | --- |
| B0 | Operand leading strides | **ignored**: A, B and C strides are rebuilt with `make_cute_packed_stride` from the shape (L155–160) | packed operands only | the #55534 class on the blockwise path; #56248 targets it |
| B1 | **Scale memory layout**: the config requires A's scales MN-major (column-major for `[M, K/128]`) and B's scales K-major (L58–64, swapped when `swap_ab`) | **not checked**: the layouts are derived from shape alone (`tile_atom_to_shape_SFA/SFB`, L164–169); the helper checks only scale dims and shape (helper L40–51) | tests build the required layout by hand (`test_cutlass_scaled_mm.py` L103–106, "make scales M-major for blockwise quant"; `test_block_fp8.py` L200–203, "CUTLASS uses column-major format for scales") | gap candidate; **prediction**: row-major activation scales with M > 1 and K/128 > 1 give silently wrong output |

Where the B1 convention lives: `per_token_group_quant_fp8` defaults to `column_major_scales=False` (`vllm/model_executor/layers/quantization/utils/fp8_utils.py` L556); the CUTLASS, FlashInfer and DeepGEMM linear kernels each pass `column_major_scales=True` (`kernels/linear/scaled_mm/cutlass.py` L285, `flashinfer.py` L282, `deep_gemm.py` L43), and the PyTorch kernel passes it only on CUDA-like platforms (`pytorch.py` L288). The CUTLASS blockwise kernel passes `Bs.T` as a transposed view (`cutlass.py` L321–327). The requirement is satisfied by each caller, not enforced by the op.

## `cutlass_moe_mm` (SM90 path read; SM100 shares the pointer and stride code)

Entry: `scaled_mm_entry.cu` L272–304 dispatches by SM with **no input checks**. SM90: `moe/grouped_mm_c3x_sm90.cu` → `moe/grouped_mm_c3x.cuh` `cutlass_group_gemm_caller` → `moe/get_group_starts.cuh`.

| # | Requirement | Where it is checked | Existing test coverage | Status |
| --- | --- | --- | --- | --- |
| M1 | `a`, `b` FP8 e4m3; scales float32; `expert_offsets` int64 | `grouped_mm_c3x_sm90.cu` L122–127; `get_group_starts.cuh` L61–71 | yes | checked |
| M2 | Output bf16 or fp16 | `get_group_starts.cuh` L81–88 (raises) | fp16 only | checked (loud) |
| M3 | **A packed**: each expert's A pointer is `base + expert_offset * k` with `k = a.size(1)` (`get_group_starts.cuh` L24), while the kernel separately reads each expert's row stride from `a_strides` | **not checked** that `a.stride(0) == k` or `a_strides[e] == a.stride(0)` | `test_cutlass_fp8_group_gemm` uses packed A only | gap candidate: a padded A would get packed-offset pointers with padded strides |
| M4 | **B packed per expert**: pointer `base + e * k * n` (L25), strides from `b_strides` | not checked | packed only | gap candidate, as M3 |
| M5 | **Output packed**: pointer `base + expert_offset * n` (L26), strides from `c_strides` | not checked | packed only | gap candidate, as M3 |
| M6 | `a/b/c_strides` are int64 with at least `num_experts` entries: the kernel reinterprets `data_ptr()` as one `int64_t` stride per expert (`grouped_mm_c3x.cuh` L129–131, L142–150, L167) | **not checked** (dtype, length, contiguity) | int64 only | gap candidate: int32 strides would be misread |
| M7 | `problem_sizes` is int32 `[E, 3]`, contiguous, consistent with `expert_offsets` | **not checked**; reinterpreted as `UnderlyingProblemShape*` (L133–136) | consistent inputs only | gap candidate |
| M8 | `per_act_token` / `per_out_ch` agree with the scale shapes | pointer offsets use flags **recomputed** from `numel` (`get_group_starts.cuh` L74–75); the epilogue uses the **caller's** flags (`grouped_mm_c3x.cuh` L165–166); no agreement check | tests always pass consistent flags | gap candidate; reachability *unverified* |
| M9 | `num_experts` ≤ 1024 | not checked; the starts kernel launches `<<<1, num_experts>>>` with no launch-error check (L36) | 8 and 64 experts only | low priority today |
| M10 | Per-group alignment of M/N/K and pointers | `can_implement` is called (L182), but group sizes live on the device | aligned sizes only (multiples of 16) | *unverified* what CUTLASS validates for grouped problems |
| M11 | Supported-capability report matches dispatch | `cutlass_group_gemm_supported` reports the SM100 path for capability 100–119 (entry L182–185); `cutlass_moe_mm` dispatches SM100 only for 100–109 (L285) | none | inconsistency, but it fails loudly (`NOT_IMPLEMENTED`); verify which architectures the SM100 MoE build includes before calling it a bug. The error text also says `cutlass_scaled_mm` (L302). |

Reachability of M3–M8: the production caller (`vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py` L223–262) passes buffers created by `_resize_cache` or `moe_permute` (packed), int64 strides built with `torch.full((e,), k|n|2n)` (L298–306, L1283–1290), and flags from the quantization config. These gaps are reachable only by other callers of the public `ops.cutlass_moe_mm` or by future layout changes.

Schema check: `cutlass_scaled_mm` and `cutlass_moe_mm` both declare their output as mutable (`Tensor!`, `csrc/libtorch_stable/torch_bindings.cpp` L114, L136). `cutlass_scaled_mm` has `opcheck` coverage (`test_cutlass_scaled_mm.py` L115, L146); `cutlass_moe_mm` does not. Low priority.

## H800 (SM90) test plan: freeze predictions before running

Each case compares against a float reference built from the same quantized values, and is scored `supported` (prediction held), `refuted`, or `unscored` (setup failure, or the case never reached the intended kernel).

| Case | Input | Control | Prediction |
| --- | --- | --- | --- |
| B1-a | Blockwise FP8, M > 1, K/128 > 1, activation scales **row-major** (`per_token_group_quant_fp8(..., column_major_scales=False)`) | same data with column-major scales | silent wrong output (mismatch against reference, no error) |
| B1-b | As B1-a with M = 1 or K = 128 | — | correct (the two layouts coincide), confirming the test isolates layout |
| B1-c | B scales passed contiguous `[N/128, K/128]` instead of the `Bs.T` view | `Bs.T` | silent wrong output |
| M3 | Grouped GEMM with A as a padded view (`stride(0) > k`) and `a_strides` set to the padded stride | packed A | silent wrong output |
| M6 | Packed inputs with `a/b/c_strides` as int32 | int64 strides | silent wrong output or illegal memory access |
| M8 | Packed inputs, `per_act_token=True` with a single scalar activation scale | consistent flags | wrong or out-of-bounds scale reads |

Notes: `cutlass_moe_mm` needs SM90+, so the 4090 cannot run the M cases. For S4/S5 reuse #55537's contract cases rather than re-deriving them. Record the vLLM commit, CUDA and driver versions, and the exact test file digest with each result. Any confirmed row still needs a re-pin to current `main` and a duplicate check before upstream use, and goes to the #55537 discussion as a sentence, not into that diff.
