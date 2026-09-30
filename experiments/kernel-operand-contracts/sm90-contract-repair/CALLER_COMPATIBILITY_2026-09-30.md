# Caller layout audit at 7b054aca — not K5 serving validation

The [Chinese version](CALLER_COMPATIBILITY_2026-09-30.zh-CN.md) is authoritative.
Source read directly from a clean checkout at
`7b054aca96cea8be1369d651c3434ad140580b92`; not extrapolated from older main.

| Point | Evidence at the pin | Implication |
| --- | --- | --- |
| Activation allocation | [cutlass.py L276–286](https://github.com/vllm-project/vllm/blob/7b054aca96cea8be1369d651c3434ad140580b92/vllm/model_executor/kernels/linear/scaled_mm/cutlass.py#L276-L286) sets column-major scales and use_ue8m0=False; [QuantFP8 L37–44](https://github.com/vllm-project/vllm/blob/7b054aca96cea8be1369d651c3434ad140580b92/vllm/model_executor/layers/quantization/input_quant_fp8.py#L37-L44) defaults TMA alignment to False; group CUDA path passes these settings into fp8_utils | For 2-D x, [fp8_utils L614–617](https://github.com/vllm-project/vllm/blob/7b054aca96cea8be1369d651c3434ad140580b92/vllm/model_executor/layers/quantization/utils/fp8_utils.py#L614-L617) allocates `(K/128,M)` and permutes to strides `(1,M)`, accepted by A guard |
| Weight scales | [cutlass.py L313–327](https://github.com/vllm-project/vllm/blob/7b054aca96cea8be1369d651c3434ad140580b92/vllm/model_executor/kernels/linear/scaled_mm/cutlass.py#L313-L327) passes `Bs.T` | If Bs is packed contiguous `(Nblocks,Kblocks)`, Bs.T strides are `(1,Kblocks)`, accepted. Transpose alone does not establish all loaded weights are packed |
| Token padding | Cutlass block class has no get_output_padding override; block superclass also has none; [base.py L282–292](https://github.com/vllm-project/vllm/blob/7b054aca96cea8be1369d651c3434ad140580b92/vllm/model_executor/kernels/linear/base.py#L282-L292) returns None. Group quantization follows the per-group path, not the non-group padded op | Inspected caller does not create the hypothesized token-padded scale slice |
| Old Hopper helper | `git grep -n _padded_cutlass <pin> -- vllm` has no match | The older checkout's helper cannot be cited as a current caller. No fictional fixture is added for that absent path |

The weight preprocessing in `process_fp8_weight_block_strategy` returns the
weight_scale on CUDA without imposing contiguity. This audit therefore supports
the inspected normal packed allocation contract, not every loader, checkpoint,
custom caller or backend. TMA-aligned scales with M=258 and stride `(1,260)`
remain rejected when more than one Kblock is present; this setting is not enabled
by the inspected CUTLASS block quantizer.

[test_sm90_contract_r2.py](../../../tests/test_sm90_contract_r2.py) adds metadata
arithmetic controls and actual CPU Torch allocations matching the pinned 2-D
activation expression and packed Bs.T expression, including singleton shapes.
These tests evaluate the candidate's actual guard predicates; they do not run
the CUDA quantizer or import the serving class. Local CPU Torch is absent:
the real tensor test is explicitly skipped, not passed. Arithmetic and publication
checks pass. This new post-run test does not modify/rescore the frozen GPU matrix.

K5 is not funded, not passed. Its full-build, serving, loaded-weight and actual
CUTLASS-path witness requirements remain unmet. K6 overhead is also unscored;
independent K7 execution remains unobserved. Record partial completion and stop.
Source reasoning lowers the inspected false-rejection risk; it is not equivalent
to observing a valid model complete requests on the fix build.
