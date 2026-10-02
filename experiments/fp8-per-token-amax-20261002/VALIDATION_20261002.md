# Native per-token FP8 validation summary

The one-line change replaces `x_max, _ = x.abs().max(dim=-1)` with `x_max = x.abs().amax(dim=-1)` in `QuantFP8.forward_native`. Evidence collection is closed; this summary does not claim a model-level speedup or whole-model output equivalence.

## Pins and scope

- Validated Python source: `b558f160a2c0abcb5902acc3c91a14c38a4af173`.
- Submission preparation main: `2a537887de7a103baa49bbfdce93e99d48d19441`.
- The target file at both pins has blob `161baf446d12e593d5c95e17211a515b5d65c776`.
- Source patch SHA-256: `a1df821202ee4314926e71d90ad5f9085b8402f2ee2639a5751717a670522cca`.
- Model: `RedHatAI/Qwen2.5-VL-3B-Instruct-FP8-dynamic`, revision `33cf8c95d465fe49c9d7864e75c9b0b8ce80eff1`; text hidden size 2048.
- Quality evaluation: one H800 PCIe, PyTorch 2.13.0+cu130; default compilation, TP=1, max model length 4096, independent caches and identical resolved configuration in both arms.

Existing parent-commit binaries at `4f1451679088e5832bce0965a254295c854aaa09` were reused, not rebuilt at the Python pin. Relevant per-token CUDA functions and wrappers, fused kernels and layernorm sources were unchanged; other FP8 code changed. Binary provenance is not whole-revision identity.

## Validation

- Eight selected native FP8 fusion cases passed. Seventy-two CUDA per-token quant tests passed as regression context; those CUDA tests do not exercise the changed Python line.
- Independent A/B/A codegen checks covered four standalone/synthetic-norm and symbolic/batch-1 contexts, with bitwise-equal cross-arm outputs and scales for tested inputs. BF16/FP32 bounded/unbounded special-value checks passed.
- Full pre-commit exited 0: 13 hooks passed and 20 were skipped for no applicable files, including a passing mypy hook.
- Model startup generated-code `max_with_index` call-text count: base 24, patch 0. This is a differential path witness, not a dynamic call count.

| GSM8K, 1,319 questions, 5 shots | Base | Patch |
| --- | ---: | ---: |
| Accuracy | 0.6178923427 | 0.6050037908 |
| Correct answers | 815 | 798 |
| Invalid answers | 1 | 0 |

Both arms passed the existing 0.60 threshold. Patch had 17 fewer correct answers. Requests were concurrent and generated metadata reports `batch_invariant=False`; run-to-run variation is plausible but was not measured. The difference remains unattributed. These scores do not establish accuracy equivalence or exclude regression.

## Offline generated-code audit

Thirteen distinct Triton kernel definitions paired between arms: five bodies were unchanged, and eight matched after an explicit normalization of temporary names and removal of index maintenance in the value-only reduction. All 13 declared launch metadata comparisons and all nine execution-wrapper comparisons matched. No added fusion, changed memory operation or removed dtype cast was found. Mutation controls detected a changed cast, load pointer and launch flag.

Selected autotune configurations, PTX/SASS and per-question outputs were not archived. Removing the index accumulator could change selected block sizes or warp counts and hence fused norm summation order; this is an untested hypothesis, not a detected difference.

Do not claim whole-model bitwise identity, confirmed batch noise, or impossibility of regression.

Quality archive SHA-256: `84257f1e021d26c227b68b02eb579620b8db409b2524810595f421c633a1b510`. Full pre-commit log SHA-256: `07bc5f233bc2aa9a13653b4cbd3815811f13273df1818ab604f5f37333183464`. Raw logs, generated code and host identifiers remain private.

Historical operator timings and reproduction scripts are in this directory. Their source pin is `378504a5`, not the submission main. All 21 shapes (hidden 4096/7168/14336, seven token counts) were classified GAIN, with 7.7–39.8% lower latency and maximum baseline spread 1.5%. Hidden 4096 alone showed 14.5–36.2% lower latency and maximum baseline spread 0.8%. No TPOT or default-serving speedup was measured.

Open #41427 also edits the reduction line for NaN handling. The value-only reduction applies equally to its `x_f` input; the later change needs rebasing at that site. This is overlapping source, not an existing implementation of this per-token performance change.
