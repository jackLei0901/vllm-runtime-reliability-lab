---
model_cells: []
status: active
upstream_exit: follow-up to vLLM PR 55537 discussion; not posted
last_scored: never
---

# Kernel operand contracts (quantized GEMM)

Kernel-correctness track, outside the M1–M6 runtime model. Source inventory:
[QUANT_GEMM_OPERAND_CONTRACT_INVENTORY_2026-09-29](../../docs/kernel/QUANT_GEMM_OPERAND_CONTRACT_INVENTORY_2026-09-29.md).
Nothing here has run.

`test_blockwise_scale_layout_sm90.py` tests inventory row B1: whether the SM90
blockwise `cutlass_scaled_mm` path honors the memory layout of its scale
tensors. Its predictions are written in the module docstring and must be
committed before the first run. Each case records `outcome` (`correct`,
`wrong`, `ambiguous`, `raised`) and `rel_diff` against a float reference built
from the same FP8 values; a test passes only when the frozen prediction holds.

Run on an H800/H100 (SM90) in the vLLM environment under test:

```bash
pytest -q -rA experiments/kernel-operand-contracts/test_blockwise_scale_layout_sm90.py --junitxml=b1.xml
```

Record with the result: the vLLM commit actually installed (preferably current
`main`, not the inventory pin), CUDA and driver versions, GPU name, and the
SHA-256 of the test file. A confirmed B1 is latent in vLLM's current callers
(they all request the required layout); it goes to the #55537 discussion as one
sentence, not into that diff.
