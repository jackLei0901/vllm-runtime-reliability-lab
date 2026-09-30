---
model_cells: []
status: active
upstream_exit: B1 comment posted on vLLM issue 55534 (2026-09-29); B2/B0 observed, not yet posted
last_scored: 2026-09-29
---

# Kernel operand contracts (quantized GEMM)

Kernel-correctness track, outside the M1–M6 runtime model. Source inventory:
[QUANT_GEMM_OPERAND_CONTRACT_INVENTORY_2026-09-29](../../docs/kernel/QUANT_GEMM_OPERAND_CONTRACT_INVENTORY_2026-09-29.md).

| Row | File | Status |
| --- | --- | --- |
| B1: blockwise scale layout | `test_blockwise_scale_layout_sm90.py` (frozen at Lab `8ab51bb`) | Observed in one preregistered H800 run ([result](B1_H800_RESULT_2026-09-29.md)). Reported on #55534 in [issuecomment-5893471238](https://github.com/vllm-project/vllm/issues/55534#issuecomment-5893471238). **Do not rerun its eight-case matrix in the B2/B0 booking.** |
| B2: blockwise operand dtype (new question) and B0: padded views (known #55534 class, Q6 regression vector) | `test_blockwise_dtype_and_stride_sm90.py` | One H800 run, 16/16 as predicted ([protocol](B2_B0_H800_PROTOCOL_2026-09-29.md), [result](B2_B0_H800_RESULT_2026-09-29.md), [中文](B2_B0_H800_RESULT_2026-09-29.zh-CN.md)). Not yet posted upstream. |

Supporting scripts: `check_anchors.py` verifies every cited source line against
the pinned vLLM blobs, and with `--local` gates a checkout's source identity;
`select_rerun.py` selects the only cases the protocol's single rerun may
execute. All three are stdlib-only apart from the test itself.

Local repair execution packet v1 (local freeze; public push required;
no compilation/GPU result):
[SM90 build/test packet](sm90-contract-repair/README.md),
[中文](sm90-contract-repair/README.zh-CN.md). This does not alter historical
results or authorize an upstream post or GPU booking.

## Reproduction: B2/B0 only

Follow the protocol exactly. In short, on an SM90 host inside the vLLM
environment under test:

```bash
python experiments/kernel-operand-contracts/check_anchors.py
python experiments/kernel-operand-contracts/check_anchors.py --local /path/to/vllm
pytest -p no:randomly --collect-only -q experiments/kernel-operand-contracts/test_blockwise_dtype_and_stride_sm90.py > collected.txt  # must list 16 cases
KOC_RECEIPTS=b2_b0_receipts.jsonl pytest -p no:randomly -q -rA -o junit_family=xunit1 experiments/kernel-operand-contracts/test_blockwise_dtype_and_stride_sm90.py --junitxml=b2_b0.xml
```

A rerun happens only if `select_rerun.py` exits 0 with a non-empty list; an
empty list means stop without invoking pytest. See the protocol.

Record the installed vLLM commit (40 hex characters), CUDA and driver
versions, GPU name, the compiled extension digest, and the file digests. B2 is
reported as new evidence; B0 is regression coverage, not a new finding. Both
are latent in vLLM's current callers.
