# Phase 2 Gate 0 result — 2026-09-12

Status: **PASS**.

The campaign ran after `verify_phase2_freeze.py` reported:

```text
PASS (15 frozen Phase 2 files)
```

## Environment

- 2 x NVIDIA GeForce RTX 4090;
- PyTorch `2.13.0+cu130`;
- PyTorch git revision `cf30153c4c131c8164ee7798e5022d810682e2cb`;
- CUDA runtime `13.0`;
- frozen reproducer SHA-256
  `8a7d2afdbf39c2a02d496de88b0ca5392dbcbc34388d5daea2ac182f539f3a46`.

## Result

| Case | Trials | Dtypes observed on each rank | Exit | Classification |
| --- | ---: | --- | ---: | --- |
| `unused-parameter` | 3/3 | bf16 only, four gradient slots | 0 | `completed_uniform_bf16` |
| `forced-mixed-gradient` | 3/3 | bf16 + fp32, two gradient slots | 1 | `mixed_gradient_dtype_assertion` |

The fail-closed verifier reported:

```text
PASS (2 cases x 3 trials; torch=2.13.0+cu130)
```

All six trials recorded both rank identities, retained no raw output and left
no tracked process alive.

## Interpretation

This confirms the proposed dtype mechanism for the earlier two-rank null: with
`param_dtype=bf16`, the real gradients and the unused-parameter placeholder
that reach `foreach_reduce` are all bf16. `reduce_dtype=fp32` does not create a
mixed input set because that conversion occurs after the uniformity check.

The positive control also establishes that the probe and classifier detect the
same bf16+fp32 condition and exact assertion when such a mixed input set is
actually present. It does not explain what produced fp32 real gradients in the
original organic workload.

Structured records are under
`results/pytorch-unused-grad-dtype-gate0-20260912/`.

## Record SHA-256

```text
forced-mixed-gradient-trial-1.json 58ce8d01c8118108232dbab6b9b82f610e50a5b22188639efae8cc2a7bc9f46e
forced-mixed-gradient-trial-2.json f8a37e3392b566e6104ea3076e9f4fcd4fc74591e9eee5623ee4e2264b6c1315
forced-mixed-gradient-trial-3.json b8f43b61fb495a4fa059098a8bfa40a2129a4adaac8e658e2594cd7525044f16
unused-parameter-trial-1.json 881c2b383cdde08209b49d55020fd2ddb18974e9816314f690e3452b1668af88
unused-parameter-trial-2.json 06cbbab0c3ec55cdb08ba4bbf203cb8ebdc285e9bc0d907adaf0d2577ee760fc
unused-parameter-trial-3.json 4440ca1eb4e2e79565d64fe68b3e45815ea0abee636092c2636f389ea5d0d6db
```
