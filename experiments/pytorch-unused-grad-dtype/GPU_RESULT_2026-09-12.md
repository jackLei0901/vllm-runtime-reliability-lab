# Two-rank FSDP2 unused-gradient dtype result — 2026-09-12

## Verdict

**The predeclared reproduction hypothesis was not supported.** Plain two-rank
FSDP2 with one conditionally unused parameter group completed in all three
affected trials. Pipeline parallelism, microbatch accumulation, or another
property of the original parameter grouping remains necessary to reproduce the
four-GPU mixed-gradient-dtype assertion.

This is not evidence that the original defect is absent. The original
four-GPU control reproduced the assertion in 3/3 trials on the same PyTorch
release; this experiment only removes several conditions from that workload.

## Environment

- GPUs: 2 x NVIDIA GeForce RTX 4090, 24 GiB
- driver: 580.105.08
- Python: 3.12.3
- PyTorch: 2.13.0+cu130
- PyTorch git version: `cf30153c4c131c8164ee7798e5022d810682e2cb`
- CUDA runtime reported by PyTorch: 13.0
- reproducer SHA-256:
  `a883752878bd2e2d6d008592f7f4dd18b0e6ab9e66137772e5bcd235ff272a3d`

## Frozen arms

The control and affected arms used the same model, seed, FSDP2 mixed-precision
policy, optimizer, two backward steps, timeout and process launcher. The only
behavioral difference was whether rank 1 executed the conditional linear layer.
Rank 0 executed it in both arms.

| Arm | Trials | Completed | Target assertion | Orphans |
| --- | ---: | ---: | ---: | ---: |
| control | 3 | 3 | 0 | 0 |
| affected | 3 | 3 | 0 | 0 |

Trial duration was 5.929–8.843 seconds. All six records captured two rank
identities, exited zero, emitted the arm-specific success marker and reported
no tracked orphan.

## Verification outcome

The expectation was declared as `affected=reproduced` before execution. The
verifier rejected the observed matrix with exit code 1:

```text
AssertionError: affected trials do not match predeclared expectation reproduced
```

A second invocation with `affected=not-reproduced` passed only to demonstrate
that the six structured records form a complete, internally consistent matrix:

```text
PASS (3 control, 3 affected; affected=not-reproduced;
torch=2.13.0+cu130; source=a883752878bd)
```

That descriptive pass does not replace or retroactively alter the failed
predeclared hypothesis.

## Interpretation boundary

Supported:

- a single ordinary FSDP2 backward path is insufficient for this reproducer;
- the four-GPU assertion is not explained solely by one rank having an unused
  parameter under the stated mixed-precision policy; and
- the runner's success, redaction and lifecycle contracts held in all trials.

Not supported:

- whether pipeline parallelism itself is required;
- that microbatch accumulation is the cause;
- that the issue is fixed in PyTorch 2.13;
- that a current nightly should pass; or
- that the original organic issue has been diagnosed or resolved.

## Next discriminating experiment

Keep the two-rank topology and introduce controlled multiple-backward/microbatch
accumulation using FSDP2's `set_is_last_backward` contract. Preserve the current
single-backward arm as the negative control. Run nightly only after the affected
2.13 arm has a non-circular positive result; otherwise a nightly success cannot
distinguish a fix from a reproducer that never exercised the defect.
