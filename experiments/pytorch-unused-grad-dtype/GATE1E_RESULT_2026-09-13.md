# Gate 1e GPU result — 2026-09-13

Verdict: **mechanism and termination PASS; strict capture FAIL; overall gate
FAIL-CLOSED.**

Gate 1e ran from a clean Linux checkout at commit
`4ec3bfe6c052c2f0076a4a2ecba1edbdf71fb0d2` after the Phase 2, Gate 1b, Gate
1c, Gate 1d and Gate 1e freeze verifiers passed with 15, 7, 7, 9 and 10 files.
The environment was two NVIDIA GeForce RTX 4090 GPUs, PyTorch `2.13.0+cu130`
at git revision `cf30153c4c131c8164ee7798e5022d810682e2cb`, and py-spy 0.4.2.

## Frozen outcomes

| arm | trials | mechanism | termination | stack capture | Flight Recorder dumps | lifecycle |
| --- | ---: | --- | --- | --- | --- | --- |
| control | 3 | 3/3 `completed_symmetric_fp32` | 3/3 normal | not requested | 0 as expected | 3/3 no tracked orphans |
| affected | 3 | 3/3 `rank1_assertion_rank0_barrier_wait` | 3/3 frozen wall-bound path | 3/3 both ranks | **0/3 complete pairs** | 3/3 no tracked orphans |

Affected trials lasted 60.755, 60.694 and 60.609 seconds. In every trial the
stack evidence placed rank 0 at the post-reduce barrier and rank 1 in
`destroy_process_group()`. In every trial exactly one decodable Flight Recorder
file was produced, always for rank 0. Rank 1 was absent from all three dump sets.

## Verification boundary

The predeclared verifier rejects the result with:

```text
AssertionError: both rank dumps must decode before participation is inferred
```

This is the intended fail-closed behavior. A rank-0-only dump cannot distinguish
"rank 1 did not participate in the collective" from "rank 1's diagnostic dump
was unavailable." Therefore the matching per-rank mechanism markers and stacks
do not establish the strict Flight Recorder join required by the protocol.

The repeated result does establish a narrower observation: in this environment,
the local dtype assertion on rank 1 consistently surfaced as rank 0 waiting at
the later barrier, while rank 1 remained in process-group teardown until the
external wall bound. External stack sampling captured both sides. The campaign
does not explain why rank 1 failed to produce a Flight Recorder dump.

## Evidence handling and next step

Only the six allow-listed JSON summaries are retained in
`results/pytorch-unused-grad-dtype-gate1e-20260913/`. Raw launcher output, raw
py-spy output and Flight Recorder pickle files were deleted with their temporary
directories. All six trials report `no_tracked_orphans: true`.

No additional GPU repetition is justified: the missing-rank pattern was stable
across all three affected trials. The next step is source-level investigation of
rank-1 dump propagation and teardown behavior. Any changed capture mechanism or
acceptance rule requires a new freeze; this result must not be reinterpreted as
a successful two-rank Flight Recorder join.
