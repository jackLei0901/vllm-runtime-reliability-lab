# DRAFT — Why Flight Recorder missed the most important rank

**Publication state:** blocked on an explicit upstream outcome for
[pytorch/pytorch#197232](https://github.com/pytorch/pytorch/pull/197232).
The PR and its issue were still open on 2026-09-20. Do not publish this draft as
a completed upstream success until the gate in
[`../V0.2_RELEASE_PLAN.md`](../V0.2_RELEASE_PLAN.md) closes.

## The incident

A four-GPU torchtitan workload hung after one data-parallel rank hit a local
failure and its peer waited in a collective. Flight Recorder produced a dump
for rank 0 but not rank 1.

The easy interpretation was that rank 1 had not participated in the collective.
That interpretation treated absence of a diagnostic file as absence of a
participant. The lab refused that inference:

```text
producer missing != member missing
```

## The narrowing sequence

The investigation separated two questions that a single dump could not answer:

1. Was rank 1 still part of the failed execution?
2. Why could rank 1 not produce the requested diagnostic artifact?

External process state and both-rank stacks showed that rank 1 remained alive.
Privacy-bounded shutdown-stage flags then showed the relevant order: rank 1 had
stopped its Flight Recorder dump responder, entered communicator destruction,
and remained blocked there. Rank 0 later timed out, broadcast the dump request,
and wrote its own trace. Rank 1 was still a member of the failed run but was no
longer a functioning diagnostic producer.

The next gate removed torchtitan, the model, FSDP, and gradient behavior. A
standalone two-rank ProcessGroupNCCL reproducer retained the same missing-rank
behavior on the legacy backend. That moved the claim from “an FSDP hang had an
incomplete dump” to a specific diagnostic lifecycle gap in ProcessGroupNCCL.

## Upstream result

The minimized result became
[PyTorch #196968](https://github.com/pytorch/pytorch/issues/196968). The proposed
C++ change in [#197232](https://github.com/pytorch/pytorch/pull/197232) keeps a
narrow, bounded dump-signal responder alive while communicator destruction is
in progress.

The validation result is deliberately asymmetric:

| Arm | Runs | Result |
| --- | ---: | --- |
| Released base behavior | 3 | failed as predicted: rank 1 did not write a complete trace |
| Proposed fix build | 3 | passed: both ranks wrote complete, decodable traces |

These arms used different PyTorch builds, CUDA toolkits, and NCCL versions.
They validate the behavior gap and the proposed behavior on current source; they
are not a same-environment performance comparison.

## Upstream resolution placeholder

Replace this section only with a dated, linked maintainer outcome:

- **Outcome:** MERGED / DESIGN ACCEPTED VIA OTHER PATH / REJECTED OR SUPERSEDED
- **Date:** YYYY-MM-DD
- **Maintainer evidence:** immutable link
- **What changed from the proposal:** exact summary
- **Final claim the lab may make:** bounded sentence

Until then, the correct wording is “lab-originated issue with a validated open
fix proposal,” not “upstream fixed” or “merged.”

## What the lab contributed

- converted missing evidence from an assumption into a testable hypothesis;
- distinguished a missing producer from a missing participant;
- replaced an organic multi-component hang with a standalone mechanism;
- preregistered base/fix outcomes and preserved repeated results;
- proposed and validated an upstream C++ change without claiming acceptance.

The important result is not that the lab collected more logs. It is that its
evidence rule prevented a false inference and exposed a blind spot in the
diagnostic system itself.

## Reproduction and evidence

- [Gate 1g protocol](../../experiments/pytorch-unused-grad-dtype/GATE1G_PROTOCOL.md)
- [Gate 1g result](../../experiments/pytorch-unused-grad-dtype/GATE1G_RESULT_2026-09-13.md)
- [Upstream fix validation](../../experiments/pytorch-c10d-shutdown-dump/UPSTREAM_FIX_VALIDATION_2026-09-16.md)
- [Published derived evidence](../../results/organic-hang-20260912/README.md)
