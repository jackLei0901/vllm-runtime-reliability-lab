# v0.2.0 — Cross-process evidence and no-progress validation

v0.2.0 turns the lab's evidence method into a bounded workflow that another
engineer can run without a GPU. It adds `collect`, offline `verify`, and an
installed `replay` command while preserving `undetermined` as a valid outcome
when the evidence cannot support a stronger claim.

## Five-minute replay

```bash
python -m pip install .  # installs distribution vllm-runtime-dfx-lab
vllm-dfx replay results/vllm-zmq-backpressure-stage1-r3-20260916
```

The replay verifies the frozen #53859 four-cell evidence, file identities,
health-green no-progress claim, blocking-stack claim, fix-arm progress, and the
measured four-batch event-loss trade-off. It does not rerun the GPU experiment.

## Delivered in v0.2.0

- bounded `collect` bundles with operator-supplied PID identity, endpoint
  privacy, and always-closed observation windows;
- independent server-counter and opt-in client-request progress producers;
- explicit demand evidence and producer-conflict reporting;
- five closed verdicts with deterministic precedence: `process_missing`,
  `health_lost`, `progress_observed`, `alive_health_ok_no_progress`, and
  `undetermined`;
- offline, fail-closed verdict recomputation from closed-shape JSON;
- typed optional `py-spy` availability and producer identity while raw stacks
  remain private and non-decisional;
- a machine-checked registry that classifies every public leaf field as
  decisional or explicitly non-decisional;
- a compatibility projection that reports evidence missing from the legacy
  #53859 result instead of inventing native v0.2 observations; and
- `vllm-dfx replay` as the installed no-GPU entry point.

## Evidence-backed results

- Lab-originated [PyTorch #196968](https://github.com/pytorch/pytorch/issues/196968)
  established that a missing Flight Recorder dump did not mean a missing rank;
  [PR #197232](https://github.com/pytorch/pytorch/pull/197232) proposes the C++
  fix. Both were open on 2026-09-21.
- The lab independently validated reported
  [vLLM #53859](https://github.com/vllm-project/vllm/issues/53859) and proposed
  [PR #53883](https://github.com/vllm-project/vllm/pull/53883): `/health`
  remained 2xx while EngineCore token progress stopped under deterministic
  event-queue backpressure. This was not a lab-originated bug.
- Lab-originated [PyTorch #196996](https://github.com/pytorch/pytorch/issues/196996)
  reduced an apparent distributed hang to a single-GPU FSDP2 mixed-gradient
  dtype correctness failure.

## Publication boundary

The complete #196968 case study is not part of this release. Its publication
gate requires an explicit upstream outcome for #197232: merge, explicit design
acceptance with another landing path, or explicit rejection/supersession with a
documented reason. An open PR, passing CI, bot labels, or silence is not an
upstream outcome.

`#49869` is an independent upstream contribution and is not a lab discovery.
`#52178` is a separately found lifecycle bug for which the lab supplied
process-level validation.

## Known boundaries

This remains an alpha research tool, not a production monitor, automatic
process/rank discovery system, general cross-rank or cross-host joiner,
native-state classifier, remediation controller, or automatic root-cause
classifier. Stack availability and producer implementation cannot change a
verdict. Endpoint-only flat progress remains `undetermined` because repeated
2xx responses do not establish process liveness.

## Immutable release references

- [v0.2.0 source tree](https://github.com/jackLei0901/vllm-runtime-reliability-lab/tree/v0.2.0)
- [v0.2 collect/verify contract](https://github.com/jackLei0901/vllm-runtime-reliability-lab/blob/v0.2.0/docs/COLLECT_VERIFY_V0_2.md)
- [Field-role audit](https://github.com/jackLei0901/vllm-runtime-reliability-lab/blob/v0.2.0/docs/V0.2_FIELD_ROLES.md)
- [Published #53859 replay evidence](https://github.com/jackLei0901/vllm-runtime-reliability-lab/tree/v0.2.0/results/vllm-zmq-backpressure-stage1-r3-20260916)

Release artifact SHA-256 values and clean-install transcripts are published
alongside the release assets built from the exact tagged commit.
