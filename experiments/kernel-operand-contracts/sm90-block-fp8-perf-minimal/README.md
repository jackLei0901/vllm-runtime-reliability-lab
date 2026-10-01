# Minimal SM90 dispatch performance successor

Status: CPU implementation ready for review; no GPU result, no public freeze
confirmation, no booking authorization. The Chinese minimal protocol is
authoritative. This directory is self-contained; the original
`sm90-block-fp8-perf` directory and its manifest are untouched.

Only `dispatch_variant.patch` changes kernel code. Shapes, input generation,
176 correctness cases, numerical threshold, A-A/ABBA mechanics and timing
constants are inherited. Successor changes are orchestration/aggregation,
numerical-versus-apparatus classification, and qualitative V2 replay evidence.

## CPU preparation

From the repository root:

```text
python -m unittest discover -s tests -p test_sm90_perf_minimal.py -v
python experiments/kernel-operand-contracts/sm90-block-fp8-perf-minimal/timing.py --session S --plan
python experiments/kernel-operand-contracts/sm90-block-fp8-perf-minimal/correctness.py --plan
python experiments/kernel-operand-contracts/sm90-block-fp8-perf/freeze_packet.py --check
python experiments/kernel-operand-contracts/sm90-block-fp8-perf-minimal/freeze_packet.py --check
```

The last check needs the successor manifest to have been generated first.
Its bytes are not a public preregistration until committed and pushed.
Every actual run requires a full public commit. The first five-minute preflight
checks raw GitHub once and saves a private receipt. Later tools require
`SM90_PUBLIC_FREEZE_RECEIPT` and verify locally, with no network fallback.
Do not modify this packet after publication without a new declared revision.

## Implementation limits

- The engineering decision checks the exact 80 calibration and 80 pair keys,
  then controls, regressions and the by-M 7/9 with 3/4-shapes rule.
- A calibrated label is a product of the existing clock/noise scorer, not an
  independent statistical-confidence assertion.
- E1 reports a numerical-only variant failure as `not_worth_proposing`.
  Runtime errors, missing cases and base failures remain `unscored`.
- E3's idle check is a Linux/NVIDIA point-in-time snapshot. It does not prevent
  external activity starting later; the existing clock rules still apply.
  It runs before CUDA initialization and in the five-minute identity preflight;
  PID inventory must be empty and GPU memory at most 128 MiB.
- `build_pair.py` builds sequentially under one 35-minute UTC deadline. The
  second arm is limited to the time remaining, never a fresh 20-minute window.
  E4 is skipped when the conservative memory-headroom gate fails.
- E3 timeout or an excessive first-three-cell projection is
  `insufficient_evidence`, never a negative performance finding.
- E2 and E1 each require `apparatus.py --idle-only` after E4 shutdown.
  E4 export/review is deferred until after E3, preferably offline after power-off.
- E4 uses CPU launch-API correlation, encoded process identity, decode NVTX
  annotations and UTC-normalized exported timestamps. Missing/ambiguous
  metadata is `unwitnessed`, not a reason to weaken E3.
- E4 checks the first ten qualifying decode graph launches per level, not
  ten selected successful steps. It neither measures exact M nor time shares.
- Batch windows retain both UTC and monotonic timestamps; a drift greater
  than 50 ms invalidates the window. This is a clock-integrity check, not a
  changed performance threshold.

For UTC export semantics, see NVIDIA's [Nsight Systems user guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
(`nsys export --ts-normalize=true`). The historical SQLite is only schema
development input. No historical trace is re-scored by this successor.

See [the runbook](RUNBOOK.zh-CN.md) for the capped session and pending checks.
