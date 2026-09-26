# #36451 health ping against the #53859 backpressure mechanism

Status: preregistration for the **exact PR-head gate, which has not run**. The first instance
failed the isolated-environment preflight; see
[environment record](PR36451_ENV_PREFLIGHT_2026-09-25.md). Source revision:
`sihyeonn/vllm@206f28dc3b33abc4d0425fce8d8e14b1ea9cd7e0`.
The separately scoped [current-wheel Python change-set result](PR36451_SOURCE_EQUIVALENT_RESULT_2026-09-25.md)
does not satisfy this exact-head gate.
Question: does the opt-in IPC round-trip turn the #53859
health-green/no-progress stall into an application-level `/health` failure
without failing normal long steps? The result may be GO, NO-GO or
undetermined. Upstream exit is a short evidence comment on
[PR #36451](https://github.com/vllm-project/vllm/pull/36451) if the
experiment is valid; no new health proposal is implied.

## Fixed contrast

One exact PR-head source/build, DP=1, one EngineCore, same model, request,
injection plugin and hardware. Restart the server for each cell. The only
feature-arm change is `VLLM_HEALTH_CHECK_TIMEOUT=0` (off) versus `60` (on).
The trigger-arm change is whether the existing Stage 1 plugin immediately
releases the event consumer (control) or holds it until after the health
observation (pause). This is a feature-toggle comparison, not a base/head
code-diff validation. Do not compare old #53859 wall times with this run.

| Cell | Ping | Consumer | Pre-registered expectation |
| --- | --- | --- | --- |
| A | off | control | Request progresses; `/health` 2xx |
| B | on | control | Request progresses; `/health` 2xx; record ping latency |
| C | off | pause | Admitted request stalls; EngineCore stays alive; `/health` 2xx |
| D | on | pause | Same stall and liveness; `/health` returns application 503 after the 60 s ping deadline |

Run A, B, C, D once each before any repeat. If a cell fails, retain the
failed record; do not silently replace it. Repeat only the same frozen four
cells, up to three campaigns if the first is valid and hardware time allows.

## Timing and bounded release

Use a direct loopback request to `/health` with a client deadline of at
least 75 seconds and record status, body/error class and monotonic elapsed
time. Confirm a 2xx baseline before starting the generation request.
Start the health call only after the generation has admitted work and the
pause arm has independently reached the queue-blocked condition. Retain
the pause until that health call completes or its client deadline fires;
then release the consumer unconditionally. Impose an outer 90-second
post-stall cap and a 180-second generation-request cap. In the off arm,
observe `/health` during the same bounded stalled window; do not hold it
for an artificial full 60 seconds if the response is immediate. Preserve
the control's normal release-before-request ordering.

The 60-second server threshold is not a 30-second detector. A proxy/client
timeout, disconnection, 500, or absent HTTP response is **not** evidence of
application 503. Record actual times; the expected status is not a license
to coerce an ambiguous response into success. Verify the real server's
request timeout and any proxy before the run.

## Evidence required for an interpretable cell

- Exact source SHA, built wheel and loaded native-binary hashes, Python,
  torch/CUDA/NCCL, GPU/model identity, flags, plugin SHA and source SHA.
- One EngineCore ready record, PID plus `/proc` start ticks stable before
  and after the observation; no hook error or duplicate EngineCore.
- Admitted work plus client-observed token progress history, queue accepted
  counts, bounded external stack if attach is available, and a
  monotonic timeline joining stall, health-call start/end and release.
- Separate health producer outcome (HTTP status or typed failure). A
  missing metric or stack producer is explicit, not silently treated as
  negative evidence. Raw stack and request content stay private; public
  output is bounded flags, hashes and timestamps.
- Recovery after release where feasible. Recovery is a useful control,
  not a substitute for the stalled-window evidence.

Fail closed to `undetermined` if the plugin is incompatible, the pause never
creates the expected blocking mechanism, demand is absent, process identity
changes, the health request is intercepted by an earlier timeout, or the
injection is not released within the cap. A 503 without a confirmed #53859
stall does not validate detection of that mechanism.

## Normal-operation controls and stopping rule

On the same pinned build with ping enabled, measure `/health` during idle,
long prefill and first-request compilation; graph capture only if enabled
by the actual launch. Pre-register workload sizes from the model's context
limit before GPU execution. Record request completion, health result and
elapsed time, including any 503. A finite control set reports observed
false positives, **not** a production false-positive rate. Any 503 in a
healthy control is a safety finding, not a case to discard as warmup.

If exact head cannot be built with the available disk/runtime, or the
existing plugin no longer hooks the intended publisher, stop at a bounded
NO-GO. Do not patch production semantics to make the experiment pass.
The prior Stage 1 R3 result and runner remain immutable.
