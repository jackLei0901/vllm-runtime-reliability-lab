# vLLM Runtime Reliability Lab

[中文文档](README.zh-CN.md) · [中文方案设计](DESIGN.zh-CN.md) ·
[中文 evidence-first 计划](PRODUCT_ROADMAP.zh-CN.md)

[![CI](https://github.com/jackLei0901/vllm-runtime-reliability-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/jackLei0901/vllm-runtime-reliability-lab/actions/workflows/ci.yml)

An evidence lab for inference failures that leave the service process alive but
stop useful work across processes or ranks.

The lab turns “it hung” into a bounded chain of claims: real failure, minimal
mechanism, preregistered prediction, base/fix comparison, external process and
stack evidence, privacy-bounded per-rank flags, a fail-closed verifier, and an
upstream issue, PR, or review result.

This is an **alpha research tool**, not a monitoring platform, an issue
collection, or an automatic root-cause classifier.

## What this lab caught

| Failure seen from outside | What the lab established | External result and current boundary |
| --- | --- | --- |
| Flight Recorder produced a rank-0 dump but no rank-1 dump | `producer missing != member missing`: rank 1 was alive, had entered `destroy_process_group()`, and could no longer answer the dump request | Lab-originated [PyTorch #196968](https://github.com/pytorch/pytorch/issues/196968) and proposed C++ fix [#197232](https://github.com/pytorch/pytorch/pull/197232). Both remain open as of 2026-09-21. |
| vLLM stayed alive and `/health` returned 2xx while token progress stopped | A deterministic full event queue blocked the real EngineCore in `ZmqEventPublisher.publish()`; the fix preserved progress by dropping event batches | Independent validation of reported [vLLM #53859](https://github.com/vllm-project/vllm/issues/53859) and proposed fix [#53883](https://github.com/vllm-project/vllm/pull/53883), not a lab-originated bug. Both remain open as of 2026-09-21. |
| A torchtitan distributed hang appeared to be a collective mismatch | Successive gates removed the distributed surface and reproduced an FSDP2 mixed-gradient-dtype assertion on one GPU | Lab-originated [PyTorch #196996](https://github.com/pytorch/pytorch/issues/196996), triaged and open as of 2026-09-21. |

The evidence changed the conclusion in each case:

| Case | Tempting conclusion | Evidence-backed conclusion |
| --- | --- | --- |
| #196968 | Missing dump means the rank did not participate | The diagnostic producer disappeared while the participating process remained alive in teardown |
| #53859 / #53883 | HTTP health means the server is healthy | Health stayed green while inference made no progress; the fix restored liveness with a measured event-loss trade-off |
| #196996 | A multi-rank hang requires a distributed root cause | The relevant correctness failure reduces to a local mixed-dtype contract violation |

`#49869` is an independent upstream contribution and is intentionally not
presented as a lab discovery. `#52178` is a separate lifecycle fix for which the
lab supplied process-level validation.

The cross-case synthesis is available in
[Failures that never reach the supervisor](docs/FAILURES_THAT_NEVER_REACH_THE_SUPERVISOR.md).

## Five-minute replay — no GPU required

The replay does not rerun a GPU experiment. It verifies the closed file set,
SHA-256 identities, four-cell base/fix contract, external progress and health
observations, blocking-stack claim, and measured trade-off from the published
#53859 campaign.

```bash
python -m pip install .  # installs distribution vllm-runtime-dfx-lab
vllm-dfx replay results/vllm-zmq-backpressure-stage1-r3-20260916
```

Expected conclusion:

```text
PASS: evidence file set and SHA-256 identities verified
PASS: controls completed without a stall or dropped event batch
PASS: EngineCore process remained alive during the injected stall
PASS: /health remained 2xx while token progress stopped
STACK: EngineCore -> ZmqEventPublisher.publish -> Queue.put
FIX ARM: token progress completed under the same trigger
TRADE-OFF: 4 event batches dropped
BOUNDARY: replay verifies archived evidence; it does not rerun the GPU experiment
```

The verifier fails closed if an evidence hash, record shape, cell identity, or
cross-cell identity changes. If you run it, please report the command, platform,
and result with the [replay report template](https://github.com/jackLei0901/vllm-runtime-reliability-lab/issues/new?template=replay-report.yml).
For a copy-paste clean-environment check, use the
[five-minute external trial](docs/V0.2_EXTERNAL_TRIAL.md).

To apply the same evidence rules to a local incident without a GPU dependency:

```bash
vllm-dfx collect \
  --base-url http://127.0.0.1:8000 \
  --pid 12345 \
  --window 60 \
  --no-progress-window 10 \
  --output incident/
vllm-dfx verify incident/
```

`verify` is offline and recomputes producer, demand, conflict, and verdict
claims from `observations.json`. The public bundle does not retain the base URL,
prompt, response content, headers, raw stack, or complete sampler command. See
the [v0.2 collect/verify contract](docs/COLLECT_VERIFY_V0_2.md) for endpoint-only,
PID-only, client-probe, stack, and trust-boundary details.

## Evidence rules

```text
alive              != making progress
health green       != serving healthy
producer missing   != participant missing
fix restores life  != fix preserves every diagnostic event
```

Architecture, recorder internals, and schemas follow the demonstrated results
because they are means to those claims, not the project headline.

## The operational problem

Two failure modes expose gaps in ordinary serving health checks:

- an EngineCore can die while the serving process exits successfully, defeating
  `Restart=on-failure` ([vLLM #48966](https://github.com/vllm-project/vllm/issues/48966));
- requests can stop producing output while the process and `/health` remain
  alive ([vLLM #52319](https://github.com/vllm-project/vllm/issues/52319)).

A final traceback is useful for a point failure, but it does not preserve the
trajectory before the failure. A single-rank record also cannot define a
missing peer or a cross-rank state mismatch at a shared logical position.
Those are relationship facts created only by joining independently produced
evidence.

This project aims to make those failure and evidence contracts executable:
detect an externally visible loss, preserve a bounded local window, state what
is still unknown, and test whether linked evidence changes the next diagnostic
action. It does not promise automatic root-cause analysis.

The product test is not “did it collect data?” It is whether the evidence closes
an operational gap:

| Operational gap | Product output | Decision it should support |
| --- | --- | --- |
| `/health=200` while admitted work stops progressing | bounded `alive_health_ok_no_progress` verdict with its supporting observations | investigate, drain or restart instead of leaving a silent outage healthy |
| one rank stalls or disappears while peers expose only local state | verified producer set, missing-peer/state divergence and ordering limits | identify the first useful fault domain without a failure-time collective |
| several files exist but cannot be trusted as one incident | closed manifest, identities, clocks and content hashes | reject mixed or tampered evidence before diagnosis |

v0.2 ships the bounded local `collect` and offline `verify` path. It selects one
decision producer and can retain one corroborating producer without merging
their scopes. It does not yet perform a general cross-rank or cross-host join.

### Optional stack producer: a hard deployment gate

v0.2 can invoke a bounded, opt-in `py-spy` stack producer. Raw stack output is
private and cannot change the verdict; the public bundle retains only typed
availability and bounded producer identity. `py-spy` reads another process's
memory: attaching on Linux usually needs root or an adjusted
`ptrace_scope`; Docker and Kubernetes commonly require `SYS_PTRACE`. The default
sampling path may pause the target briefly. `--nonblocking` avoids that pause but
can return sampling errors or partial frames because the reads are not atomic.
See the [py-spy deployment and nonblocking
notes](https://github.com/benfred/py-spy#frequently-asked-questions).

The shipped adapter does not publish arbitrary frames, classify native state,
or join stacks across ranks. Those capabilities require a separate go/no-go
experiment under a fixed duration and sample-count budget. Capture denial,
timeout, or partial output is a normal explicit result, not a recorder failure.

### Why continuous metrics are not enough for these incidents

vLLM already exports Prometheus metrics (`vllm/v1/metrics/prometheus.py`) and
OpenTelemetry traces (`vllm/tracing/otel.py`). This project replaces neither,
and it consumes the same `/metrics` endpoint as one of its inputs. It exists
because default continuous metrics alone do not reliably preserve three kinds
of evidence needed after a fatal or stalled incident.

**The final interval may be missing.** Prometheus pulls on an interval. A
process that dies stops answering, so activity between the last successful
scrape and the failure may never reach the time-series store. This recorder
keeps a bounded local history and freezes it when an external trigger fires.
It does not eliminate sampling limits, but it preserves the samples already
held at the observation boundary when the remote endpoint disappears.

**The diagnostic datum is high-cardinality.** Establishing where ranks diverged
may require per-rank, per-collective records: sequence number, input shapes,
dtypes and stack frames. Encoding those records as continuously exported metric
labels creates an operational cardinality cost. A one-shot artifact avoids that
continuous cost, while remaining bounded by this project's 256 KiB file cap.

**The answer is a relation, not a value.** "Rank 1 raised locally while rank 0
waited in the collective" is a statement about two independently produced
records joined at a shared logical position. Per-target metric values do not by
themselves encode that relation, and wall-clock timestamps are not a safe total
order when sampling cadence and clock uncertainty are comparable to the event.
The correlation design therefore prefers logical collective position over
timestamp ordering.

A boundary found in this repository makes the gap concrete. In the Phase 2
Gate 1 run, a two-rank job reached a sixty-second wall timeout, but the original
runner discarded the per-rank output needed to tell whether one rank asserted
locally before its peer stalled or both ranks stalled. The original Gate 1b
contract was withdrawn before execution after source review showed that it put
the expected CPU wait at the wrong call site. Gate 1c was also withdrawn before
execution because its wall bound was too tight and its stop rule coupled the
mechanism and termination gates. Gate 1d then stopped on a runner parsing defect.
Gate 1e fixed that defect and reproduced the rank-local assertion plus peer wait
three times. Both rank stacks were captured, but only rank 0 produced a Flight
Recorder dump in every trial, so the strict correlation gate failed closed.
Gate 1f then ran one frozen affected trial with eight allow-listed per-rank
PyTorch shutdown-stage flags. Rank 0 successfully broadcast the dump request
and wrote its dump; rank 1 had stopped its heartbeat monitor and entered
communicator destruction, never completed destruction and never observed the
request. The diagnostic gate passed, while Gate 1e remains failed closed.

The inverse case matters as much. In a reported health-green stall
([vLLM #52319](https://github.com/vllm-project/vllm/issues/52319)), `/health` and
`/metrics` continued to return HTTP 200 while generation throughput fell to
zero and waiting work accumulated. Metrics can detect the loss of progress;
they do not, by themselves, identify the internal rank or execution point where
progress stopped.

If a deployment already retains equivalent per-rank, trigger-time data and can
package it reliably at failure time, this recorder adds nothing. The planned
Prometheus baseline and the unlinked-versus-linked ablation are explicit
product gates, not assumptions.

The design was informed by the investigation behind
[vLLM #48966](https://github.com/vllm-project/vllm/issues/48966),
[PR #52178](https://github.com/vllm-project/vllm/pull/52178), and the
[runtime incident snapshot RFC](https://github.com/vllm-project/vllm/issues/54229).
It remains independent of vLLM and does not modify EngineCore.

The closest published precedent is PyTorch Flight Recorder: per-rank buffers are
aligned offline to expose collective mismatches that no single rank can define.
Its documented before/after, limitations and transfer boundary are summarized in
[`PRIOR_ART_AND_VALUE.md`](PRIOR_ART_AND_VALUE.md).

## What v0.2 provides

- `/health` and selected `/metrics` polling;
- explicit PID liveness and Linux RSS/VMS/thread sampling;
- aggregate `nvidia-smi` sampling on a slower, independent cadence;
- a fixed-size in-memory history;
- process-exit, health-loss, KV-pressure and preemption triggers;
- a closed JSON contract with `additionalProperties: false`;
- allow-listed aggregate fields only: no prompt, token IDs, request IDs or paths;
- per-process HMAC incident IDs that do not correlate across restarts;
- 256 KiB artifact cap, four-file rotation and POSIX mode `0600`;
- fail-open writer behavior: artifact failure does not signal the observed service;
- Markdown summaries and explicit signal-injection helpers;
- bounded `collect` bundles with stable PID identity and always-closed windows;
- server-counter and opt-in client-request progress producers;
- demand-gated, fail-closed `verify` verdict recomputation;
- an optional private `py-spy` capture with public typed producer status; and
- the no-GPU `vllm-dfx replay` for the published #53859 four-cell campaign.

v0.2 does **not** provide continuous autonomous monitoring, process/rank
discovery, a general cross-rank or cross-host join, native-state classification,
automatic remediation, or automatic root-cause analysis.

## Live CPU-only recorder demo

Python 3.10 or newer is required. The runtime package has no third-party
dependencies.

```bash
python -m pip install -e .
python examples/fake_vllm_server.py --port 18000
```

In a second terminal:

```bash
vllm-dfx record \
  --base-url http://127.0.0.1:18000 \
  --output demo-output \
  --history 16 \
  --duration 5
```

This live recorder demo is separate from the published-result replay above. The
healthy demo does not manufacture an incident. To exercise the health-loss
path, stop the fake server while the recorder is running:

```bash
vllm-dfx record \
  --base-url http://127.0.0.1:18000 \
  --output demo-output \
  --history 16 \
  --unhealthy-samples 2 \
  --stop-on-incident
```

Then summarize the generated artifact:

```bash
vllm-dfx summarize \
  --input demo-output/incident-*.json \
  --output demo-output/incident-summary.md
```

The shell wildcard in the last command is intended for POSIX shells. On
PowerShell, pass the concrete artifact filename.

## Real vLLM usage

Wait until the server is ready, then provide the API-server PID explicitly:

```bash
vllm-dfx snapshot-env --output private-run/environment.private.json

vllm-dfx record \
  --base-url http://127.0.0.1:8000 \
  --pid "$API_SERVER_PID" \
  --target-vllm-version "<exact server version or commit>" \
  --target-torch-version "<exact server Torch version>" \
  --output shareable-incidents \
  --sample-interval 1 \
  --gpu-interval 5 \
  --history 300
```

`snapshot-env` and `run-summary.private.json` are private lab metadata. Review
them before sharing. Files named `incident-*.json` follow the stricter external
artifact contract.

The recorder may run in a different Python environment from vLLM. It therefore
never infers the target's vLLM or Torch version from its own installed packages.
Those artifact fields remain `null` unless the two explicit target-version
arguments are provided.

Raw timeline persistence is disabled by default. The explicit
`--private-raw-timeline` option writes an unbounded private debugging file and
should not be enabled for unattended production use.

## Fault injection

The tool never discovers a process through fuzzy command matching. Inspect the
process tree and pass the intended PID:

```bash
vllm-dfx inject-signal \
  --pid "$ENGINE_CORE_PID" \
  --signal SIGKILL \
  --event-log private-run/injections.jsonl \
  --dry-run
```

Remove `--dry-run` only in an isolated environment where process termination is
expected. The injection event log is private lab evidence and is not part of the
shareable incident schema.

## Artifact contract

The machine-checkable contract is
[`schema/external-runtime-observation-v1.schema.json`](schema/external-runtime-observation-v1.schema.json).
A synthetic example is available at
[`examples/external-runtime-observation-v1.json`](examples/external-runtime-observation-v1.json).

The shareable artifact contains:

```text
typed external trigger
runtime version and GPU-model allow-list
bounded health/process/metrics/GPU history
recorder overwrite/drop counters
writer success/failure counters
```

It intentionally omits prompt text, token IDs, request identifiers, model paths,
environment variables, command lines, tracebacks and arbitrary configuration.
The allow-list is the primary boundary; the legacy deny-list is only a secondary
guard for private lab metadata.

## Validation

Install the development dependencies, then run the CPU suite:

```bash
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
python -m compileall -q src tests
vllm-dfx --help
```

The repository includes reviewed summaries from RTX 4090 experiments:

- startup KV-capacity boundary;
- paired KV-pressure/preemption behavior;
- a 1,200-request, ten-minute smoke soak;
- intentional SIGTERM, EngineCore SIGKILL and controlled `execute_model` OOM;
- separate TP=2 process-level validation associated with PR #52178.

These results establish test-harness behavior only for the pinned environments.
They do not establish long-term stability, DP/NCCL behavior or production value.
The fresh alpha.2 result is in
[`results/gpu-20260909-alpha2/VALIDATION_SUMMARY.md`](results/gpu-20260909-alpha2/VALIDATION_SUMMARY.md).

The alpha.4 organic-hang campaign moves beyond project-designed faults. On a
four-GPU PyTorch FSDP2 workload with an independently known answer, three
`DebugLevel.DETAIL` trials and three automatic ProcessGroupNCCL Flight Recorder
trials exposed the same `_REDUCE_SCATTER_BASE` input-shape mismatch. This is a
known-answer reconstruction, not root-cause discovery, and the campaign remains
short of GO because a same-version no-divergence control is still required. See
the [review entry](experiments/organic-hang/REVIEW_RESPONSE_2026-09-10.md) and
[derived-only public evidence](results/organic-hang-20260912/README.md).

A separate two-GPU dtype campaign has also completed its mechanism gate: the
ordinary unused-parameter arm produced uniform BF16 gradient lists, while a
forced mixed-gradient control produced BF16+FP32 and triggered the expected
PyTorch assertion. A later accumulated-gradient trial reached the wall timeout,
but its old runner did not retain enough per-rank evidence to classify the
sequence. Gates 1b and 1c were withdrawn before execution, and Gate 1d stopped on
a runner parsing defect. Gate 1e then matched the frozen mechanism and termination
predictions in all three affected trials, with both rank stacks and clean process
lifecycle. It failed the strict capture gate because only rank 0 produced a Flight
Recorder dump in every trial. See the
[review entry](experiments/pytorch-unused-grad-dtype/REVIEW_PHASE2_RESULTS_CN.md)
and [result](experiments/pytorch-unused-grad-dtype/GATE1E_RESULT_2026-09-13.md).
The Gate 1f shutdown-stage diagnostic is documented in its
[`protocol`](experiments/pytorch-unused-grad-dtype/GATE1F_PROTOCOL.md) and
[`result`](experiments/pytorch-unused-grad-dtype/GATE1F_RESULT_2026-09-13.md).
The FSDP-free, two-rank Gate 1g boundary experiment then reproduced the same
missing-rank dump behavior after model, gradient and FSDP variables were
removed. The result is limited to PyTorch 2.13.0+cu130 with NCCL 2.29.7 and does
not identify the exact NCCL blocking call or validate a fix. See the frozen
[`protocol`](experiments/pytorch-unused-grad-dtype/GATE1G_PROTOCOL.md) and
[`result`](experiments/pytorch-unused-grad-dtype/GATE1G_RESULT_2026-09-13.md).

The campaign also isolated a separate FSDP2 gradient-accumulation dtype defect
and reproduced both of its triggers on a current PyTorch nightly. It was filed
as [pytorch/pytorch#196996](https://github.com/pytorch/pytorch/issues/196996),
is now triaged, and has been taken up for a maintainer-owned fix. The lab will
track the upstream change rather than open a competing PR; see the
[validation result](experiments/pytorch-unused-grad-dtype/upstream_dtype_issue/VALIDATION_RESULT_2026-09-14.md)
and [upstream status](experiments/pytorch-unused-grad-dtype/upstream_dtype_issue/UPSTREAM_STATUS_2026-09-14.md).

The vLLM #53859 Stage 1 campaign adds a different kind of result: a real
single-GPU EngineCore remained health-responsive while KV-event publisher
backpressure stopped token progress. An external stack located the blocking
queue path, and releasing the consumer restored the request. With #53883
applied, the same request completed without a stall while the EngineCore-local
counter measured one accepted event batch and four dropped batches. This is
evidence for **liveness restored with event loss**, not reliable delivery or a
production loss-rate estimate. See the
[reviewed result](experiments/vllm-zmq-event-backpressure/STAGE1_R3_RESULT_2026-09-16.md)
and [public summaries](results/vllm-zmq-backpressure-stage1-r3-20260916/).

The remaining GPU validation plan is in [`TEST_PLAN.md`](TEST_PLAN.md).

## Documentation

- [`README.zh-CN.md`](README.zh-CN.md): detailed Chinese usage and readiness guide.
- [`DESIGN.zh-CN.md`](DESIGN.zh-CN.md): Chinese architecture and design review.
- [`PRODUCT_ROADMAP.zh-CN.md`](PRODUCT_ROADMAP.zh-CN.md): evidence-first stages and gates.
- [`REQUIREMENTS.md`](REQUIREMENTS.md): scope and acceptance criteria.
- [`DESIGN.md`](DESIGN.md): trust boundary and component design.
- [`PRIOR_ART_AND_VALUE.md`](PRIOR_ART_AND_VALUE.md): linked-evidence precedent,
  demonstrated value and transfer limits.
- [`TEST_PLAN.md`](TEST_PLAN.md): CPU and GPU validation matrix.
- [`docs/V0.2_RELEASE_PLAN.md`](docs/V0.2_RELEASE_PLAN.md): bounded v0.2 payload
  and release gates.
- [`docs/V0.2_LAUNCH_POST.md`](docs/V0.2_LAUNCH_POST.md) and
  [`docs/V0.2_LAUNCH_POST.zh-CN.md`](docs/V0.2_LAUNCH_POST.zh-CN.md): restrained
  English and Chinese launch copy with the exact clean-install replay command.
- [`docs/FAILURES_THAT_NEVER_REACH_THE_SUPERVISOR.md`](docs/FAILURES_THAT_NEVER_REACH_THE_SUPERVISOR.md)
  and its [Chinese version](docs/FAILURES_THAT_NEVER_REACH_THE_SUPERVISOR.zh-CN.md):
  a synthesis of four failure signals that lost meaning across process, rank,
  health, or diagnostic boundaries.
- [`docs/V0.2_FIELD_ROLES.md`](docs/V0.2_FIELD_ROLES.md): machine-checked
  decisional and non-decisional public-field audit.
- [`docs/ADOPTION_PLAN.md`](docs/ADOPTION_PLAN.md): focused external-reuse plan
  and scorecard.
- [`docs/case-studies/flight-recorder-missing-rank-DRAFT.md`](docs/case-studies/flight-recorder-missing-rank-DRAFT.md):
  unpublished #196968 case study, gated on an explicit #197232 outcome.
- [`SECURITY.md`](SECURITY.md): privacy assumptions and reporting guidance.
- [`CHANGELOG.md`](CHANGELOG.md): release history.

## Status

The published [`v0.2.0` release](https://github.com/jackLei0901/vllm-runtime-reliability-lab/releases/tag/v0.2.0)
ships bounded no-progress
collection, offline verification, and the no-GPU published-result replay; it is
not a production monitor or a general cross-process joiner. Paired overhead,
fresh KV-pressure, cross-host correlation, long-duration, and production-utility
gates remain open. The #196968 case study remains unpublished while #197232 is
open; an open PR is not an upstream outcome.

## License

MIT
