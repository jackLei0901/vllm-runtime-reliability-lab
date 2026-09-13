# vLLM Runtime Reliability Lab

[中文文档](README.zh-CN.md) · [中文方案设计](DESIGN.zh-CN.md) ·
[中文产品化计划](PRODUCT_ROADMAP.zh-CN.md)

[![CI](https://github.com/jackLei0901/vllm-runtime-reliability-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/jackLei0901/vllm-runtime-reliability-lab/actions/workflows/ci.yml)

An opt-in, out-of-process reliability evidence and validation lab for vLLM. The
current alpha freezes a bounded, privacy-safe runtime history when an externally
observable condition is met. The next product step is to correlate independent
process/rank evidence and detect health-green loss of progress without relying
on a failure-time collective.

This is an **alpha research tool**, not a production monitor and not a root-cause
classifier. An external observer can preserve chronology, but it cannot infer an
internal EngineCore exception kind or execution stage. The artifact records both
as `unknown` by construction.

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
| `/health=200` while admitted work stops progressing | bounded `suspected_no_progress` transition with its supporting observations | investigate, drain or restart instead of leaving a silent outage healthy |
| one rank stalls or disappears while peers expose only local state | verified producer set, missing-peer/state divergence and ordering limits | identify the first useful fault domain without a failure-time collective |
| several files exist but cannot be trusted as one incident | closed manifest, identities, clocks and content hashes | reject mixed or tampered evidence before diagnosis |

The current alpha supplies the bounded local evidence primitive. The table's
no-progress and multi-producer outputs are planned v0.2 gates, not shipped
features.

### Planned stack adapter: a hard deployment gate

The proposed CPU-stack adapter is also not shipped. `py-spy` reads another
process's memory: attaching on Linux usually needs root or an adjusted
`ptrace_scope`; Docker and Kubernetes commonly require `SYS_PTRACE`. The default
sampling path may pause the target briefly. `--nonblocking` avoids that pause but
can return sampling errors or partial frames because the reads are not atomic.
See the [py-spy deployment and nonblocking
notes](https://github.com/benfred/py-spy#frequently-asked-questions).

Before any joiner is built, a go/no-go experiment must establish that useful
Python/native context can be captured from a blocked process within a fixed
duration and sample-count budget. Single-process and single-GPU tests can cover
attachment and CUDA/native waits; a real rank blocked in an unmatched NCCL
collective requires a multi-rank GPU test. Capture denial, timeout or partial
output is a normal explicit result, not a recorder failure.

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

## What the alpha provides

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
- Markdown summaries and explicit signal-injection helpers.

The alpha does **not** yet provide a no-progress detector, process/rank discovery,
a cross-producer join or a correlation manifest. Those are v0.2 targets and must
not be inferred from the current feature list.

## Five-minute CPU-only demo

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

The healthy demo does not manufacture an incident. To exercise the health-loss
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

The remaining GPU validation plan is in [`TEST_PLAN.md`](TEST_PLAN.md).

## Documentation

- [`README.zh-CN.md`](README.zh-CN.md): detailed Chinese usage and readiness guide.
- [`DESIGN.zh-CN.md`](DESIGN.zh-CN.md): Chinese architecture and design review.
- [`PRODUCT_ROADMAP.zh-CN.md`](PRODUCT_ROADMAP.zh-CN.md): productization stages and gates.
- [`REQUIREMENTS.md`](REQUIREMENTS.md): scope and acceptance criteria.
- [`DESIGN.md`](DESIGN.md): trust boundary and component design.
- [`PRIOR_ART_AND_VALUE.md`](PRIOR_ART_AND_VALUE.md): linked-evidence precedent,
  demonstrated value and transfer limits.
- [`TEST_PLAN.md`](TEST_PLAN.md): CPU and GPU validation matrix.
- [`SECURITY.md`](SECURITY.md): privacy assumptions and reporting guidance.
- [`CHANGELOG.md`](CHANGELOG.md): release history.

## Status

`v0.1.0-alpha.4` adds an auditable four-GPU known-answer reconstruction for an
organic PyTorch FSDP2 hang. It does not establish unknown-root-cause discovery
or a campaign-level GO decision. Paired overhead, fresh KV-pressure,
health-green no-progress, cross-host correlation, long-duration, and
production-utility gates remain open. Treat this release as an evaluation
build.

## License

MIT
