# External Runtime Incident Recorder — MVP plan

> Status on 2026-09-09: M0 and M1 are complete in `v0.1.0-alpha.3`, and M2
> has been reconciled against the published evidence. M3 is partial: the fresh
> single-GPU fatal, writer-failure, and disabled controls are complete; fresh
> KV-pressure/preemption and multi-GPU observer coverage remain open. The CPU
> paired-overhead harness exists, but GPU overhead and adoption evidence remain
> open. The no-progress and correlation work below is a post-alpha direction,
> not an alpha feature.

## Outcome

Build a standalone, opt-in recorder that produces real, bounded external
incident artifacts without modifying vLLM. Its purpose is to validate capture,
privacy, persistence, overhead and adoption workflow, then grow into an external
reliability evidence plane for health-green loss of progress and cross-process
correlation.

The external observer is not a disposable precursor to an in-process recorder.
It remains useful when an internal thread is stalled or a process has died.
EngineCore, PyTorch/NCCL Flight Recorder and supervisor evidence are independent
producers that may later be linked offline through a closed manifest.

The external preview does **not** claim to know an internal EngineCore fatal kind
or stage. The September 1 boundary experiment showed that an outside observer
can see health/process loss but cannot reliably distinguish intentional API
shutdown, EngineCore loss and model-execution CUDA OOM. Those unknowns remain
explicit instead of being inferred from timing or exception text.

## Existing implementation inventory

The current `dfxlab` implementation already provides:

- `/health` and selected `/metrics` polling;
- explicit PID liveness plus Linux RSS/VMS/thread sampling;
- `nvidia-smi` GPU sampling;
- a bounded in-memory `deque`;
- health-loss, process-exit, KV-pressure and preemption triggers;
- atomic temporary-file replacement;
- environment capture, Markdown summaries and signal-injection helpers;
- unit coverage for basic collection, metric parsing, bounded history and
  trigger classification;
- single-GPU fault-boundary evidence under `results/`;
- separate TP=2 process-lifecycle evidence associated with vLLM PR #52178,
  which is not current multi-producer observer validation.

## Decisions implemented in the alpha preview

| Gap in current lab | MVP decision |
| --- | --- |
| `Incident` accepts arbitrary dictionaries | Add a strict external-artifact schema and explicit projection; reject unknown fields |
| Redaction is a deny-list applied after collection | Collect only allow-listed aggregate fields; keep the deny-list as a secondary canary guard |
| Bare SHA-256 identifiers correlate across restarts | Use an ephemeral per-process HMAC key, or omit the identifier when it is not needed |
| `timeline.jsonl` grows without a bound | Disable raw timeline output by default; keep it only behind an explicit private-lab flag |
| Environment includes executable/path-adjacent data | Split private lab metadata from the shareable artifact; never emit paths in the shareable form |
| Trigger is a pair of free-form strings | Define `ExternalTriggerContext` with a closed enum and an explicit `internal_kind: unknown` boundary |
| No overwrite/drop accounting | Track appended, overwritten and dropped totals plus retained sequence range |
| No artifact size cap or rotation | Enforce 256 KiB, rotate feature-owned files to a maximum of four during startup |
| File mode is inherited from the environment | Create temporary and final files as `0600` on POSIX |
| Writer error can terminate `run()` | Record one bounded warning and preserve recorder/process observation; writer failure must not alter the observed service |
| `nvidia-smi` runs every sample | Give GPU polling an independent slower cadence and measure its cost separately |
| External causes are ambiguous | Never map `process_exit` or `health_lost` to CUDA OOM, worker loss or intentional shutdown without a typed external source |

## Artifact boundary

Use a separate contract such as `external-runtime-observation-v1`, not the RFC's
`incident-snapshot-v1`. The latter claims an EngineCore-owned typed trigger and
per-completed-iteration history that an external poller cannot provide.

```text
ExternalObservation
  sequence
  monotonic_ns
  health_status
  process_alive
  process_rss_bytes?
  selected_vllm_metrics
  aggregate_gpu_memory?

ExternalTriggerContext
  kind: process_exit | health_lost | kv_pressure | preemption_storm
  observed_at_monotonic_ns
  internal_kind: unknown
  internal_stage: unknown

ExternalIncidentArtifact
  schema_version
  component_type: external_observer
  created_at
  incident_id
  trigger
  runtime_allowlist
  history[]
  recorder_health
  writer_health
```

The component seam is the projected observation, not the file format:

```text
collectors -> ExternalObservation -> BoundedRecorder -> IncidentWriter

future EngineCore/PyTorch/NCCL/supervisor artifacts
                          -> closed manifest -> semantic join
```

A future in-process implementation may reuse bounded-recorder and writer ideas,
but it must emit its own versioned contract. It does not replace the external
observer, and it must not relabel an external artifact as internal evidence.

## MVP implementation sequence

### M0 — contract and writer, CPU only

1. Add closed dataclasses/enums for observation, trigger and recorder health.
2. Add a machine-checkable JSON Schema with `additionalProperties: false`.
3. Replace recursive arbitrary-dict serialization with an explicit projection.
4. Add ephemeral-HMAC support and privacy canary tests.
5. Add size enforcement, initialization rotation, `0600` creation and atomic
   replacement.
6. Make writer failures fail-open with respect to the observed vLLM process.

Exit criterion: all contract, privacy and writer tests pass without vLLM, CUDA
or network access.

### M1 — bounded external collection, CPU/local service

1. Separate health/metrics/process/GPU collector cadences.
2. Add monotonic sequence, appended/overwritten/dropped counters and retained
   range consistency checks.
3. Disable raw timeline persistence by default.
4. Preserve the existing explicit-PID rule; never discover a process by fuzzy
   command matching.
5. Add a tiny fake HTTP service for healthy, 503, timeout and malformed-metrics
   tests.

Exit criterion: deterministic local tests cover healthy operation, health loss,
process exit, metric failure, overwrite and writer failure.

### M2 — reuse existing GPU evidence

Re-evaluate the archived September 1 evidence without claiming internal cause:

- healthy baseline;
- intentional API-server SIGTERM;
- EngineCore SIGKILL;
- one-shot `execute_model` CUDA OOM.

Exit criterion: real external artifacts remain schema-valid, bounded and honest
about `internal_kind=unknown`; the report explains which scenarios remain
indistinguishable externally.

### M3 — new GPU validation

| Scenario | Environment | Required observation |
| --- | --- | --- |
| Healthy + SIGTERM | 1 GPU | exit/health observed; no internal-cause claim |
| EngineCore failure | 1 GPU | bounded artifact; external cause remains unknown |
| Runtime CUDA OOM | 1 GPU | health/process transition preserved; no traceback parsing |
| KV pressure/preemption | 1 GPU | ordered aggregate pressure history and warning trigger |
| Worker loss | TP=2 | explicit producer identity, missing-peer/state transition and ordering boundary |
| Read-only/full directory | CPU first, then GPU smoke | service outcome unaffected; writer failure visible |
| Disabled recorder | 1 GPU A/B | no polling process and no artifact |

Run at least three repetitions for fatal scenarios and check top-level exit,
recorder exit and orphaned processes separately.

### M4 — overhead and adoption evidence

- Interleave recorder-disabled and recorder-enabled runs with identical request
  sets and seeds.
- Report throughput, TTFT, TPOT, recorder CPU/RSS, bytes written and collector
  latency distributions.
- Report GPU polling separately from health/metrics polling.
- Attach an artifact only after validating it and checking the privacy canary.
- Record whether it changed a diagnostic action or avoided a reproduction; do
  not infer value from file creation alone.

### M5 — progress and correlation value

1. Implement CPU fake-service states for idle, healthy progress, long prefill,
   waiting with progress, health-green stall and recovery.
2. Define pre-failure `run_id`, per-process `producer_id`, content-addressed
   `artifact_id`, per-source clocks and a coordinator-assigned `incident_id`.
3. Publish a closed correlation manifest and one bounded vLLM process/progress
   semantic joiner.
4. Add an optional external CPU main-thread stack adapter and correlate its
   per-process snapshots with declared rank identity and existing Flight
   Recorder dump references. Raw stacks remain opt-in private evidence under a
   separate contract; they do not enter the current shareable schema.
5. Compare unlinked and linked presentations of identical producer artifacts.
6. Compare the result with the Prometheus/OpenTelemetry evidence normally
   retained by an operator.
7. Move to TP=2 only after the CPU state-machine, integrity and missing-producer
   cases pass.

Exit criterion: the linked arm creates at least one checkable relationship fact
that the unlinked arm cannot define, while incomplete or incomparable evidence
remains explicit. A better-looking report does not count as value.

## Acceptance matrix

| Dimension | MVP acceptance |
| --- | --- |
| Disabled behavior | No recorder process, polling or files |
| Privacy | Strict allow-list; unknown fields rejected; canaries absent |
| Retention | Fixed history and four completed artifacts per recorder instance |
| Size | Final shareable artifact no larger than 256 KiB |
| Persistence | Atomic replacement; POSIX mode `0600`; stale temporary cleanup |
| Non-interference | Encode/write/replace failure does not signal or terminate vLLM |
| Trigger honesty | External observations never masquerade as an internal fatal kind/stage |
| Repeatability | Three repeated fatal trials per tested topology, with cleanup evidence |
| Overhead | Measured with identical interleaved workloads; no target claimed before data |
| Adoption | Per-incident decision impact recorded, including null results |

## Explicit non-goals

- modifying EngineCore in the external-preview phase;
- automatic upload, fleet identity or cross-restart stable fingerprints;
- parsing arbitrary logs or exception strings into a root cause;
- automatic restart or remediation;
- a general telemetry database, query engine or retention service;
- DP aggregation, NCCL collectives or GPU XID collection inside the recorder;
- claiming that external polling reproduces per-iteration EngineCore history.

## Release boundary

`v0.1.0-alpha.3` completes the M0 contract/writer work and the M1 bounded
collection work, reconciles M2, and publishes the completed subset of M3 with
explicit exclusions. It does not claim paired overhead, fresh KV-pressure,
multi-GPU observer behavior, diagnostic utility, production adoption, DP/NCCL
coverage, health-green no-progress detection, cross-producer correlation or safe
automatic remediation. Those claims remain gated by the unfinished M3–M5 work.
