# v0.2 `collect` → `verify` contract

Status: implemented for review. The progress core, native bundle verifier,
legacy R3 projection, optional stack producer, bounded collector, and CLI are
covered by CPU-only tests; this is not yet a tagged v0.2 release.

This document defines the smallest interface that lets an engineer who is not
the lab author apply the lab's evidence method to an incident. It commits only
to claims already supported by published campaigns and keeps unavailable
evidence distinct from negative evidence.

## Scope

In scope for v0.2:

- explicitly identified process and endpoint targets, supplied by the operator;
- a bounded observation and evaluation interval;
- one explicit progress decision source and an optional corroborating source;
- evidence that useful work was expected before declaring no progress;
- optional external stack sampling with producer identity;
- a closed-shape, privacy-bounded evidence bundle;
- offline, fail-closed verification of that bundle.

Out of scope for v0.2:

- process, rank, or endpoint discovery;
- inferred identity between a process target and an endpoint target;
- root-cause classification beyond the closed verdict set below;
- remediation or restart;
- writes to the observed service except an explicitly requested client progress
  probe or the existing explicit `inject-signal` command;
- multi-node correlation or an automatic joiner across hosts.

## Commands

Normal progress collection supplies an endpoint and a process target:

```bash
vllm-dfx collect \
  --base-url http://127.0.0.1:8000 \
  --pid 12345 \
  --window 60 \
  --no-progress-window 10 \
  --decision-source server_counter \
  --output incident-2026-09-20/
```

An opt-in client probe changes the decision scope to one request:

```bash
vllm-dfx collect \
  --base-url http://127.0.0.1:8000 \
  --pid 12345 \
  --progress-request request.private.json \
  --decision-source client_request \
  --output incident-2026-09-20/
```

PID-only collection is allowed only when the operator explicitly accepts that
progress will normally be undetermined:

```bash
vllm-dfx collect \
  --pid 12345 \
  --observation-only \
  --window 60 \
  --output process-observation-2026-09-20/
```

Verification is always offline:

```bash
vllm-dfx verify incident-2026-09-20/
```

There is no default target. If neither a PID nor a base URL is supplied,
`collect` exits non-zero. PID-only use without `--observation-only` also exits
non-zero.

## Target identity

The process and endpoint are separate targets because they may be different
processes. In the #53859 campaign, for example, the stack target was EngineCore
while health came from the API server.

The publishable bundle contains:

```text
targets.process
  supplied
  pid
  start_identity.kind
  start_identity.value
  identity_source = operator_supplied

targets.endpoint
  supplied
  endpoint_id
  identity_source = operator_supplied

targets.relation = operator_asserted_same_incident
```

On Linux, the process start identity is `/proc/<pid>/stat` start-time ticks. On
Windows it is the process creation time. Other platforms must define an equally
stable, closed identity kind or leave process identity unavailable. A reused PID
means that the original process target is missing; the replacement process does
not inherit its evidence.

`endpoint_id` is a per-run, non-correlatable fingerprint. The base URL remains
a private operator input and is not copied into the publishable bundle. The lab
does not prove that the PID serves the endpoint; it records only the operator's
assertion that both targets belong to the same incident.

## Observation and evaluation intervals

The collection window may include readiness and warm-up. Verdicts use a
separate terminal evaluation interval ending at collection completion or at a
higher-precedence terminal event.

The bundle records monotonic start and end values for both intervals, the
configured minimum duration, and the minimum fresh sample count for each
producer. Only observations that name a source in `sampled_sources` count as a
fresh sample; cached values carried between collector cadences never count as
new evidence.

Server-counter `flat` requires at least two valid fresh samples spanning the
configured `--no-progress-window`. A single sample, a shorter span, a counter
reset, an identity discontinuity, or a collector gap produces
`insufficient_evidence`. Client-request chunks are events, not cadence samples;
client `flat` instead requires that the request remain incomplete across the
entire evaluation interval and that the interval meet the configured duration.

## Progress producers

The two producers remain independent:

1. **Server counter:** `vllm:generation_tokens_total`, aggregated over its label
   variants. A positive delta inside the evaluation interval is progress. Equal
   valid samples may establish flatness. A decrease is a counter reset and
   therefore insufficient evidence, not progress and not flatness.
2. **Client request:** an explicitly requested streaming inference probe.
   Progress requires a content-bearing token chunk. Role-only, usage-only and
   empty chunks do not count. A request that is still incomplete throughout the
   evaluation interval and produces no content-bearing chunk may establish
   flatness.

Each producer has exactly one state:

- `progressing`;
- `flat`;
- `producer_missing` — no usable producer was available;
- `insufficient_evidence` — the producer existed, but its samples cannot prove
  progress or flatness.

Each producer also records bounded reason codes, sample count, monotonic sample
bounds, and whether its identity remained stable.

`decision_source` is explicitly `server_counter` or `client_request`.
`progress_scope` is respectively `service` or `request`. The other producer is
corroborating evidence only and never silently changes the verdict. A
disagreement is recorded as `producer_conflict: true` and remains visible to the
reader.

The client request is an active, opt-in probe. Its input file is private. The
publishable bundle retains only a digest and bounded request metadata; it never
retains prompt text, response text, token IDs, headers, credentials, or raw
response chunks.
Collection rejects plans above 10,000 cadence samples and request or individual
HTTP event inputs above 1 MiB. For the client producer, the public timeline
retains at most the latest content-bearing event because one event inside the
evaluation interval is sufficient to establish progress; role, usage, empty,
and earlier content events are not persisted.

## Demand evidence

A flat producer is not sufficient to claim no progress: an idle server is
expected to have a flat generation counter.

Demand has one of three states:

- `present`;
- `absent`;
- `insufficient_evidence`.

For `client_request`, demand is present only when the probe started before the
evaluation interval and remained incomplete throughout it. For
`server_counter`, demand is present only when at least two fresh metrics samples
span the evaluation interval and every usable sample reports one or more
running or waiting requests. Missing, malformed, discontinuous, or contradictory
demand samples are insufficient evidence.

`alive_health_ok_no_progress` requires `demand = present`. Flat progress with
absent or insufficient demand yields `undetermined`.

## Closed verdict set and precedence

v0.2 emits exactly one verdict. `verify` recomputes it from observations rather
than trusting the value stored in `summary.json`.

The deterministic precedence is:

1. `process_missing`;
2. `health_lost`;
3. `progress_observed`;
4. `alive_health_ok_no_progress`;
5. `undetermined`.

| Verdict | Required evidence |
| --- | --- |
| `process_missing` | The supplied process target stopped existing or its start identity changed. |
| `health_lost` | The endpoint was healthy at least once and then produced the configured number of consecutive fresh non-2xx or unreachable health samples; the original process target, if supplied, has not already become missing. |
| `progress_observed` | The selected decision producer reports `progressing` in the evaluation interval and no higher-precedence event occurred. |
| `alive_health_ok_no_progress` | A process and endpoint were both supplied; the original PID identity remained alive throughout the evaluation interval; every fresh health sample in that interval was 2xx with no health collection error; at least two fresh health samples span the interval; the decision producer is `flat`; demand is `present`; and no higher-precedence event occurred. |
| `undetermined` | Anything else, including a missing or insufficient decision producer, insufficient demand, inadequate health sampling, or observation-only survival. |

The stored verdict also carries `progress_scope` and `decision_source`; therefore
a request-scoped result cannot be presented as service-wide no progress.
`undetermined` is a successful collection outcome, not a verifier error.

Endpoint-only collection is intentionally conservative in v0.2. It may establish
`progress_observed` or `health_lost`, but it cannot establish
`alive_health_ok_no_progress`: repeated 2xx responses prove endpoint
responsiveness, not process liveness. An endpoint-only flat producer with demand
present therefore yields `undetermined` until a stable process target is also
supplied.

## Stack evidence

Stack sampling is opt-in and can never change the verdict. Its section is always
present with one state:

- `disabled`;
- `produced`;
- `unavailable`.

The publishable producer identity contains only:

- sampler name and version;
- sampler binary SHA-256 when readable;
- bounded attach-context fields such as platform and Yama `ptrace_scope`;
- exit status, output-produced flag, raw-output SHA-256, and bounded error kind.

The resolved executable path, complete command, raw stack, function names,
module paths, and line numbers are private. A future explicit redaction format
may publish allow-listed stack predicates, but v0.2 does not publish arbitrary
frames. A failed or unauthorized sampler remains visible as `unavailable`; its
section is never removed and its absence is never interpreted as absence of a
blocked thread.

## Bundle shape and privacy boundary

```text
incident-2026-09-20/
  summary.json          # closed-shape, publishable
  observations.json     # closed v0.2 observation schema
  private/              # optional and excluded from public verification
    endpoint.json
    request.json
    stacks.txt
    raw-timeline.json
```

`summary.json` uses `schema_version: "incident-evidence-bundle-v1"`. It contains
targets, interval bounds, decision and corroborating producers, demand evidence,
stack producer identity, the recomputed verdict, and a `sources` map containing
the expected publishable file set and SHA-256 digests.

`observations.json` uses a new closed schema. The published
`external-runtime-observation-v1` schema remains immutable: it does not retain
`generation_tokens_total`, process start identity, or always-output bounded
windows and must not be silently extended in place.

No publishable file contains a base URL, hostname, prompt, completion, token,
header, credential, complete filesystem path, arbitrary command line, raw stack,
or environment variable.

## Integrity and verification claims

`verify` provides structural and internal-consistency checking, not
authentication against a malicious editor. It fails closed when:

1. a schema version is unknown, a required field is missing, or an unexpected
   field or publishable file is present;
2. a digest in `sources` does not match its file;
3. sequences or monotonic bounds are invalid;
4. a cached observation is counted as a fresh producer sample;
5. the recorded producer, demand, conflict, or verdict fields differ from the
   values recomputed from observations;
6. the observation or evaluation interval lacks the declared duration or fresh
   sample count;
7. a claim cites `producer_missing` or `insufficient_evidence` as negative
   evidence;
8. an identity source is not `operator_supplied`;
9. a process-dependent claim lacks a stable process start identity;
10. a publishable field crosses the privacy boundary.

Changing whitespace or reserializing a valid self-generated `summary.json`
cannot be detected without an external trust anchor and is not claimed as
tamper detection. For an immutable published bundle, a release manifest may pin
the `summary.json` digest, as the existing replay pins published evidence.

Verification requires no network access, GPU, PyTorch, vLLM, target process, or
stack sampler.

## Implementation boundary

Reuse these existing components:

| Need | Existing module |
| --- | --- |
| health, process, metrics, and GPU sampling cadence | `collectors.py` |
| bounded in-memory history | `recorder.py` |
| Prometheus text parsing and label aggregation | `prometheus.py` |
| atomic private writes and digests | `schema.py`, `external_writer.py` |

The implementation is composition plus a new contract, not only a CLI alias:

- `collect_metrics()` currently does not retain
  `vllm:generation_tokens_total`, even though the parser recognizes it;
- the published v1 observation schema has no field for that counter or process
  start identity;
- `record` writes a public artifact only after an incident trigger, while
  `collect` must always close its bounded observation window.

The implementation is split across:

- `progress.py`: pure producer, demand, conflict, and verdict logic;
- `stacks.py`: opt-in `py-spy` execution and private/public producer identity;
- `bundle.py`: deterministic summary and observation assembly;
- `verify_bundle.py`: offline fail-closed recomputation and legacy R3 projection;
- `collect_bundle.py`: bounded process, health, metrics, and opt-in client-probe
  composition;
- `cli.py`: `collect` and `verify` wiring.

The shipped ordering preserved the review gates: claim logic first, then native
bundle verification, then stack and network I/O, and CLI wiring last.

## Acceptance tests

The progress and verdict core must cover at least:

- an idle flat counter with no demand yields `undetermined`;
- a flat counter with stable PID, all-2xx health, and continuous demand yields
  `alive_health_ok_no_progress`;
- a rising decision counter yields `progress_observed`;
- a missing metrics endpoint yields `producer_missing`, never `flat`;
- one valid counter sample yields `insufficient_evidence`;
- a counter reset yields `insufficient_evidence`;
- PID reuse yields `process_missing`;
- configured consecutive health failures yield `health_lost`;
- an endpoint that never produced a prior 2xx sample cannot yield `health_lost`;
- endpoint-only flat progress with demand and 2xx health yields `undetermined`;
- decision and corroborating producer disagreement records a conflict without
  silently changing decision scope;
- both disagreement directions are covered: flat/advancing and advancing/flat;
- an unavailable stack sampler leaves the verdict unchanged and remains visible;
- PID-only observation yields `undetermined` while the process survives and
  `process_missing` if the original identity disappears;
- cached-only server demand samples yield `insufficient_evidence`;
- a client request that starts after the evaluation interval opens yields
  `insufficient_evidence` for demand;
- a semantic mutation of a digest, identity, producer state, demand state, or
  verdict fails verification;
- cached collector values cannot satisfy a fresh-sample minimum.

The published #53859 Stage 1 R3 replay remains the legacy oracle and must
continue to pass unchanged. Its public summary contains only one
`health_during_stall` classification and no PID start identity, so it cannot be
relabeled as a native v0.2 bundle. Instead:

1. a native synthetic fixture reproduces the R3 facts and must yield
   `alive_health_ok_no_progress` for the base pause arm and
   `progress_observed` for the fix pause arm;
2. a compatibility projection of the public R3 summary must report which v0.2
   evidence is unavailable rather than inventing samples;
3. if retained private R3 observations can satisfy the new contract, they may
   generate a separately identified native bundle without altering the frozen
   published result.

## Success criteria

Two external engineers run `collect` on incidents the lab did not produce, and
at least one external issue or contribution improves the bundle format or its
usability. Bug counts and stars are not technical success signals for this
release.
