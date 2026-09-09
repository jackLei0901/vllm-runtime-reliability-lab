# Design

## Problem model

The product is not a generic data collector. It addresses two evidence gaps:

1. a service can remain process- and health-alive while useful inference progress
   has stopped; and
2. a distributed failure is expressed as relationships between producers, while
   each producer can record only its local state.

The current alpha covers bounded single-target capture. A planned v0.2 will test
health-green no-progress detection and offline cross-producer correlation. Those
capabilities are not claims of the current release.

## Trust boundary

The recorder is a separate process. It can observe HTTP status, selected
low-cardinality metrics, an explicitly supplied process ID and aggregate GPU
state. It cannot observe the EngineCore exception object or identify the exact
internal execution stage.

```text
vLLM HTTP + metrics ----|
explicit PID -----------|--> CadencedCollector
nvidia-smi -------------|          |
                                   v
                            ExternalObservation
                                   |
                              bounded deque
                                   |
                            typed external trigger
                                   |
                                   v
                      IncidentWriter -> incident JSON
```

The file contract is named `external-runtime-observation-v1`. It is deliberately
different from the in-process `incident-snapshot-v1` proposed in vLLM RFC #54229.

## Collection

`CadencedCollector` caches the most recent value from each source. Health,
metrics and process polling default to one second; `nvidia-smi` defaults to five
seconds because spawning it is materially more expensive. Transport failures are
reduced to fixed error kinds. Raw exception text and URLs are not serialized.

Python/platform fields describe the recorder. Target vLLM and Torch versions are
never inferred from the recorder environment; they remain `null` unless the
operator supplies exact observed-server values explicitly.

Prometheus input is projected into four fields only: KV-cache usage,
preemptions, running requests and waiting requests. Label variants are summed
before projection, avoiding high-cardinality output.

## Recording

The recorder is single-writer. Each successful observation receives a monotonic
sequence number and is appended to a fixed-size deque. Overwrites and unexpected
collection failures are counted separately. The retained sequence range makes
the overwrite accounting checkable from the artifact.

Raw JSONL persistence is disabled by default. The optional private timeline is
outside the shareable contract and may grow without a bound.

## Triggering

The trigger kind describes only the external condition:

- `process_exit`
- `health_lost`
- `kv_pressure`
- `preemption_storm`

Health loss is eligible only after health was observed as successful. This
prevents startup connection failures from being mislabeled as a service death.
No timing or log-string heuristic maps an external trigger to CUDA OOM, worker
loss or intentional shutdown.

## Persistence

`IncidentWriter` accepts an `ExternalIncidentArtifact`, validates its explicit
projection, encodes it, enforces the size limit, rotates old files and writes by
temporary-file replacement. Temporary and final files are created with mode
`0600` on POSIX.

Writing is fail-open with respect to vLLM: validation, size and I/O failures are
returned as a bounded error kind instead of being raised through the recorder.
Rotation runs before each write, which also bounds repeated crash-loop output.

## Identity

An incident ID is an HMAC over recorder-local values using a random key created
for the process. It groups output within one recorder lifetime without providing
a stable identifier across restarts or hosts.

## Evolution

The conversion seam is `ExternalObservation`. A future in-process producer may
reuse bounded-recorder and writer ideas, but it must emit a separately versioned
EngineCore contract. The external schema will not be relabeled as internal
evidence.

## Planned correlation boundary

Cross-layer correlation does not require this project to own ingestion, storage
or a query service. The proposed boundary is a closed manifest plus one concrete
vLLM process/progress joiner:

```text
independent producer artifacts
        | run identity, producer identity, clocks, content hashes
        v
closed correlation manifest
        | topology and time alignment
        v
vLLM process/progress semantic join
        |
        v
first observed divergence + explicit unknowns
```

The identities have separate roles:

- `run_id` is random or operator-supplied before failure and shared by producers;
- `producer_id` identifies one process instance and its declared role/rank;
- `artifact_id` is content-addressed;
- `incident_id` is assigned by the bundle coordinator and does not require a
  stalled producer to acknowledge it.

Each producer must declare its clock domain. Same-host monotonic clocks may be
aligned when the platform contract supports it; cross-host monotonic clocks are
not assumed comparable. Unknown synchronization error prevents a total-order
claim.

The first semantic join will report missing producers, state/progress divergence
and ordering limitations. It will not translate those observations into a CUDA,
NCCL or scheduler root cause.

The first optional evidence adapter after the manifest will target a gap stated
by the PyTorch Flight Recorder team: distributed CPU main-thread stack context.
An external sampler such as `py-spy` can produce per-process stack snapshots;
the lab can associate those snapshots with declared producer/rank identity and
reference rank-local Flight Recorder dumps when they already exist. This is an
adapter and join experiment, not a core runtime dependency. Sampling permission,
capture failure, stack timestamp precision and observer overhead must remain
explicit in the artifact.

Raw stacks are private by default because function names and source paths may be
sensitive. They do not enter `external-runtime-observation-v1`. Collection
requires explicit operator opt-in and the platform's process-attachment
permission; a manifest may reference only a separately versioned, reviewed
stack artifact and its content hash.

Prometheus and OpenTelemetry remain continuous telemetry systems. A correlation
manifest may reference their reviewed outputs, PyTorch/NCCL Flight Recorder
dumps, process evidence and supervisor events. It does not copy their storage or
query responsibilities.

## Value test

The correlation design must pass an unlinked-versus-linked ablation using the
same underlying producer records. Required structural outcomes include capture
coverage, join coverage, tamper detection, missing-producer detection and the
ability to identify the first externally observed divergence when the declared
clock precision permits it. Human utility is evaluated separately through
hypotheses eliminated and time to the next diagnostic action.

PyTorch Flight Recorder is the closest published precedent: its offline
cross-rank alignment produces mismatch facts that a single rank cannot define.
See [`PRIOR_ART_AND_VALUE.md`](PRIOR_ART_AND_VALUE.md) for the before/after and
the boundary on transferring training evidence to inference serving.
