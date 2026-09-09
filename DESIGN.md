# Design

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
