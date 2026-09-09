# Requirements

## Objective

Produce a bounded external incident artifact for an operator investigating a
vLLM runtime failure, without changing vLLM or claiming internal state that an
external process cannot observe.

## Functional requirements

1. Poll health, selected Prometheus metrics, an explicit PID and aggregate GPU
   state on configurable cadences.
2. Retain only the newest configured number of observations in memory.
3. Trigger on process exit, health loss after a healthy observation, KV pressure
   or a preemption delta.
4. Write a schema-valid incident artifact and provide a readable summary.
5. Expose appended, overwritten and dropped observation counts.
6. Support explicit, dry-run-first signal injection for isolated experiments.

## Safety and privacy requirements

1. Shareable artifacts use a closed allow-list and reject unknown fields.
2. They contain no prompt, token IDs, request IDs, paths, command lines or
   arbitrary configuration.
3. Incident IDs use an ephemeral per-process HMAC key and are not comparable
   across restarts.
4. Each artifact is at most 256 KiB and at most four completed artifacts are
   retained per output directory.
5. Files use mode `0600` on POSIX and atomic replacement.
6. Collection, encoding, rotation or write failure never signals or terminates
   the observed vLLM process.
7. External triggers always report `internal_kind=unknown` and
   `internal_stage=unknown`.
8. Target vLLM/Torch versions are never inferred from the recorder environment;
   unknown values remain `null` unless explicitly supplied.

## Alpha acceptance

- Unit and fake-service tests pass on Python 3.10, 3.12 and 3.13 in CI.
- A clean editable install exposes the `vllm-dfx` command.
- The synthetic example validates against the published v1 schema.
- Raw timeline persistence is off unless explicitly requested.
- A writer failure is visible in bounded health state and does not escape the
  recorder.

GPU overhead, repeated fatal trials, supervisor recovery and operator adoption
are post-alpha gates, not claims of this release.
