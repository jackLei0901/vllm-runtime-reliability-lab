# Requirements

## Objective

Produce a bounded external incident artifact for an operator investigating a
vLLM runtime failure, without changing vLLM or claiming internal state that an
external process cannot observe. This alpha is the first evidence layer of a
broader reliability-validation product; it is not the final product boundary.

## Functional requirements

1. **FR-001:** Poll health, selected Prometheus metrics, an explicit PID and aggregate GPU
   state on configurable cadences.
2. **FR-002:** Retain only the newest configured number of observations in memory.
3. **FR-003:** Trigger on process exit, health loss after a healthy observation, KV pressure
   or a preemption delta.
4. **FR-004:** Write a schema-valid incident artifact and provide a readable summary.
5. **FR-005:** Expose appended, overwritten and dropped observation counts.
6. **FR-006:** Support explicit, dry-run-first signal injection for isolated experiments.

## Safety and privacy requirements

1. **SR-001:** Shareable artifacts use a closed allow-list and reject unknown fields.
2. **SR-002:** They contain no prompt, token IDs, request IDs, paths, command lines or
   arbitrary configuration.
3. **SR-003:** Incident IDs use an ephemeral per-process HMAC key and are not comparable
   across restarts.
4. **SR-004:** Each artifact is at most 256 KiB and at most four completed artifacts are
   retained per output directory.
5. **SR-005:** Files use mode `0600` on POSIX and atomic replacement.
6. **SR-006:** Collection, encoding, rotation or write failure never signals or terminates
   the observed vLLM process.
7. **SR-007:** External triggers always report `internal_kind=unknown` and
   `internal_stage=unknown`.
8. **SR-008:** Target vLLM/Torch versions are never inferred from the recorder environment;
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

## Post-alpha problem and product gates

The following are planned evaluation gates, not requirements or capabilities of
the current alpha. They deliberately have no `FR-*` or `SR-*` identifiers until
their contracts and tests are implemented.

- **Health-green no-progress:** distinguish idle, healthy progress, long-running
  work, queued work with progress, a sustained stall and recovery without
  treating `/health=200` as proof of useful progress.
- **Producer identity:** establish a pre-failure `run_id`, per-process
  `producer_id`, content-addressed `artifact_id` and coordinator-assigned
  `incident_id` without requiring acknowledgement from a stalled producer.
- **Clock declaration:** record the clock domain and precision of every source;
  do not claim a total order when clocks are not comparable.
- **Closed manifest:** reference independently produced evidence by identity and
  hash rather than copy it into a new telemetry store.
- **Semantic join:** produce checkable vLLM process/progress facts such as a
  missing producer, state divergence or first externally observed divergence.
- **Optional stack producer:** collect CPU main-thread stacks only with explicit
  operator consent and attachment permission. Raw stacks remain private and use
  a separately reviewed contract rather than entering the current shareable
  schema.
- **Existing-system baseline:** compare with retained Prometheus/OpenTelemetry
  evidence rather than assuming an additional recorder is useful.
- **Linkage ablation:** compare unlinked and linked views of identical producer
  artifacts. Packaging alone is not counted as diagnostic value.

These gates do not authorize a general telemetry platform, automated root-cause
classification or automatic remediation.
