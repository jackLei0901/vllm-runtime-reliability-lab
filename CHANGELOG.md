# Changelog

## Unreleased

- Reframe the product around health-green loss of progress and cross-producer
  evidence correlation rather than generic data collection.
- Document the planned identity, clock, closed-manifest and process/progress
  semantic-join boundaries without claiming them as current alpha features.
- Add a PyTorch Flight Recorder before/after case and an explicit
  unlinked-versus-linked value test with transfer limitations.
- Add a detailed Chinese guide covering usage, evidence boundaries, validation,
  current product readiness, and the path from public alpha to production preview.
- Add a Chinese architecture/design document and a gated product roadmap.
- Add a CPU-testable paired-overhead harness and an explicitly unapproved GPU
  plan template.
- Record collector latency by source, preserve graceful recorder summaries, and
  report adjacent disabled/enabled relative deltas.

## 0.1.0-alpha.3 - 2026-09-09

- Document the development extra required to run the validation suite.
- Install the schema under the repository's current project name.
- Restore machine-checked requirement-to-test traceability.

## 0.1.0-alpha.2 - 2026-09-09

- Stop inferring target vLLM and Torch versions from the recorder environment.
- Add explicit `--target-vllm-version` and `--target-torch-version` inputs.
- Preserve unknown target versions as `null`.
- Publish repeated RTX 4090 fatal-path, writer fail-open, and disabled-control
  validation with explicit limitations.

## 0.1.0-alpha.1 - 2026-09-09

- Add a closed, machine-checkable external incident artifact contract.
- Collect only allow-listed aggregate runtime fields in shareable artifacts.
- Use per-process HMAC incident identifiers that do not correlate across restarts.
- Bound artifacts to 256 KiB and rotate to at most four completed files.
- Create POSIX artifacts with mode `0600` and isolate writer failures.
- Separate health, metrics, process and GPU collection cadences.
- Disable raw timeline persistence by default.
- Report appended, overwritten and dropped observation counts.
- Add deterministic fake-service, schema, privacy, writer and recorder tests.
