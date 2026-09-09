# Test plan

## Alpha CPU gate

| Area | Requirements | Required checks |
| --- | --- | --- |
| Schema | FR-004, SR-001, SR-007 | valid example; unknown top-level and nested fields rejected |
| Privacy | SR-002, SR-007 | prompt/token/auth canaries absent; external cause stays unknown |
| Identity | SR-003 | same value under different process keys produces different IDs |
| Recorder | FR-002, FR-003, FR-005 | bounded history; overwrite/drop counts; retained range |
| Writer | SR-004, SR-005, SR-006 | size cap; four-file rotation; atomic write; fail-open I/O |
| HTTP | FR-001 | healthy, 503, timeout and malformed metrics |
| Cadence | FR-001 | GPU polling occurs less frequently than health/metrics polling |
| Injection | FR-006 | dry run records intent without signalling the target |
| Version provenance | SR-008 | unknown remains null; explicit target values are preserved |
| Packaging | FR-004 | clean editable install; Ruff; CLI help; source compilation |

The machine-checkable requirement-to-test mapping is in
[`tests/cases.json`](tests/cases.json). `test_testplan.py` verifies that every
functional and safety requirement is mapped and that every referenced test
case exists.

## Post-alpha GPU gate

Run fatal scenarios at least three times. Record API-server exit, recorder exit
and orphaned processes independently.

| Scenario | Topology | Required observation |
| --- | --- | --- |
| Healthy then SIGTERM | 1 GPU | graceful control; no internal-cause claim |
| EngineCore SIGKILL | 1 GPU | bounded artifact and cleanup evidence |
| Controlled runtime CUDA OOM | 1 GPU | health/process chronology; cause remains unknown externally |
| KV pressure/preemption | 1 GPU | ordered pressure history and counter delta |
| Worker loss | TP=2 | rank/process transition and topology metadata |
| Writer path unavailable | CPU then GPU | vLLM outcome unaffected; writer error visible |
| Recorder disabled | 1 GPU | no recorder process, polling or artifacts |

## Overhead protocol

Interleave disabled and enabled arms using the same model, seed, request set and
arrival pattern. Report:

- request and token throughput;
- TTFT, TPOT and end-to-end latency;
- recorder CPU and RSS;
- collector latency by source;
- bytes written and artifact count.

Do not set an overhead target after reading the measurements. Publish raw paired
results and null outcomes.

The executable CPU harness contract and the unapproved GPU configuration
template are in [`experiments/overhead/`](experiments/overhead/). The CPU run
validates pairing, signatures, process cleanup and report generation only; its
latency values are not vLLM overhead evidence.

## Result boundary

Passing the CPU gate validates the artifact and failure-isolation contracts. It
does not demonstrate diagnostic utility, production adoption, DP/NCCL behavior
or an internal root cause.
