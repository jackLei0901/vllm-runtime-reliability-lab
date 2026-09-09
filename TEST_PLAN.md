# Test plan

## Alpha CPU gate

| Area | Required checks |
| --- | --- |
| Schema | valid example; unknown top-level and nested fields rejected |
| Privacy | prompt/token/auth canaries absent; external cause stays unknown |
| Identity | same value under different process keys produces different IDs |
| Recorder | bounded history; overwrite/drop counts; retained range |
| Writer | size cap; four-file rotation; atomic write; fail-open I/O |
| Permissions | final artifact is `0600` on POSIX CI |
| HTTP | healthy, 503, timeout and malformed metrics |
| Cadence | GPU polling occurs less frequently than health/metrics polling |
| Packaging | clean editable install; Ruff; CLI help; source compilation |

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

## Result boundary

Passing the CPU gate validates the artifact and failure-isolation contracts. It
does not demonstrate diagnostic utility, production adoption, DP/NCCL behavior
or an internal root cause.
