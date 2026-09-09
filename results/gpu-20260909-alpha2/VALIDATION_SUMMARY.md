# GPU validation of v0.1.0-alpha.2

## Objective

This batch validates the corrected external-observer contract against a real
vLLM server. In particular, it checks explicit target-version provenance,
bounded and private artifact output, fatal-failure repetition, writer
non-interference, and the disabled control.

This is a contract and failure-boundary validation. It is not an overhead
benchmark, a long-duration stability test, or evidence that the external
recorder can identify an internal root cause.

## Environment

- GPU: NVIDIA GeForce RTX 4090, 24,564 MiB
- Driver: 580.105.08
- PyTorch: `2.13.0+cu130`
- CUDA runtime reported by PyTorch: 13.0
- vLLM module: `0.23.1rc1.dev1447+ga454a1dd2`
- vLLM base commit: `c4e969294ecab9ffefb995b734303f13f62b723f`
- Model: `Qwen2.5-VL-3B-Instruct`
- Topology: single GPU
- Recorder sample interval: 200 ms in the fatal trials
- Retained-history bound: 64 samples in the fatal trials

The server checkout also contained the unmerged process-exit changes from
[vLLM PR #52178](https://github.com/vllm-project/vllm/pull/52178). The runtime
OOM trials used a separate worktree with the documented one-shot fault-injection
patch. Results below must therefore not be attributed to an unmodified vLLM
release.

## Valid experiment matrix

| Scenario | Repetitions | Request observation | Server exit | Recorder exit | Artifact result |
| --- | ---: | --- | ---: | ---: | --- |
| Intentional `SIGTERM` | 1 | pre-signal completion 200 | 0 | 0 | schema valid; 6 samples; mode `0600` |
| EngineCore `SIGKILL` | 3 | pre-trigger completion 200 | 1 in all 3 | 0 in all 3 | schema valid; 8 samples in each |
| Controlled `execute_model` CUDA OOM | 3 | pre-trigger 200; injected request 500 | 1 in all 3 | 0 in all 3 | schema valid; 6-7 samples |
| Writer directory unavailable | 1 | completion 200 before and after capture | normal control shutdown | 0 | no artifact; fail-open error visible |
| Recorder disabled | 1 | completion 200 | 0 after `SIGTERM` | not started | no recorder process or artifact |

No orphaned vLLM process remained after any of the six valid fatal trials.

## Findings

### 1. Target package versions are explicit observations

The recorder ran from a different Python environment than the server. Every
shareable artifact nevertheless contained the explicitly supplied server
versions shown above. The alpha.2 implementation did not substitute package
metadata from the recorder environment. When those arguments are omitted, the
schema contract leaves both values `null`, as covered by the CPU regression
suite.

### 2. The external boundary preserves chronology, not internal cause

Intentional shutdown, EngineCore loss, and the controlled CUDA OOM all reached
the same externally observable `health_lost` trigger. The artifact's internal
exception kind and execution stage remained unknown. This is the intended
boundary: an external sampler can retain health, process, GPU, and selected
metric history, but it cannot reconstruct an EngineCore exception type merely
from those observations.

The different top-level exit codes are supplied by the server lifecycle under
test, not inferred by the recorder. With the PR #52178 changes present,
intentional `SIGTERM` exited 0 while both unexpected fatal scenarios exited 1.

### 3. Fatal-path behavior repeated cleanly

Three EngineCore `SIGKILL` trials and three controlled runtime-OOM trials each
produced one bounded, schema-valid incident artifact. The recorder exited 0,
and process cleanup left no orphan behind. Repetition reduces the chance that a
single scheduler or signal-ordering accident was mistaken for the contract.

### 4. Artifact failure remained fail-open

The writer was directed to `/proc/vllm-dfx-unwritable` while the server was
healthy, and a zero KV threshold forced a capture attempt. The recorder logged
`incident writer failed open: io`, wrote no artifact, and exited 0. A completion
returned HTTP 200 both before and after the failed write. This validates
non-interference for this I/O-failure case; it is not a proof for every storage
failure mode.

### 5. Disabled means absent

In the disabled control, no recorder process was started, no polling artifact
was created, a completion returned HTTP 200, and normal `SIGTERM` produced
server exit 0.

## Evidence exclusions

Two setup attempts are excluded from the matrix. A shell wrapper was initially
signalled instead of its API-server child, leaving that child alive; the next
control could not start because the GPU was still occupied. The child was
removed and the disabled control was rerun with the real server PID. These are
harness errors, not product observations, and no conclusions use them.

Private raw logs, process identities, injection records, and run summaries stay
on the isolated test host. This public report contains only reviewed aggregate
facts. The shareable artifacts themselves followed the closed schema and were
created with POSIX mode `0600`.

## Remaining gates

- A paired disabled/enabled overhead run has not been completed for alpha.2.
- Fresh KV-pressure/preemption coverage has not been completed for alpha.2.
- TP=2 worker-loss behavior was validated previously for PR #52178, but this
  alpha.2 package has not been re-run as a multi-GPU observer.
- DP supervisor, NCCL failure, long-duration operation, and production utility
  remain unvalidated.

