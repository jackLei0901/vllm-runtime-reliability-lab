# GPU fault-boundary validation summary

## Objective

This batch checks what an external bounded recorder can and cannot establish around vLLM fatal failures. It covers a healthy baseline, intentional shutdown, abrupt EngineCore loss, attempted external GPU-memory pressure, and a controlled runtime `execute_model` CUDA OOM.

The goal is evidence collection and boundary discovery. The source-level OOM injection is a lab-only fault injector; it is not an upstream fix or part of the proposed RFC implementation.

## Environment

- GPU: NVIDIA GeForce RTX 4090, 24,564 MiB
- Python: 3.12.3
- PyTorch: 2.13.0+cu130
- vLLM: `0.26.1rc1.dev1103+g7ca49fbe4`
- Model: `Qwen3-4B-Instruct-2507-FP8`
- Topology: single GPU
- Recorder interval for the final controlled OOM run: 250 ms
- Recorder retained-history bound: 64 samples

The machine-readable environment snapshot is in [`environment.json`](environment.json).

## Experiment matrix

| Scenario | Fault or action | Request / health observation | Top-level server exit | Recorder result |
| --- | --- | --- | ---: | --- |
| Healthy baseline | None | Requests succeeded; final health remained 200 | Not forced | 19 samples; no incident |
| Intentional shutdown control | `SIGTERM` to API server | Normal shutdown path | 0 | `process_exit`, 2 retained samples |
| Abrupt EngineCore loss | `SIGKILL` to EngineCore | Health reached 503 before disconnect | 0 | `process_exit`, 4 retained samples |
| External GPU-memory pressure | A helper allocated about 10.75 GiB of remaining GPU memory | Completion still returned 200; health remained 200 | Cleaned up by harness | 23 samples; no incident |
| Controlled runtime CUDA OOM | One-shot lab injection raises `torch.OutOfMemoryError` in the next real `execute_model` call | Pre-trigger request 200; injected request 500; health then 503 and later connection refusal | 0 | `health_lost`, 140 collected / 64 retained |

## Results

### 1. Healthy baseline

The baseline collected 19 samples without producing an incident artifact. The final health probe was HTTP 200 and the request counter showed 11 successful requests. This confirms that the recorder does not manufacture a failure during the short healthy run.

Evidence: [`baseline-run2/dfx/run-summary.json`](baseline-run2/dfx/run-summary.json), [`baseline-run2/dfx/timeline.jsonl`](baseline-run2/dfx/timeline.jsonl), and [`baseline-run2/server.log`](baseline-run2/server.log).

### 2. Intentional SIGTERM control

Sending `SIGTERM` to the API server produced top-level exit code 0, as expected for an operator-initiated shutdown. The external recorder classified the terminal observation as `process_exit`.

Evidence: [`sigterm-control/server-exit-code.txt`](sigterm-control/server-exit-code.txt) and [`sigterm-control/dfx/incident-2026-09-01T09-44-32.645573+00-00.json`](sigterm-control/dfx/incident-2026-09-01T09-44-32.645573+00-00.json).

### 3. Abrupt EngineCore loss

After `SIGKILL` was sent to EngineCore, `/health` returned 503 before the endpoint disconnected. The top-level server nevertheless exited with code 0. No orphaned vLLM process remained.

The recorder again emitted `process_exit`. Therefore, the artifact proves that the observed process ended, but the external sampler alone cannot tell whether the exit was intentional or caused by the internal engine failure.

Evidence: [`enginecore-sigkill/server-exit-code.txt`](enginecore-sigkill/server-exit-code.txt), [`enginecore-sigkill/dfx/incident-2026-09-01T09-47-14.094823+00-00.json`](enginecore-sigkill/dfx/incident-2026-09-01T09-47-14.094823+00-00.json), and [`enginecore-sigkill/server.log`](enginecore-sigkill/server.log).

### 4. External GPU-memory pressure was not a reliable runtime-OOM injector

The pressure helper consumed roughly 10.75 GiB of the GPU memory left after server startup, briefly leaving almost no free device memory. The tested completion still returned HTTP 200 and the service remained healthy; the recorder produced no incident.

This negative result does **not** show that a runtime OOM cannot occur. It only shows that consuming externally visible free memory after vLLM startup is not a reliable way to force the model execution path to allocate and fail under this configuration. vLLM has already reserved its main runtime buffers, including the KV cache, during startup.

Evidence: [`runtime-cuda-oom/request-http-code.txt`](runtime-cuda-oom/request-http-code.txt), [`runtime-cuda-oom/gpu-pressure.log`](runtime-cuda-oom/gpu-pressure.log), and [`runtime-cuda-oom/dfx/run-summary.json`](runtime-cuda-oom/dfx/run-summary.json).

### 5. Controlled `execute_model` CUDA OOM

The final run used an isolated vLLM worktree with a one-shot sentinel. The next real `execute_model` invocation atomically consumed the sentinel and raised a lab-only `torch.OutOfMemoryError`.

Observed propagation:

1. A pre-trigger completion returned HTTP 200.
2. The injected completion returned HTTP 500.
3. EngineCore logged the injected `torch.OutOfMemoryError`.
4. The API server observed `EngineDeadError`.
5. `/health` returned 503, followed by connection refusal.
6. The top-level API process exited with code 0.
7. The recorder emitted a `health_lost` artifact with 140 samples collected and the latest 64 retained.
8. No vLLM orphan or GPU process remained after shutdown.

This reaches the actual EngineCore fatal path while keeping the injected cause deterministic. It reproduces the same process-lifecycle defect class as issue #48966: an unexpected fatal engine failure can still result in a successful top-level process exit on the tested main revision.

Evidence: [`runtime-cuda-oom-injected-final/server.log`](runtime-cuda-oom-injected-final/server.log), [`runtime-cuda-oom-injected-final/injected-http-code.txt`](runtime-cuda-oom-injected-final/injected-http-code.txt), [`runtime-cuda-oom-injected-final/server-exit-code.txt`](runtime-cuda-oom-injected-final/server-exit-code.txt), and [`runtime-cuda-oom-injected-final/recorder/incident-2026-09-01T13-30-26.613088+00-00.json`](runtime-cuda-oom-injected-final/recorder/incident-2026-09-01T13-30-26.613088+00-00.json).

The fault injector is documented by [`../../experiments/fault-recovery/inject_execute_model_cuda_oom.patch`](../../experiments/fault-recovery/inject_execute_model_cuda_oom.patch). It was applied only to an isolated private worktree; the main remote checkout was left clean.

## Interpretation for the RFC

The experiment separates three facts that should not be collapsed into one:

1. **Internal cause:** the server log identifies `torch.OutOfMemoryError` at the model-execution boundary.
2. **Externally observable failure:** the recorder sees health loss and eventual process loss, with bounded pre-failure history.
3. **Supervisor contract:** the top-level process still returns exit code 0, which is a launcher/lifecycle defect rather than something the incident-snapshot RFC should silently fix.

An external recorder is useful for chronology and independent process/health evidence, but it cannot reliably recover the internal causal boundary from those signals alone. This supports keeping the proposed v1 capture point inside EngineCore and attaching a typed `TriggerContext` at the fatal boundary. It also supports retaining the external harness as a control oracle rather than treating it as a replacement for the in-process recorder.

## Limitations

- This batch used one RTX 4090 and a single-GPU topology. It does not establish TP, DP-supervisor, NCCL, or cross-rank behavior.
- The runtime CUDA OOM is a deterministic source-level injection, not an organically occurring allocator exhaustion.
- The baseline is a short functional control, not a long-duration stability claim.
- A 250 ms external polling interval cannot capture every EngineCore iteration or establish ordering finer than the sampling cadence.
- Exit code 0 is reported only for the tested vLLM revision and launch path.

## Evidence integrity

The complete evidence archive is [`../gpu-20260901-fault-boundary.tar.gz`](../gpu-20260901-fault-boundary.tar.gz).

SHA-256:

```text
1a31f534580ef1d86b59d56a4053840573465a056165f329b8d69e4b24d789c8
```
