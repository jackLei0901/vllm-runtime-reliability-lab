# Fault recovery

## Completed case: #48966 / PR #52178

The contract under test is: an unexpected EngineCore death must make the top-level process exit nonzero, while an intentional SIGTERM must still exit zero. Each trial first waits for `/health=200`, requires one successful completion, and then injects the signal.

### Single-GPU paired result

| Path | Injection | Top-level exit code |
| --- | --- | ---: |
| baseline | SIGKILL EngineCore | 0 |
| patched | SIGKILL EngineCore | 1 |
| patched | SIGTERM API server | 0 |

### TP=2 result

Three process-level trials were run on two RTX 4090 GPUs:

| Injection | Healthy request | Top-level exit code | Orphan process |
| --- | ---: | ---: | --- |
| SIGKILL `VllmWorker-0` | 1 | 1 | none |
| SIGKILL EngineCore | 1 | 1 | none |
| SIGTERM API server | 1 | 0 | none |

The raw evidence remains in the private workspace under `vllm_validation_20260820/52178/`. A DP supervisor has a separate parent-process lifecycle and is outside this result's scope.

## Re-run with the common tool

```bash
vllm-dfx snapshot-env --output results/48966/environment.json
vllm-dfx record --pid "$API_PID" --output results/48966 --stop-on-incident
vllm-dfx inject-signal --pid "$ENGINE_PID" --signal SIGKILL \
  --event-log results/48966/injections.jsonl
```

The launcher must separately record the top-level exit code, complete process tree, and restart time. The incident recorder supplies pre-failure state; it does not replace process-level assertions.
