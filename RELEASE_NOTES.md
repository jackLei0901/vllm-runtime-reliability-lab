# v0.1.0-alpha.2

This corrective alpha preserves the external observer's evidence boundary: it
does not infer target package versions from the recorder's Python environment.

## Correction

- `vllm_version` and `torch_version` now remain `null` by default.
- Operators may supply exact observed-server values with
  `--target-vllm-version` and `--target-torch-version`.
- Two regression tests cover both unknown and explicitly supplied versions.

All `alpha.1` functionality remains: closed schema, bounded cadenced history,
privacy canaries, four-file rotation, fail-open persistence, fake-service tests,
and Markdown summaries.

Fresh RTX 4090 validation covers intentional `SIGTERM`, three EngineCore
`SIGKILL` trials, three controlled `execute_model` CUDA OOM trials, an
unavailable writer directory, and a recorder-disabled control. See
[`results/gpu-20260909-alpha2/VALIDATION_SUMMARY.md`](results/gpu-20260909-alpha2/VALIDATION_SUMMARY.md)
for the exact environment, exclusions, and remaining gates.

Wheel SHA-256:

```text
fb5ac984e19347fbb304c41ae3da8695e0a8a1c257b4df1b8f691982b3e60ce6
```
