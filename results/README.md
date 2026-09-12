# Results policy

Each experiment uses its own directory and records the environment, raw timeline, workload output, service logs, injection events, and summary. Large reviewed bundles may be attached to a GitHub Release, but the README must record the SHA-256 digest, code commit, and generation command.

Do not present any of the following as a validated conclusion:

- example data that was not executed
- performance comparisons with mismatched environments or server arguments
- failure trials that preserve a log excerpt but not the complete exit code or request result
- multi-GPU conclusions extrapolated from a single-GPU run

## Published summaries

- [`organic-hang-20260912/`](organic-hang-20260912/): derived-only evidence for
  a four-GPU known-answer FSDP2 hang reconstruction, including a standalone
  verifier and explicit retention limitations.
- [`gpu-20260909-alpha2/VALIDATION_SUMMARY.md`](gpu-20260909-alpha2/VALIDATION_SUMMARY.md):
  fresh RTX 4090 validation of the alpha.2 schema, fatal-path repetition,
  fail-open writer, and disabled control.
- [`gpu-20260901-fault-boundary/VALIDATION_SUMMARY.md`](gpu-20260901-fault-boundary/VALIDATION_SUMMARY.md):
  earlier external-observer boundary experiments.
