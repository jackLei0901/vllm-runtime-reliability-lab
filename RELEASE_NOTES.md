# v0.1.0-alpha.1

This first public alpha turns the existing vLLM reliability experiments into a
reproducible, installable external recorder with a deliberately narrow trust
boundary.

## Included

- a closed `external-runtime-observation-v1` JSON contract;
- bounded, independently cadenced health, metrics, process, and GPU sampling;
- process-exit, health-loss, KV-pressure, and preemption triggers;
- 256 KiB artifact cap, four-file rotation, atomic replacement, and POSIX
  `0600` permissions;
- privacy canaries and per-process HMAC incident identifiers;
- fail-open collection and persistence behavior;
- a fake vLLM HTTP service and 30 CPU tests;
- reviewed summaries of earlier RTX 4090 and TP=2 experiments.

## Validation completed for this tag

- clean editable install on Windows;
- Ruff lint and format checks;
- 30 unit/local-service tests passed, with the POSIX permission test skipped on
  Windows and scheduled for Linux CI;
- end-to-end fake-service health-loss capture and Markdown summarization;
- published example validated against the Draft 2020-12 JSON Schema.

## Not established by this alpha

Fresh GPU validation, production diagnostic utility, DP/NCCL behavior,
long-duration stability, and automatic remediation remain out of scope. See
`TEST_PLAN.md` for the next evidence gates.
