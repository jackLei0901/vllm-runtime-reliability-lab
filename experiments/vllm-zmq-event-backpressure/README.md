---
model_cells: [M3, M4]
status: active
upstream_exit: vLLM PR 36451 validation comment; #53859 case
last_scored: 2026-09-25
---

# ZMQ event-backpressure and health-signal case

M3 tracks admitted demand and useful-work progress; M4 records the missing
no-progress signal path. The existing [Stage 1 R3 result](STAGE1_R3_RESULT_2026-09-16.md)
and [source-equivalent health-ping result](PR36451_SOURCE_EQUIVALENT_RESULT_2026-09-25.md)
retain their separate evidence grades. The latter is a loop-liveness check,
not a token-progress guarantee.
