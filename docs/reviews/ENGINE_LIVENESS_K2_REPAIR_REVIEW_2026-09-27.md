# K2 nested shutdown check: review repair record

This is the pre-run repair record. The subsequent Linux CPU result, including
the explicit low-memory `fork` mode, is in
[the K1/K2 run record](ENGINE_LIVENESS_K1_K2_CPU_RUN_2026-09-27.md).

Status: **local apparatus repair, not a C2 runtime result**. This note reviews the CPU-only [K2 script](../../experiments/engine-liveness-contract/k2_nested_shutdown_budget.py) against the [engine liveness inventory](../ENGINE_LIVENESS_CONTRACT_INVENTORY_2026-09-27.md). Nothing here has been run against a Linux vLLM installation, and no upstream claim follows from it.

## Review findings and changes

1. The original nested score could call C2 confirmed when the outer process was SIGKILLed **before** it entered `_ensure_worker_termination`: it required only `inner_schedule_completed=False` and `engine_sigkilled_by_outer=True`. The score now requires observed EngineCore SIGTERM reception and an `inner_start` marker for **every** cell. Missing either produces `apparatus_failed`.
2. The fake EngineCore previously signalled ready after starting worker processes, without knowing whether their SIGTERM handlers were installed. Each worker now acknowledges handler installation through a separate multiprocessing event. Engine readiness follows all acknowledgments; failure is bounded and recorded as apparatus failure.
3. The result name is now `c2_tested_nested_budget_shortfall`. It applies to the finite tested `x` values and the deliberately slow worker; the broader `x + 4 > x` budget argument remains source-derived. The inventory no longer says the first inner SIGTERM step is impossible: it may race with the outer kill. The full escalation has no reserved time.

## Verification performed locally

- [Five pure classification tests](../../tests/test_engine_liveness_k2.py) pass, including negative vectors for missing `inner_start`, unready worker, failed control and a nested cell that completes. The intended slow-worker vector produces only `c2_tested_nested_budget_shortfall`.
- CPython 3.14.2 compiled both K1 and K2 scripts without syntax errors.
- With `PYTHONPATH=src`, the Lab suite ran **323 tests, OK, 15 skipped**. The scripts, inventory and this record remain untracked, so no staged-diff check is claimed.
- Repaired K2 script SHA-256: `9a356ff6763098132fc0e7198fe5b069a6730d9d50ef63d29eded1685f0c63ee`. Classification test SHA-256: `165a911d21d69c12e21879c3d5810cefe0cbdfb03585a5028b2329ea160abf9d`.

## Remaining execution gate

Run K1 and K2 in a Linux environment with the pinned vLLM source/build and verify that the provenance reports the intended version and source digests. Capture the per-cell JSON and exit codes before making a C1/C2 runtime claim. The current desktop session has no active `2222` tunnel, and Windows cannot score these POSIX-process tests. A CPU-only Linux host is sufficient; GPU time is not required. Keep this inventory and review record local until the owner decides what to publish.
