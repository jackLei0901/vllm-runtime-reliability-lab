# Unified review: engine-liveness contract and next validation gate

Decision: **no GPU booking for the current Lab RFC outline**. The
[source inventory](../ENGINE_LIVENESS_CONTRACT_INVENTORY_2026-09-27.md),
[internal outline](../ENGINE_LIVENESS_RFC_OUTLINE_2026-09-27.md),
[K1/K2 CPU result](ENGINE_LIVENESS_K1_K2_CPU_RUN_2026-09-27.md), and
[K5 dual-GPU result](ENGINE_LIVENESS_K5_DUAL_GPU_RESULT_2026-09-27.md)
are sufficient to review the *contract question*. They are not sufficient
to claim a CUDA leak, a validated per-request progress detector, or the root
cause of any organic issue. This review checks the pinned source and retained
records; it does not refresh upstream PR states or complete duplicate search.

## Evidence grades that must remain separate

| Cell | What is established | What is not established | GPU needed now? |
| --- | --- | --- | --- |
| C1 / C8 | Source couples request-drain time to process-kill grace. K1 confirms the zero-grace helper kills a fake child before its bounded SIGTERM handler can finish. | CUDA resource leak, exact client-visible abort behavior, or a safe new default. | No. Optional platform corroboration belongs after design review. |
| C2 | The inner worker wait and outer EngineCore grace share one variable; K2 tests three `x` values and a completing longer-grace control. The general `x + 4 > x` conflict is source-derived. | Which default budget is right across platforms. | No. |
| C3 | K5 scores TP=1 green health during a 45 s worker hold and recovery. TP=2 observes 200→503 near the experimentally shortened 30 s RPC deadline (pinned default: 300 s), request 500, and `sample_tokens` timeout; source explains the default async wait order. | Strict TP=2 K5 score (frozen marker was `execute_model`), indefinite observed TP=1 hang, or linkage to a cited organic report. Effective async mode was not emitted as a separate runtime witness. | No for the outline. A fresh GPU run is justified only by a later implementation or reviewer question that needs the exact two-arm assertion. |
| C4 / C7 | Pinned source shows lower-level health checks are unwired and the named iteration timeout has no reader. | User-visible impact or absence of a newer replacement contract. | No; refresh source and duplicates first. |

At the pinned commit, **API-server shutdown** passes `shutdown_timeout`
(default 0) through the launcher to `MPClient.shutdown(timeout)` and its
process manager (`launcher.py:125-136`, `core_client.py:682-694`);
the manager's kill deadline begins after SIGTERM (`v1/utils.py:618-642`).
Separately, **background resource cleanup** passes
`VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS` to that manager
(`core_client.py:433-436`), while the executor uses the same value for its
first worker wait before a further 4 s escalation
(`multiproc_executor.py:444-459`). These are different entry paths: repairing
the latter's nested inequality would not repair zero grace on the former.
For C3, the
multi-process executor gives both `execute_model` and `sample_tokens` the same
RPC deadline (`multiproc_executor.py:321-342`); default-compatible async
scheduling queues both but waits on the latter future first
(`core.py:647-705`). This is a source explanation, not retroactive scoring.

## Architectural review result

The outline has a coherent top-down unit of work: separate lifecycle from
progress/health facts, join them for judgment (including intentional
pause/sleep), assign each fact an owner and freshness rule, and
define shutdown budgets so the outer kill cannot preempt the inner
escalation by construction. It correctly leaves numeric defaults and the
public meaning of `/health` open. Existing FT `UNHEALTHY` is acknowledged;
a new non-terminal state cannot simply reuse that name without reconciling
its current opt-in scope. The plan also avoids treating a missing observation
as proof of a missing process.

The unresolved design questions are **not** missing GPU evidence:

1. Should a non-terminal no-progress suspicion affect the existing `/health`,
   a separate readiness/progress surface, or initially only a metric? Each
   option needs demand, freshness, identity, lifecycle state, remote-KV wait,
   DP dummy-batch, and slow-step controls.
2. Is zero request-drain time meant to imply zero EngineCore/worker cleanup
   grace on CUDA? What budget owns each nested teardown stage?
3. Which existing lifecycle PR already owns a transition or timeout, and
   which proposed setting is now redundant? The inventory's PR statuses are
   dated snapshots and must be refreshed before upstream text.

## Stop/go for the next step

**GO now, CPU/read-only:** refresh source against current upstream `main`,
perform the full duplicate search for C1-on-CUDA, generic C2, C7 and C8,
then reduce the internal outline to a short upstream discussion with one
state model, one timeout-budget table, and explicit alternatives. This is
review material, not authorization to open a new issue or PR.

**Do not book GPU merely to repair K5.** A future GPU gate should be
preregistered only if it changes a concrete decision: for example, a
maintainer requires an accepted async-on/sync-off timeout assertion, or an
agreed progress-health implementation needs false-positive and DP/TP
controls. Such a run must record effective `async_scheduling`,
`max_concurrent_batches`, and saved `K5_RESULT` files. The prior unscored TP=2
arm remains unscored.

This review makes no new upstream claim and changes no experiment verdict.
The subsequent [current-main and duplicate-work check](ENGINE_LIVENESS_MAIN_AND_DUPLICATES_2026-09-27.md)
narrows the public exit: an earlier shutdown RFC and active overlapping PRs
must be credited, and the Python zero-grace result must not be generalized to
the Rust managed-engine path.
