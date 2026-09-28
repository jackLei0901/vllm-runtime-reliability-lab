# vLLM engine liveness and failure-signaling contract — source inventory

Status: **internal design inventory, 2026-09-27**. It is a source map with
separately linked K1/K2/K5 run records, not itself an experiment or upstream
text. No conflict below is promoted to a user-visible bug solely by this
inventory. This is the top-down input the Lab asked for
after pausing the TP in-flight line: one system model against which scattered
issues and PRs can be placed, weighted and tested.

## Pins and scope

- Primary pin: vLLM `c8602c79062440074a018c1d5f875a5571eb6881` (committed
  2026-08-03), the revision used by the Lab's recent serving runs. Line numbers
  below refer to this commit unless marked **main**.
- Drift check: upstream `main` at `2b9b55c7f1` (2026-09-27). Each finding says
  whether it still holds there.
- In scope: the online `vllm serve` path (API server → `AsyncLLM` → `MPClient`
  → `EngineCoreProc` → executor → workers), its liveness signals, failure
  propagation, timeouts and shutdown. Out of scope here: Ray executor details,
  multi-node DP supervision beyond one note, disaggregated/KV-connector
  timeouts, and model-specific waits.
- Relation to the [fault taxonomy](FAULT_TAXONOMY_V0_2026-09-24.md): V1 (work
  stops behind a responsive endpoint) is a *progress* failure; V2/V3 are
  *propagation* failures. This inventory supplies the mechanism map both
  categories sit on.

## 1. Process topology and who holds liveness state

| Process | Liveness state it holds | Observed by | Source |
| --- | --- | --- | --- |
| API server (uvicorn + `AsyncLLM`) | `engine_dead` flag (via `MPClient.resources`), `output_handler` task state | `/health`, launcher watchdog | `core_client.py:426`, `async_llm.py:1086-1096` |
| `MPClient` monitor thread | EngineCore process sentinels | itself; sets `engine_dead`, calls `shutdown()` | `core_client.py:708-733`, `v1/engine/utils.py:222-239` |
| `EngineCoreProc` | `shutdown_state` (RUNNING/REQUESTED/SHUTTING_DOWN); optional FT sentinel status | its own busy loop; client via `ENGINE_CORE_DEAD` | `core.py:1378-1389`, `core.py:1460-1504` |
| `MultiprocExecutor` monitor thread | worker process sentinels, `is_failed` | itself; failure callback into EngineCore | `multiproc_executor.py:286-305` |
| Workers | `shutdown_requested` event (death pipe, SIGTERM) | themselves | `multiproc_executor.py:795-841` |
| Optional: FT `EngineCoreSentinel` | `EngineStatusType` HEALTHY/UNHEALTHY/DEAD, recovery wait | FT utility calls | `fault_tolerance/engine_core_sentinel.py:173-194`, `v1/engine/__init__.py:296-299` |

There are two unrelated state vocabularies: a global `engine_dead` boolean on
the default path, and a three-state `EngineStatusType` used only by the opt-in
fault-tolerance framework (`enable_fault_tolerance=False` by default,
`config/parallel.py:398`; scoped to DP+EP external-LB deployments by #44428,
merged by tlrmchlsmth). Neither has a "degraded / no progress" state.

## 2. Liveness and failure signals

Trust status: **wired** = reaches `/health` or process exit; **log-only** =
visible only in logs; **unwired** = implemented but never called in serving;
**absent** = no producer.

| # | Condition | Signal path | Detection latency | Status |
| --- | --- | --- | --- | --- |
| S1 | EngineCore process exits | sentinel wait (1 s poll) → `engine_dead=True` → `MPClient.shutdown()` → `/health` 503 | ≈1 s | wired |
| S2 | Exception escapes EngineCore busy loop | `_send_engine_dead()` puts `ENGINE_CORE_DEAD`, output thread joined ≤5 s → client sets `engine_dead` | immediate | wired (`core.py:1347-1352`, `1605-1616`; client `core_client.py:491-492`) |
| S3 | Worker process exits | executor monitor → `is_failed`, executor shutdown, failure callback → `EXECUTOR_FAILED` → `RuntimeError` in EngineCore → S2 | immediate | wired (`multiproc_executor.py:286-305`, `core.py:1029-1031`, `1534-1535`) |
| S4 | Worker alive but stuck (TP/PP>1, multiproc executor) | `execute_model`/`sample_tokens` RPC deadline → `TimeoutError("RPC call to … timed out")` → S2 | **300 s** (`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`) | wired, slow |
| S5 | Worker/GPU stuck, world size 1 (uniproc executor) | none: uniproc futures reject timeouts (`uniproc_executor.py:32-34`) and `execute_model` passes none | **unbounded** | absent |
| S6 | `AsyncLLM` output handler dies | `errored` becomes true → `/health` 503 | immediate | wired |
| S7 | Executor-level health RPC | `MultiprocExecutor.check_health()` = `collective_rpc("check_health", timeout=10)`; worker implementation is a no-op | n/a | **unwired**: only tests call it (`multiproc_executor.py:505-507`, `gpu_worker.py:1181-1183`) |
| S8 | Shared-memory ring buffer wait | warning every 60 s (`VLLM_RINGBUFFER_WARNING_INTERVAL`) | 60 s | log-only (`shm_broadcast.py:680`, `756`) |
| S9 | Throughput collapses to 0 with requests queued | periodic stats line | stats interval | log-only (what #39863 users read) |
| S10 | Engine alive, no token progress | — | — | **absent** (the Lab's V1; #36451 proposes an EngineCore ping, which proves loop responsiveness, not progress) |
| S11 | Engine errored → API process exit | launcher watchdog every 5 s; exits unless `VLLM_KEEP_ALIVE_ON_ENGINE_DEATH` | ≤5 s | wired (`launcher.py:168-190`) |

`/health` is exactly `AsyncLLM.check_health()`: raise if `engine_dead` or the
output handler is done (`async_llm.py:920-923`, `serve/instrumentator/health.py:22-33`).
It performs no RPC. It therefore detects S1–S3 and S6, detects S4 only after
300 s, and cannot detect S5 or S10. **main:** unchanged; `check_health` still
has no non-test caller below `AsyncLLM`.

## 3. Timeout and grace inventory

| # | Name / literal | Default | Owner → what expires | Source |
| --- | --- | --- | --- | --- |
| T1 | `VLLM_ENGINE_READY_TIMEOUT_S` | 600 s | client waiting for EngineCore startup | `core_client.py:656`, `1666` |
| T2 | `HANDSHAKE_TIMEOUT_MINS` | 5 min | EngineCore ↔ client handshake poll | `core.py:98`, `1253` |
| T3 | DP coordinator ZMQ address wait | 120 s (literal) | coordinator startup | `coordinator.py:61` |
| T4 | `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | 300 s | per-step RPC deadline, multiproc only | `multiproc_executor.py:329`, `341` |
| T5 | executor `check_health` | 10 s (literal) | unused (S7) | `multiproc_executor.py:506` |
| T6 | `VLLM_ENGINE_ITERATION_TIMEOUT_S` | 60 s | **nothing** — no reader in `vllm/` | see C7 |
| T7 | `shutdown_timeout` (`--shutdown-timeout`) | **0** | documented as in-flight request drain; also used as the EngineCore process-kill grace | `config/vllm.py:425-429`, `launcher.py:125-136` |
| T8 | `shutdown(procs, timeout=None)` fallback | 5 s (literal) | process-manager SIGTERM→SIGKILL window when no timeout given | `v1/utils.py:590-642` |
| T9 | `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS` | 5 s | **both** the client's EngineCore-process grace (`core_client.py:433-436`) **and** the executor's first worker wait (`multiproc_executor.py:444-446`) | see C2 |
| T10 | executor SIGTERM wait | 4 s (literal) | worker SIGTERM → SIGKILL | `multiproc_executor.py:459` |
| T11 | `_send_engine_dead` output-thread join | 5 s (literal) | delivery of `ENGINE_CORE_DEAD` | `core.py:1612` |
| T12 | launcher watchdog period | 5 s (literal) | API exit after engine error | `launcher.py:174` |
| T13 | ring-buffer warning interval | 60 s | log only | `envs.py:715` |
| T14 | FT `engine_recovery_timeout_sec` | 120 s | opt-in FT recovery wait | `config/fault_tolerance.py:12` |

No rule states how T4, T7, T8, T9, T10 and T11 nest. Open PRs in the same
area each add another knob without one (§6).

## 4. Transitions and where the contract breaks

| Transition | Who decides | Signal | Gap |
| --- | --- | --- | --- |
| starting → ready | client (T1/T2) | handshake | — (not examined further here) |
| ready → dead (process exit, fatal exception, worker exit) | monitors, S1–S3 | `engine_dead`, 503, API exit (S11) | propagation to process exit status: taxonomy V2/#52178 |
| ready → stuck (GPU/worker hang) | multiproc: T4 after 300 s; uniproc: nobody | S4 / none | **C3** |
| ready → no progress (loop alive) | nobody | none | S10 absent (taxonomy V1) |
| ready → draining/aborting (SIGTERM) | API server → client → process manager | logs only | **C1, C2, C8** |
| shutting down → exited | nested process managers | exit codes | **C2**; shutdown-vs-crash ambiguity **C6** |

## 5. Contract findings

Confidence: **source** = follows directly from cited lines; **hypothesis** =
consequence not yet observed; each has a cheapest check.

**C1 — The request-drain timeout is also the process-kill grace; its default
of 0 means immediate SIGKILL.** `launcher.py:125-136` passes
`vllm_config.shutdown_timeout` (default 0, documented only as an in-flight
request grace) through `AsyncLLM.shutdown` → `MPClient.shutdown` →
`CoreEngineProcManager.shutdown` → `shutdown(procs, timeout)`
(`core_client.py:682-693`, `v1/engine/utils.py:216-220`). With `timeout=0`,
`shutdown()` sends SIGTERM, computes an already-expired deadline, skips every
join, and calls `kill_process_tree` — SIGKILL to EngineCore and all
descendants (`v1/utils.py:618-642`, `utils/system_utils.py:256-279`). EngineCore's
abort handling (`core.py:1474-1483`) and the executor's worker teardown
(`multiproc_executor.py:419-470`) are then unreachable. Platform patches show
upstream has hit this twice: XPU raises 0 to 5 because oneCCL/Level Zero
otherwise hang on the next start (`platforms/xpu.py:352-361`, #46433), and
**main** adds a ROCm-only 15 s grace when both timeouts are 0
(`get_engine_process_shutdown_timeout`, main `v1/engine/utils.py:43-65`, #52281),
whose docstring states the separation this inventory calls for. **CUDA keeps
zero grace on main.** Confidence: source. Impact on CUDA (leaked shared-memory
segments, skipped communicator teardown): hypothesis; #57303 is a possibly
related shared-memory leak on non-graceful shutdown.

**C2 — The same variable bounds two nested levels, so the inner escalation can
never fit inside the outer one.** When no drain timeout is given, the client
kills the EngineCore tree after `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS`
(`core_client.py:433-436`; or T8's 5 s). Inside EngineCore, the executor waits
the *same* value for workers, then SIGTERM, then 4 s, then SIGKILL
(`multiproc_executor.py:444-468`). For every setting *x*, the inner schedule
needs *x* + 4 s plus EngineCore's own teardown, while the outer allows *x*.
The full inner escalation therefore has no reserved time inside the outer
deadline; whether its first SIGTERM step happens just before the outer kill
is a scheduling race, not an invariant. #55632 reported the ROCm instance
(inner 9 s vs the new outer 15 s) and closed without a merged fix (#55646
closed). Confidence:
source. **main:** unchanged (main `multiproc_executor.py:479`, `493`,
`core_client.py:494`).

**C3 — Whether a stuck step is ever detected depends on executor type.**
Multiproc (world size > 1) turns a stuck worker into `EngineDeadError` after
300 s (S4); uniproc (world size 1) has no deadline (S5), so the engine can
wedge indefinitely with `/health` 200. This splits one failure class into two
different user-visible symptoms. It is consistent with the issue cluster: the
TP=1 reports describe indefinite wedges with green health (#37729, #58422),
while the TP=8 report dies after about five minutes with "RPC call to
sample_tokens timed out" (#41530). Per-issue root causes are not established
here. EngineCore-side waits add more unbounded paths (#52247:
`AsyncModelRunnerOutput` event sync). Confidence: source for the mechanism;
hypothesis for each issue's mapping. The separate [K5 run record](reviews/ENGINE_LIVENESS_K5_DUAL_GPU_RESULT_2026-09-27.md)
scores the TP=1 injected-stall behavior. Its TP=2 health transition and
`sample_tokens` RPC timeout are source-consistent, but formally unscored
because the frozen scorer expected an `execute_model` timeout. **main:**
unchanged.

The K5 comparison also shows the shape of this contract, within its injected
stall: TP=1 remained `/health`-green for the measured 45-second hold and
recovered on release; an *indefinite* green hang follows from the missing
uniproc deadline in source, not from a 45-second run. TP=2 went from 200 to
503 at the declared deadline, with a request 500 and `EngineDeadError` after
the RPC timeout. The experiment pinned
`VLLM_KEEP_ALIVE_ON_ENGINE_DEATH=1` to retain the API server; without that
setting the launcher's watchdog may exit it instead. This is a binary
healthy/terminal-failure path, not a non-terminal progress-degraded state.
The FT framework's `UNHEALTHY` remains opt-in and separately scoped, as §1
records. The TP=2 method-name mismatch prevents strict K5 acceptance but does
not erase the observed transition or the source-derived timeout chain.

**C4 — Health checks exist below EngineCore but are not wired.** S7 is
implemented at executor and worker level and is never called in serving; the
worker check is a no-op. `/health` reports only frontend flags. Confidence:
source. **main:** unchanged.

**C5 — RPC reply correspondence after a worker error.** `collective_rpc`
raises on the first non-success reply and leaves later workers' replies
unread; the next call can consume them (`multiproc_executor.py:397-411`).
Reported as #58242, PR #58279. Confidence: source. **main:** unchanged.

**C6 — Intentional shutdown is indistinguishable from a crash in the output
handler.** `AsyncLLM.shutdown` tears down the engine client before cancelling
`output_handler` (`async_llm.py:262-274`), so the handler observes the
cleanup's `engine_dead` state as a failure. Reported as #48745; PR #49000
waiting since 2026-07-18. Confidence: source plus independent reproduction in
the issue. **main:** unchanged.

**C7 — `VLLM_ENGINE_ITERATION_TIMEOUT_S` has no reader.** Declared with a
60 s default and the comment "timeout for each iteration in the engine"
(`envs.py:26`, `780-783`), but no code under `vllm/` reads it; the last reader
(Voxtral realtime) was removed in `8974ed89cd` on 2026-07-05. Nine
`.buildkite/performance-benchmarks` configs and three tests still set it to
120–600 s. Operators can believe a per-iteration timeout exists when none
does. Confidence: source. **main:** still no reader.

**C8 — The drain contract is enforced by killing the engine, not by aborting
requests.** The config promises that remaining requests "are aborted once the
timeout is reached" (`config/vllm.py:425-429`). EngineCore's drain loop has no
deadline (`core.py:1484-1504`); the only enforcement is the parent's
SIGKILL at the same deadline, which **main**'s docstring states explicitly
("EngineCore relies on that deadline to enforce request draining"). Drain and
teardown therefore share one budget, and requests still running at the
deadline end through engine death rather than `FINISHED_ABORTED`.
Confidence: source for the mechanism; client-visible outcome is a hypothesis.

## 6. Existing issues and PRs mapped to the model

| Item | Cell | State |
| --- | --- | --- |
| #52178 (Lab owner's PR) | ready→dead propagation to process exit (taxonomy V2) | open, green, awaiting njhill |
| #58125 (Lab owner's PR) | shutdown test synchronization | open, label gate |
| #36451 | S10 (loop responsiveness, not progress) | open; Lab evidence posted |
| #53883 | S10 producer for backpressure stalls (taxonomy V1) | open; Lab R3 evidence posted |
| #58242 / PR #58279 | C5, plus a new RPC deadline knob | open, njhill requested, no human review |
| #48745 / PR #49000 | C6 | open since 07-18, njhill requested |
| #55632 / #55646 | C2 (ROCm instance) | issue closed, PR closed unmerged |
| #52281 (merged), #46433 (merged) | C1 platform patches (ROCm, XPU) | merged; CUDA untreated |
| PR #54638 worker kernel watchdog | C3 / S5, adds a knob | open since 08-31, 4 reviewers requested, no human review |
| PR #54553 bounded fatal shutdown | C2 fatal path, adds a knob | open since 08-31, njhill requested |
| #52247 / PR #52365 bounded async event waits | C3 (EngineCore-side unbounded wait) | open since 08-14 |
| PR #58738 multi-node MP peer failure exit | S3 across nodes | open since 09-25, njhill requested |
| #37729, #58422, #39863, #41530, #45094, #57429 | C3 / S10 symptom cluster | open; mostly hardware the Lab lacks |
| #57303 | possible C1 consequence | open; unverified link |
| FT framework #44428 | alternative state vocabulary (§1) | merged; opt-in |

At least six open PRs from different authors patch this contract piecemeal,
most waiting on the same CODEOWNER, and three of them introduce new timeout
variables with no nesting rule. That, more than any single bug, is the case for
a written contract.

## 7. Stop-rule evaluation

The rule set before starting: if the inventory predicts nothing beyond
known issues, it stays an internal Lab doc; with at least two unreported,
checkable conflicts, it is RFC-grade.

Candidates with no report found in a quick search of issues and PRs:
C1 on CUDA (only platform-specific patches exist), C2 in its generic form
(only the ROCm instance was reported), C7, and C8. A full duplicate check is
required before any of these appears in upstream text. **Provisional reading:
the RFC-grade threshold is met, subject to the checks below.**

## 8. Cheapest checks, in order

| Check | Covers | Cost | Pass condition |
| --- | --- | --- | --- |
| K1: CPU unit — `shutdown([proc], timeout=0)` against a child whose SIGTERM handler writes a marker after 100 ms | C1 | minutes, no GPU | marker never written; child killed |
| K2: CPU unit — fake EngineCore parent with fake workers using the real `_ensure_worker_termination`, parent deadline *x* | C2 | < 1 h, no GPU | ready workers and inner teardown start are witnessed; the inner schedule does not complete at each **tested** *x* while a longer-grace control does. The general budget inequality is argued from source, not inferred from finite samples. |
| K3: repo grep (done above) plus `git log -S` | C7 | done | no reader on pin and on main |
| K5: 2×4090 — inject one bounded worker stall at TP=1 and TP=2 (reuse the Lab's stall-plugin pattern) and poll `/health`; preregister a shorter T4, e.g. `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30`, for the TP=2 arm | C3 | one bounded two-GPU booking | TP=1: 200 throughout the hold; TP=2: 503 only after the declared T4 deadline |
| K4 (optional): 1×4090 — `vllm serve` small model, SIGTERM with default settings; diff `/dev/shm` and record which `[shutdown]` log lines appear | C1 impact, C8 | one short session | executor teardown lines absent; any leftover segments recorded. This is downstream corroboration, not a gate for discussing C1/C2's timeout design. |

K1 and K2 have since passed the named CPU checks against an exact Python
source snapshot of the pinned commit, with K2 using a disclosed `fork` startup
mode on a memory-limited host. See the [run record](reviews/ENGINE_LIVENESS_K1_K2_CPU_RUN_2026-09-27.md)
for the controls, hashes, failed default-startup attempts, and limits. This
does not satisfy K4/K5 or demonstrate a user-visible failure.
K5 was preregistered in the [bounded two-GPU protocol](../experiments/engine-liveness-contract/K5_PROTOCOL_2026-09-27.md).
The [run record](reviews/ENGINE_LIVENESS_K5_DUAL_GPU_RESULT_2026-09-27.md)
separates the original unscored startup attempt from the environment-corrected
round. The strict two-arm K5 acceptance condition was not met.

K1/K2 support a design-level question about separating request drain from
process-kill grace and nesting the worker budget; they do **not** establish a
CUDA leak. The platform-specific XPU and ROCm patches in §5 make this question
worth asking without treating K4 as a prerequisite: is zero process grace on
CUDA intentional when those platforms needed explicit grace? K5 supplied a
bounded executor comparison, with the scoring limit described above. Before
any upstream RFC text, perform a full duplicate
check for C1-on-CUDA, generic C2, C7, and C8; the quick search in §7 is not
that check. The [Lab-internal RFC outline](ENGINE_LIVENESS_RFC_OUTLINE_2026-09-27.md)
now puts a two-axis state model, explicit timeout nesting, health-surface
compatibility, and the related PRs into one reviewable contract. It is not
an upstream RFC. The natural reviewers are njhill (CODEOWNER of
`v1/engine`) and tlrmchlsmth (merged the FT framework). No upstream action is
taken by this document.
