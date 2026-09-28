# Engine liveness, health, and shutdown: internal RFC outline

Chinese companion: [中文版提纲](ENGINE_LIVENESS_RFC_OUTLINE_2026-09-27.zh-CN.md).

Status: **Lab design outline, not an upstream RFC or implementation proposal**.
The [unified evidence review](reviews/ENGINE_LIVENESS_UNIFIED_REVIEW_2026-09-27.md)
records why no further GPU run is needed for this outline and what would
justify one later.
The [current-main and duplicate-work check](reviews/ENGINE_LIVENESS_MAIN_AND_DUPLICATES_2026-09-27.md)
narrows its upstream exit; in particular, an earlier shutdown RFC already
exists, so this is not a first proposal of shutdown semantics.
The [targeted duplicate check](reviews/ENGINE_LIVENESS_DUPLICATE_CHECK_2026-09-27.md)
refreshes the source against `upstream/main` `55de40a2fc` and maps C1/C2/C7/C8
and the K5 attribution question to existing discussions. Its GitHub states
are a 2026-09-27 snapshot, not a promise of current status.
Source baseline: vLLM `c8602c79062440074a018c1d5f875a5571eb6881`;
the [contract inventory](ENGINE_LIVENESS_CONTRACT_INVENTORY_2026-09-27.md)
contains the source map and a 2026-09-27 `main` drift check. Issue and PR
states in that inventory are snapshots, not current-status claims. Before
posting upstream, refresh the target code and thread states again and review
this outline against the affected owners' existing work.

## 1. Question and intended outcome

What should an API server, EngineCore, executor, worker, and external health
probe be allowed to conclude when an engine stops making progress or is
shutting down? Today, the answer depends on which process owns the wait and
which executor is selected. The goal is a written cross-process contract for
**state, signal ownership, and nested time budgets**. It should let existing
fixes compose without adding unrelated timeout knobs or mistaking a missing
producer for a missing engine.

Any eventual upstream discussion of shutdown should continue [#24885](https://github.com/vllm-project/vllm/issues/24885),
which was closed as stale rather than resolved. For stall detection, this
outline should answer the scope question already raised in [#52365](https://github.com/vllm-project/vllm/pull/52365),
not introduce a fifth watchdog. A small implementation may follow agreement;
this outline does not request a new collector, automatic restart policy, or
wholesale replacement of vLLM's failure handling.

## 2. Evidence that motivates the contract

| Contract tension | Evidence and limit | Upstream-safe claim |
| --- | --- | --- |
| Request drain and process-kill grace share `shutdown_timeout` (C1); request-drain output handling is separate prior work (C8) | [K1 CPU check](reviews/ENGINE_LIVENESS_K1_K2_CPU_RUN_2026-09-27.md) exercised the real shutdown helper: zero grace sent SIGTERM then SIGKILL before a 100 ms handler could finish. [#36666](https://github.com/vllm-project/vllm/pull/36666) added a 5 s process-grace floor; [#40985](https://github.com/vllm-project/vllm/pull/40985) proposed removing it and was closed after a maintainer described it as intentional. [#43016](https://github.com/vllm-project/vllm/pull/43016) later removed the floor in a schema-CI change, and [#52281](https://github.com/vllm-project/vllm/pull/52281) restored separate zero-drain cleanup grace on ROCm only. On the checked `main`, `_shutdown_subprocesses` still has a 5 s floor while `shutdown` does not. No CUDA resource leak or client-visible abort result was measured. | Ask whether removing the minimum for CUDA was intended; treat this as a regression **question**, not proof the change was accidental. C8 is already raised in [#36964](https://github.com/vllm-project/vllm/pull/36964). |
| The same worker timeout bounds inner and outer teardown (C2) | K2 used the real inner function on a fake process tree. Tested `x=1,2,5` all cut off the inner escalation; a longer-grace control finished at `x+4`. [#43154](https://github.com/vllm-project/vllm/pull/43154) introduced the shared value, and [#55632](https://github.com/vllm-project/vllm/issues/55632) reported the same nested-budget class on ROCm. The general `x < x+4` conflict on the background-cleanup path is source-derived, not estimated from three trials. | Credit the prior report and specify a nesting invariant before adjusting another default; do not claim this class is new. |
| Worker-stall detection differs by executor (C3) | [K5](reviews/ENGINE_LIVENESS_K5_DUAL_GPU_RESULT_2026-09-27.md) scored TP=1: `/health` remained 200 during a 45 s injected hold and recovered. TP=2 changed 200→503 around the **experimentally shortened** 30 s deadline, with request 500 and a `sample_tokens` RPC timeout. The pinned default of `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` is 300 s. TP=2 is **formally unscored** because the frozen rule expected `execute_model`; the async wait order explains the method-name prediction error. | Present the measured contrast and the source path separately. Neither proves an organic issue's root cause, an indefinite observed hang, or an observed five-minute default run. |
| Health below EngineCore is unwired (C4) and a configured iteration timeout has no reader (C7) | Source inventory only; no new behavioral test. Open [#52365](https://github.com/vllm-project/vllm/pull/52365) proposes reading the C7 setting again for GPU-event waits. | Treat C4 as a contract question and C7 as context for existing work, not a new bug report. |

The most consequential shape is binary: a held TP=1 engine appeared healthy
for the measured window; the TP=2 deadline produced a terminal engine error.
The API returned 503 in K5 only because the experiment set
`VLLM_KEEP_ALIVE_ON_ENGINE_DEATH=1`; default launch behavior may instead close
the API process. Neither path expresses a non-terminal "alive but not
progressing" condition. The opt-in fault-tolerance framework has an
`UNHEALTHY` state with different scope and must be reconciled, not ignored.
The RPC-reply mismatch (C5, #58242/#58279) and shutdown-as-crash reporting
(C6, #48745/#49000) already have upstream owners; they are integration inputs,
not new findings claimed by this outline.

An incidental log in [#48745](https://github.com/vllm-project/vllm/issues/48745)
also shows `timeout=0s`, a process-manager force-kill branch, and EngineCore
starting resource teardown in the same logged second. This independently
corroborates that the zero-grace branch is reached in a reported shutdown, not
that SIGKILL interrupted cleanup: the one-second timestamps do not establish
ordering or completion. The reporter raised shutdown-as-crash logging, not C1;
this observation is not a scored C1 case.

## 3. Contract skeleton for review

### State and signal ownership

Keep **lifecycle** and **health/progress** as distinct axes, but evaluate them
together before judging a lack of progress. Candidate lifecycle transitions
include `starting → ready → draining → stopping → exited`, an independent
terminal failure transition, and intentional `ready ↔ paused` and
`ready ↔ sleeping` branches. These are candidate contract states, not a claim
that the current code implements a single enum. `pause_generation(mode="keep")`
freezes queued requests until resume, and `sleep` accepts the same pause
modes; admitted demand without token progress can therefore be healthy.
Candidate health facts are:

- process alive and contactable;
- EngineCore loop responsive;
- admitted work making progress within a declared observation window;
- progress unobserved because its producer is absent or stale;
- terminal failure confirmed by an owned error or process exit.

Names and public API representation are intentionally undecided. In
particular, a missing progress sample must not mean "engine missing" or
"engine healthy". A non-terminal suspect state needs a bounded observation
window, demand, freshness, process/engine identity, and lifecycle state. Long
prefill, compilation, graph capture, idle time, slow multimodal work,
intentional pause/sleep, requests in `WAITING_FOR_REMOTE_KVS`, and DP dummy
batches are required negative controls before it can drive `/health` or a
supervisor. A successful
loop ping is not token progress. Each transition needs one owner and an
explicit propagation path to the API, logs/metrics, and exit status.

### Timeout and shutdown budget

Define distinct budgets for request drain, EngineCore teardown, worker grace,
worker SIGTERM→SIGKILL escalation, and outer process-tree kill. The pinned
source has distinct Python entry paths, with the Rust manager wrapping the
headless Python path:

| Entry path | Outer process-tree grace today | Contract question |
| --- | --- | --- |
| Python API-server shutdown through `MPClient.shutdown(timeout)` | The launcher passes `shutdown_timeout` (default 0) to the manager. | Does zero request-drain time really mean zero EngineCore and worker teardown grace (C1)? |
| `vllm serve` parent in headless or multi-API-server mode | On signal-initiated shutdown, the parent calls `engine_manager.shutdown(timeout=shutdown_timeout)` in headless mode, or passes the remaining `shutdown_by` budget after stopping API servers in multi-API mode; default `shutdown_timeout` is 0. On other exits, `timeout` remains `None` and the helper's 5 s fallback applies. This path does not go through `MPClient`. | The same C1 question applies to signal-initiated shutdown. In multi-API mode, how much of the budget remains for EngineCore after API-server shutdown? |
| Background `MPClient` resource cleanup | `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS = x`, also used for the executor's first worker wait. | How can the outer deadline reserve the inner `x` wait **plus** its 4 s escalation **plus** EngineCore cleanup and delivery margin (C2)? |
| Rust managed-engine frontend | Rust starts a headless Python `serve` process and applies a minimum 5 s grace to that process group; when configured shutdown timeout is zero, it does not pass a positive `--shutdown-timeout` to Python. Inside, the headless parent still reaches `engine_manager.shutdown(timeout=0)`. | The 5 s bounds the outer group; it is not an EngineCore or worker teardown budget. Whether EngineCore shares the process group receiving Rust's SIGTERM was not checked. |

Fixing the C2 inequality alone does not fix the default API shutdown paths.
Conversely, fixing C1 only in `MPClient` or its launcher would miss headless,
multi-API, and the Python child of Rust-managed serving. Review the common
EngineCore manager shutdown boundary and platform-specific exceptions before
choosing a value; this outline does not choose new numeric defaults.

Specify what happens to outstanding requests at the drain deadline (abort
acknowledgment versus process death), and how intentional shutdown differs
from a crash in frontend status and exit code. A timeout must distinguish the
**future the caller waited on** from the **operation the worker was actually
running**, and name the affected rank where known, stage, owner, and whether
it means **suspected stall** or **terminal failure**. In K5 the worker was
held in `execute_model`, but the timeout named `sample_tokens`: the async
EngineCore waits on that future first, and the current message omits rank.
This is a diagnosability requirement illustrated by K5 and relevant to
[#54638](https://github.com/vllm-project/vllm/pull/54638), not a separate
proposal or assertion of a new root cause. Async scheduling changes which RPC
future is observed first; future tests must record its effective value and
concurrent-batch count.

### Health surface and compatibility

Decide whether progress-aware health belongs in an opt-in `/health` mode, a
separate readiness/progress surface, or only a metric initially. Preserve the
current default endpoint contract until false-positive controls and operator
semantics are agreed. Do not make a new controller/restart policy part of this
RFC. Keep the existing opt-in FT `UNHEALTHY` vocabulary in the comparison so
two subsystems do not introduce contradictory meanings for the same state.

## 4. Relationship to existing work

This is an integration outline, not a claim that the Lab should replace
other contributors' patches. The shutdown half continues [#24885](https://github.com/vllm-project/vllm/issues/24885),
closed as stale with its contract questions unresolved. The stall half should
help resolve the unanswered "whole EngineCore iteration or only the async
output wait?" question in [#52365](https://github.com/vllm-project/vllm/pull/52365),
not add another watchdog. Map at least these threads before implementation:

- [#54638](https://github.com/vllm-project/vllm/pull/54638),
  [#52365](https://github.com/vllm-project/vllm/pull/52365),
  [#56816](https://github.com/vllm-project/vllm/issues/56816) /
  [#55700](https://github.com/vllm-project/vllm/pull/55700), and merged
  [#58779](https://github.com/vllm-project/vllm/pull/58779): worker watchdog,
  bounded GPU wait, diagnostic stack-dump watchdog, and draft-token RPC
  timeout respectively. Define each observer's scope and whether its result
  is suspect, diagnostic-only, or terminal; K5's RPC attribution belongs in
  the #54638 context;
- [#54553](https://github.com/vllm-project/vllm/pull/54553),
  [#43154](https://github.com/vllm-project/vllm/pull/43154), and
  [#55632](https://github.com/vllm-project/vllm/issues/55632), with the
  ROCm/XPU grace fixes: inner/outer shutdown budget ownership. Credit the
  existing nested-timeout report and its invariant-test direction;
- [#58279](https://github.com/vllm-project/vllm/pull/58279): RPC reply
  correspondence and any proposed new deadline;
- [#49000](https://github.com/vllm-project/vllm/pull/49000) and
  [#52178](https://github.com/vllm-project/vllm/pull/52178): intentional
  shutdown versus fatal cause propagation;
- [#36258](https://github.com/vllm-project/vllm/pull/36258): `/live` versus
  `/health` liveness/readiness separation, following #24885; do not propose
  that distinction as new;
- [#31252](https://github.com/vllm-project/vllm/issues/31252): earlier
  report that an API parent can interrupt EngineCore cleanup;
- [#36964](https://github.com/vllm-project/vllm/pull/36964): open work on
  aborting requests and draining outputs at shutdown; C8 must not be pitched
  as an unowned output-abort fix;
- C7's unread `VLLM_ENGINE_ITERATION_TIMEOUT_S` is background for #52365;
  C8's drain/abort path has an owner in #36964. Neither is an unowned finding;
- [#36451](https://github.com/vllm-project/vllm/pull/36451): loop ping versus
  actual request progress; the Lab's evidence is already in a comment there,
  so an upstream draft should link it without re-mentioning reviewers;
- the opt-in FT framework: scope and semantics of `UNHEALTHY`.

The inventory records historical PR states. An upstream version must refresh
them and avoid assigning work to a PR that has merged, changed, or been
superseded.

## 5. Validation and staged implementation questions

1. **Contract-only review:** agree on state vocabulary, owner for each signal,
   `/health` compatibility, and the timeout inequality. Record alternatives
   and why each was rejected. Do not open a broad code PR first.
2. **CPU invariants:** retain K1/K2 as narrow controls; add assertions for
   zero drain with nonzero teardown grace, nested deadline ordering, and
   intentional-shutdown versus crash propagation. A test should fail if its
   decisive signal is removed.
3. **GPU behavior only where required:** K5 is a mechanism example, not a
   two-arm accepted verdict. If repeated for another reason, preregister
   async-on → `sample_tokens` and async-off → `execute_model`, record both
   effective configuration values, and preserve separate scores. Do **not**
   book GPUs merely to repair this document.
4. **Non-terminal health controls:** test admitted demand plus no progress
   against long prefill, first-request compilation, idle, long generation,
   intentional pause/sleep, `WAITING_FOR_REMOTE_KVS`, DP dummy batches,
   and multi-engine partitioning. Measure detection latency and false
   positives before attaching a health status or restart action.
5. **Small upstream slices:** after maintainer direction, implement one
   agreed seam at a time with its regression tests and backward-compatibility
   note. Do not add another parallel PR while overlapping lifecycle work is
   awaiting review.

## 6. Admission gate and upstream exit

This Lab outline is complete when each proposed transition has an owner,
observable input, timeout rule, and falsifiable test plan. The targeted
duplicate check for C1-on-CUDA, generic C2, C7, C8, and K5 attribution is
recorded in the linked review at `55de40a2fc`; refresh the target source and
thread states again immediately before any upstream post. Separate scored K5
facts from its source interpretation, and ask whether an existing PR already
provides the intended contract. An upstream draft should
be short: one state diagram, one timeout-budget table, two concrete examples,
and a few binary design questions. It should link the Lab's deeper evidence,
not paste this entire inventory into a new issue. The default upstream exit
is a scoped contribution to an active owner's discussion, **not** a new issue,
fifth mechanism, or assumed new RFC. C1 is the only candidate new upstream
question; a comment in an existing shutdown thread would be its lowest-cost
venue, subject to the standing decision to advance existing PRs first.
