# Engine liveness: duplicate check and `main` refresh

Status: **read-only check, no upstream action taken**. This fulfils the gate in
the [RFC outline](../ENGINE_LIVENESS_RFC_OUTLINE_2026-09-27.md) §6 and the
[unified review](ENGINE_LIVENESS_UNIFIED_REVIEW_2026-09-27.md) "GO now" step.
Source refreshed against `upstream/main` `55de40a2fc` (2026-09-28 UTC), which is
2,589 commits after the pin `c8602c7`. GitHub states are a 2026-09-27 snapshot.
Searches used `gh search issues --include-prs` over open and closed items for
the env names, helper names, error strings and area keywords listed per row;
multi-word keyword searches without an exact identifier returned nothing and are
not evidence of absence.

## Summary

| Item | Still on `main`? | Filed upstream? | Disposition |
| --- | --- | --- | --- |
| C1 zero process grace (CUDA) | yes | **no**, but it reverses a stated maintainer intent (below) | Strongest new item. Frame as a regression question, not a new idea. |
| C2 nested worker budget | yes | same class filed for ROCm (#55632, open; fix #55646 closed unreviewed) | Not a duplicate, but must cite #55632 and #43154. |
| C7 iteration timeout without reader | yes (env defined, no reader) | no, but open PR #52365 proposes to reuse it | Do not file separately; it is context for #52365. |
| C8 drain enforced by SIGKILL | yes | acknowledged by njhill on #36666; open PR #36964 | Duplicate as a finding; cite, do not re-report. |
| RPC timeout names waited-on method, not stalled rank/method | yes (`multiproc_executor.py:437`) | covered in substance by open PR #54638 | Drop as a separate proposal; K5 is a supporting example for #54638. |

## C1: the 5 s floor was intentional, then removed by an unrelated PR

- #36666 (markmc, merged 2026-03-13) added `timeout = max(timeout, 5.0)` to
  `vllm.v1.utils.shutdown` with the comment "Allow at least 5 seconds".
- #40985 (2026-04-27) proposed removing that floor for `timeout=0`. markmc
  declined: the minimum is "pretty crude, but it is intentional", deliberately
  giving children time to finish shutdown, possibly so abort responses can
  propagate (cc njhill). The PR was closed.
- #43016 (2026-05-24, titled "[ROCm][CI] Stabilize 400 error return code for
  invalid schema inputs") removed the floor inside a schema-validation PR. Its
  stated reason in a PR comment was that clamping explicit `0`/`2` to `5`
  "made the tests less deterministic". No linked discussion of #40985.
- #52281 (2026-08-19) then re-added a 15 s grace for the zero/zero path **on
  ROCm only**, after CI showed 11.72 GB VRAM resident following a zero-grace kill.
  Its docstring (`v1/engine/utils.py:46-66` on `main`) states the general need:
  "The parent process manager still needs a separate window in which the
  EngineCore can release device resources before it is force-killed."
- On `main`, `_shutdown_subprocesses` (the subprocess-wrapper path,
  `v1/utils.py:479-485`) **still** clamps to `max(timeout, 5.0)`, while
  `shutdown` (`:615-650`) does not. Two helpers documented as mirrors disagree.

No issue or PR references the #43016 floor removal. Upstream-safe framing: "Was
removing the 5 s minimum for `timeout=0` in #43016 intended for non-ROCm
platforms, given #40985 and the #52281 rationale?" K1 is the CPU evidence;
still no CUDA leak claim.

## C2: nested budgets, same class already reported on ROCm

- #43154 (rjrock, merged 2026-06-12) introduced
  `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS` and passed it to **both** the executor
  worker wait and the engine-manager shutdown. That is the origin of the shared
  value.
- #55632 (fululi12, open; the author commented "Close issue" on 09-22 without
  closing it) reports the ROCm mirror image: the outer 15 s EngineCore grace is
  defeated by the inner 5+4 = 9 s worker-kill schedule. Fix #55646 derived the
  inner grace from the outer constant with an invariant test
  (`grace + WORKER_SIGTERM_GRACE_S >= ROCM_ENGINE_PROCESS_SHUTDOWN_TIMEOUT_S`)
  and was closed by its author on 09-17 with no human review.
- The Lab's C2 is the other direction on the background-cleanup path
  (`core_client.py:494` on `main`): outer `x` < inner `x + 4`.

Both are "two nested timers not derived from each other". The outline's
timeout-budget table should cite #55632 as prior evidence and credit its
invariant-test shape rather than present C2 as a new discovery.

## C3 / progress health: four overlapping proposals, no agreed boundary

| Thread | Scope | State |
| --- | --- | --- |
| #54638 worker-side watchdog | wedged kernel; names `execute_model on worker rank N`; notes the RPC timeout only observes the output rank | open since 08-31, no human review |
| #52365 bounded GPU event waits | async output waits; reuses `VLLM_ENGINE_ITERATION_TIMEOUT_S` | open; njhill (08-15) against polling; author's question "whole EngineCore iteration or only the async output wait?" (08-21) unanswered |
| #56816 RFC / #55700 PR | no-progress heartbeat → all-thread stack dump, engine + workers; diagnostic only | open; wangxiyuan asked for the RFC; ZJY0516 asked for config and a counter (added 09-28) |
| #58779 | draft-token RPC bounded by execute-model timeout | merged 09-26 |
| #36258 `/live` probe | liveness (200 during drain, 503 on fatal) vs `/health` readiness; follow-up to #24885 | open since 03-06, markmc "will try to review", no review since |

This is the strongest argument for the contract outline: several contributors
are each adding a watchdog or deadline with a different owner and meaning, and
the scope question in #52365 is exactly the outline's "which owner observes
which progress" question. The outline must position itself as answering that
question for these threads, not as a fifth mechanism.

## Shutdown semantics: #24885 is the parent RFC

markmc's [RFC] Clarifying vLLM Shutdown Semantics (#24885) defines parent-owned
quiescence, bounded drain, and child processes not making independent
decisions. Its open questions include "should the 'I'm shutting down' status be
observable via /health?". It was **closed as stale on 2026-09-21**, not
resolved. Open follow-ups: #36258 (`/live`), #36964 (abort in-flight and drain
outputs on shutdown, njhill's #36666 notes; no review since March), #42753 (KV
transfer during shutdown), #43208 (k8s drain docs). #50529 (ignore signals
during EngineCore cleanup) and #54553 (bounded fatal shutdown) touch the same
budget.

Any upstream text on the shutdown half should be framed as continuing #24885
(and could ask whether to reopen it), not as a new RFC.

## Other threads seen, not claimed

#54184 (liveness monitor SIGTERMs a healthy EngineCore on spurious sentinel
readiness), #57429 (ROCm engine stops progressing and dies with no error),
#58279 / #58242 (C5, now titled "Improve engine rpc failure handling and
shutdown"), #48745 / #49000 (C6), #52314 (log force-killed subprocesses).
Organic `RPC call to sample_tokens timed out` reports are numerous (for example
#41530, #40926, #52504); they show the message is what operators see, but none
is claimed as explained by K5.

## Consequences for the outline

1. C1 becomes a provenance question anchored on #36666/#40985/#43016/#52281,
   with the `_shutdown_subprocesses` inconsistency as a second concrete example.
2. C2 cites #43154 and #55632; drop any novelty claim.
3. C7 and C8 leave the "new findings" list; cite #52365 and #36964.
4. The RPC-attribution point moves into #54638's context.
5. §4 "Relationship to existing work" adds #24885, #36258, #36964, #56816/#55700,
   #55632 and #43154, and removes nothing.
6. No new issue is justified yet. The only candidate new upstream question is C1,
   and the cheapest venue is a comment on the thread its owners already read
   (for example #52281 or #36964), subject to the standing "push existing PRs
   first" decision.
