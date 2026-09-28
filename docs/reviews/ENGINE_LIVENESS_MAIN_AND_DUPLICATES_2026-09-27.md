# Engine liveness: current-main source refresh and duplicate-work check

Scope: read-only review on 2026-09-27. The upstream `main` commit read here is
[`73859fec5865700c5b2021b22890cb3b2005f66e`](https://github.com/vllm-project/vllm/commit/73859fec5865700c5b2021b22890cb3b2005f66e).
The K1/K2/K5 experiments still belong to their original pinned build
`c8602c79062440074a018c1d5f875a5571eb6881`; reading new source does not
turn them into current-main runs. Issue and PR states below are a dated
snapshot, not a promise about their eventual disposition. No upstream post
or code change is authorized by this review.

## Source refresh

| Question | Current-main observation | Limit |
| --- | --- | --- |
| C1/C8, Python API shutdown | The Python `MPClient.shutdown(timeout)` forwards its argument to the EngineCore manager ([source](https://github.com/vllm-project/vllm/blob/73859fec5865700c5b2021b22890cb3b2005f66e/vllm/v1/engine/core_client.py#L702-L713)). The CLI supervisor also forwards the remaining `shutdown_timeout` budget to local engines ([source](https://github.com/vllm-project/vllm/blob/73859fec5865700c5b2021b22890cb3b2005f66e/vllm/entrypoints/cli/serve.py#L368-L381)). The process manager treats an explicit zero as no grace after SIGTERM, then force-kills survivors ([source](https://github.com/vllm-project/vllm/blob/73859fec5865700c5b2021b22890cb3b2005f66e/vllm/v1/utils.py#L567-L618)). | This is a Python process-manager path, not all possible frontends. K1 tests the helper with a fake child, not CUDA teardown or a full API server. |
| Rust managed-engine exception | The Rust managed-engine parent imposes a minimum 5 s shutdown timeout ([source](https://github.com/vllm-project/vllm/blob/73859fec5865700c5b2021b22890cb3b2005f66e/rust/src/managed-engine/src/process.rs#L113-L145)). | It does not establish that 5 s covers all nested worker teardown; it prevents a blanket “zero grace on every frontend” claim. |
| C2, background cleanup | The background `MPClient` cleanup passes `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS` to the outer manager ([source](https://github.com/vllm-project/vllm/blob/73859fec5865700c5b2021b22890cb3b2005f66e/vllm/v1/engine/core_client.py#L451-L461)); the executor uses that same value for its first wait and can then wait another 4 s after SIGTERM ([source](https://github.com/vllm-project/vllm/blob/73859fec5865700c5b2021b22890cb3b2005f66e/vllm/v1/executor/multiproc_executor.py#L435-L464)). | The `x` versus `x+4` conflict belongs to this entry path. Fixing it alone leaves the explicit-zero API path untouched. |
| C3, timeout attribution | On the async batch-queue path, EngineCore can queue `execute_model` and `sample_tokens` but wait on the latter future first ([source](https://github.com/vllm-project/vllm/blob/73859fec5865700c5b2021b22890cb3b2005f66e/vllm/v1/engine/core.py#L643-L692)). The RPC timeout message names the awaited method but not the response rank ([source](https://github.com/vllm-project/vllm/blob/73859fec5865700c5b2021b22890cb3b2005f66e/vllm/v1/executor/multiproc_executor.py#L393-L412)). | K5's worker-running operation was known from injection; production diagnostics do not thereby know it. The TP=2 arm remains formally unscored. |
| C8, request side | With `shutdown_timeout=0`, EngineCore now visibly aborts scheduler requests and sends abort outputs at shutdown entry; with a positive value it enters drain mode ([source](https://github.com/vllm-project/vllm/blob/73859fec5865700c5b2021b22890cb3b2005f66e/vllm/v1/engine/core.py#L1423-L1466)). | C8 must be limited to deadline/outer-kill interaction, especially for positive drain. “No request abort exists” would be false. |
| C7 | The variable remains declared in current [`envs.py`](https://github.com/vllm-project/vllm/blob/73859fec5865700c5b2021b22890cb3b2005f66e/vllm/envs.py#L777-L779). A default-branch code-search index at `55de40a2` returned the declaration and test/CI settings, but no runtime reader. | The index is not the exact `73859fec` tree; a complete repo-wide absence claim at that SHA has **not** been re-proven. Open #52365 proposes a runtime consumer. |

## Targeted duplicate and ownership search

Search covered vLLM issues **and** PRs, open and closed, with the terms
`shutdown timeout EngineCore CUDA grace`, `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS`,
`VLLM_ENGINE_ITERATION_TIMEOUT_S`, `shutdown drain abort EngineCore`,
`RPC call to sample_tokens timed out rank`, `request drain process kill`,
and `timeout error rank worker`. The most relevant candidates were read in
context. Search ranking is not a proof that no other discussion exists.

| Topic | Closest prior or active work | Disposition for the Lab outline |
| --- | --- | --- |
| C1-on-CUDA | [#24885](https://github.com/vllm-project/vllm/issues/24885) already defined library/serving shutdown expectations (closed as stale); [#31252](https://github.com/vllm-project/vllm/issues/31252) reported parent termination interrupting EngineCore cleanup (closed as not planned). [#52281](https://github.com/vllm-project/vllm/pull/52281) and [#46433](https://github.com/vllm-project/vllm/pull/46433) supplied platform-specific ROCm/XPU responses. | **Not a novel shutdown-contract proposal.** No exact CUDA zero-process-grace report was identified in these searches, but overlap is substantial. Discuss as a scoped unresolved path, not a new issue by default. |
| Generic C2 | [#55632](https://github.com/vllm-project/vllm/issues/55632) reports the ROCm nested-budget problem (open); its [#55646](https://github.com/vllm-project/vllm/pull/55646) patch closed unmerged. The merged [#43154](https://github.com/vllm-project/vllm/pull/43154) deliberately passes the same variable to outer and inner levels. | The generic algebra is still useful, but must credit these owners and distinguish the background path from API shutdown. No separate generic PR now. |
| C7 | [#52365](https://github.com/vllm-project/vllm/pull/52365) is open and proposes using `VLLM_ENGINE_ITERATION_TIMEOUT_S` for GPU-output event waits. | Not an unowned “dead env var” fix. Keep the current-main observation conditional on a complete tree check; review this PR before proposing removal or reuse. |
| C8 | [#24885](https://github.com/vllm-project/vllm/issues/24885) covers drain semantics; [#36964](https://github.com/vllm-project/vllm/pull/36964) remains open for abort-output delivery. Main already aborts and sends outputs on its zero-time path. | Narrow C8 to the distinction between request-drain deadline and process-kill grace. Do not claim an absent abort mechanism. |
| K5 operation/rank attribution | [#58279](https://github.com/vllm-project/vllm/pull/58279) is open for RPC failure/reply handling and shutdown; [#46195](https://github.com/vllm-project/vllm/pull/46195) proposes a better stalled-call error on a different PP path. Organic reports such as [#36921](https://github.com/vllm-project/vllm/issues/36921) show the `sample_tokens` marker in practice. | The distinction between **awaited RPC** and **worker-executing operation** remains a design question, not a demonstrated generic production attribution. Check overlap with #58279's next revision before any proposal. |
| Progress/health | [#36451](https://github.com/vllm-project/vllm/pull/36451) remains open for an EngineCore loop ping; the Lab already posted evidence there. | Link that discussion without another mention or redundant new thread. A loop ping and token progress remain different signals. |

## Decision

The read-only refresh supports continuing the **internal** contract design;
it does not justify a GPU booking or a new broad upstream RFC. The first
public route, if maintainers want it, should be a short, scoped contribution
to an active owner’s thread, with K1/K2/K5 evidence graded as above. Before
posting, recheck that thread’s head/status and complete exact-SHA source
inspection for any claim about a missing reader. Do not treat “no exact match
in targeted search” as proof of novelty.
