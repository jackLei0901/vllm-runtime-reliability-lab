# Draft comment for vLLM PR #36451 — not posted

Replace `[evidence link]` with a commit-pinned public URL before posting. The author should review the wording against the live PR head and post it personally. Timing suggestion: keep this separate from the planned #52178 follow-up rather than posting both in the same week.

---

I tested this PR's EngineCore health ping against an in-step blocking mechanism rather than `SIGSTOP`: the full KV-event queue in `ZmqEventPublisher.publish()` from #53859. On one RTX 4090, I applied the PR's Python health-ping behavior to an isolated copy of a current vLLM wheel and ran a DP=1 four-cell comparison (one run per cell). This is **source-equivalent for the DP=1 path**, not a run of the exact PR-head build.

With the event consumer paused after work was admitted, `/health` returned 200 with the ping disabled, versus an application-level 503 at the configured 60 s deadline with it enabled. Both requests completed after release; both normal-control cells returned 200. This fits the engine-loop-deadlock category in terafin's split above, not the loop-responsive no-progress case targeted by #45526. It does not establish production false-positive rates or exact-head compatibility. The bounded result, port patch, runner and limits are at [evidence link].

One question for the rebase: under the internal DP load balancer, `DPLBAsyncMPClient` inherits `check_health_async()`, and its `call_utility_async()` gathers every engine. One unresponsive engine would then fail that frontend's `/health` without identifying which engine. Is service-wide failure the intended behavior there, or should the error carry the engine index for attribution and restart decisions? I have not run DP>1 or long-step latency controls.

AI assistance was used for source review and drafting; I checked the experiment and the reported results.
