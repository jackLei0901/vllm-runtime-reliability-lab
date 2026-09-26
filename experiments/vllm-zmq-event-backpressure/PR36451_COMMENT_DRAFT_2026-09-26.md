# Draft comment for vLLM PR #36451 — not posted

Replace `[evidence link]` with a commit-pinned public URL before posting. The author should review the wording against the live PR head and post it personally.

---

I tested the proposed EngineCore health ping against the event-publisher backpressure stall from #53859. On one RTX 4090, I applied the PR's four-file Python health-ping behavior to an isolated copy of a current vLLM wheel and ran a DP=1 four-cell comparison (one run per cell). This is **source-equivalent for the DP=1 path**, not a run of the exact PR-head build.

With the event consumer paused after work was admitted, `/health` returned 200 in 0.003 s with the ping disabled, versus an application-level 503 after the configured 60 s deadline with it enabled. Both requests completed after the consumer was released. Both normal-control cells returned 200. This supports detecting this particular *EngineCore loop-liveness* failure; it does not establish token-progress detection, production false-positive rates, or exact-head compatibility. The bounded result, port patch, runner, identities and remaining gates are at [evidence link].

Two source-review points before merge: the diff appears to reintroduce `VLLM_RPC_TIMEOUT`, which #44128 removed as an unused V0 setting; and `DPAsyncMPClient` inherits the new check while its utility call gathers all engines, so one unresponsive engine may fail whole-service health without identifying which engine. I have not run DP>1, repeated probes or long-step latency controls. The default-on 60 s ping also makes every multiprocess `/health` call an EngineCore round trip. Would you want those boundaries covered in this PR?
