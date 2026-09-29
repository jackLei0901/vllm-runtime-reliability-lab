---
model_cells: [M2, M5, M6]
status: active
upstream_exit: no post; user review required after a scored process result
last_scored: never
---

# PR #54553 fatal-shutdown CPU check

This is the smallest proposed test for the [candidate gate](../../docs/reviews/PR54553_THIRD_CANDIDATE_2026-09-29.zh-CN.md). It calls the actual `EngineCoreProc.run_engine_core` from two clean pinned checkouts. A fake EngineCore injects the fatal/clean outcome and a bounded or blocked `shutdown()`; **the patched exception/finally branch is not copied**. This cannot reproduce an XPU driver wedge or prove a supervisor restart.

Preflight requires Linux, two compatible Python environments that import their checkout's vLLM on CPU, and enough RAM for two separate vLLM imports (sequential, not concurrent). No GPU, model or server is used. Patch head: `41dddf7`; base: its parent `810bc3250c945829c64a745b4f695ddfd8f9a598`. The runner refuses changed tracked files or a `core.py` that differs from its Git blob. Check the PR discussion again before running, so this does not duplicate a result the author has since posted.

From the Lab root, with a **new private output directory**:

```bash
python experiments/pr54553-fatal-shutdown/probe.py run \
  --base-src /path/vllm-base --patch-src /path/vllm-pr54553 \
  --base-python /path/base-venv/bin/python \
  --patch-python /path/patch-venv/bin/python \
  --out /private/pr54553/first-run
```

All three cells run once per revision: fatal plus a held teardown, fatal plus a quickly completing teardown, and clean `SystemExit` plus a quickly completing teardown. The new timeout is set to 1 s for this CPU process test. The parent waits at most 45 s for import/setup, then at most 4 s after the teardown-entry witness, killing **only its own child** if still running. On the expected base hold, that parent kill is the observation; it is not an engine exit. The other two cells are negative controls. A missed marker or imported module from the wrong checkout is `unscored`.

`result.json` contains only commit IDs, exit codes, rounded relative durations and fixed markers. Child stderr (which can contain paths) remains in the private output directory; do not publish it unreviewed. This runner has not been executed against vLLM. Windows syntax and scorer tests do not count as a process result.
