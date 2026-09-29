---
model_cells: [M5]
status: active
upstream_exit: vLLM PR 52365 review; not posted
last_scored: never
---

# PR #52365 default event-wait bound on a completing step

Draft runner for the [selection record](../../docs/reviews/PR52365_SECOND_CANDIDATE_2026-09-29.zh-CN.md)
and [addendum A1](../../docs/reviews/PR52365_SECOND_CANDIDATE_ADDENDUM_A1_2026-09-29.zh-CN.md),
which must be committed before any GPU is booked. Nothing here has been run.

The question: with PR head `d996d76` and default settings, does a request
that completes on the base `157bcb7c` and on the PR with the timeout disabled
fail specifically at the new 60 s event wait? Whole-request time is only a
screen. The disabled arm must have a measured, completed single wait >= 65 s.
vLLM source is never modified.

| Stage | Command | What it establishes |
| --- | --- | --- |
| 1 | `runner.py sweep` on the base | Whole-request time by a fixed increasing length ladder; after the first >= 70 s screen, measures exactly one more preregistered step and chooses that as long only if successful and >= 70 s; also picks a <= 50 s short control, or records `no_candidate` |
| 2 | `runner.py ab` | The same requests on three fresh servers: base, PR default, PR with `VLLM_ENGINE_ITERATION_TIMEOUT_S=0`; only the PR arms use a private timing wrapper around the unchanged wait function |

Every server is started with `--distributed-executor-backend uni`,
`--async-scheduling`, `--no-enable-prefix-caching`, `--max-num-seqs 1` and
`--max-num-batched-tokens` equal to `--max-model-len`, so each request's
prefill can fit in one step; actual wait timing is observed separately. At
both pins the V1 generation path returns the async
output that the PR's wait covers only when async scheduling is on
(`gpu_model_runner.py`, `if not self.use_async_scheduling` before
`AsyncGPUModelRunnerOutput`), so the flag is set explicitly rather than left
to automatic selection. The experiment also sets `VLLM_USE_V2_MODEL_RUNNER=0`
in all arms; this is not a test of the default runner. Prompts are token-id
lists; each request uses a different token id.

## Setup (one Linux GPU host)

The PR changes only Python files, so both checkouts can reuse the base's
precompiled binaries:

```bash
git clone https://github.com/vllm-project/vllm.git vllm-base
git -C vllm-base checkout 157bcb7c489689dd34cf28d9c9970a465d326a03
git clone https://github.com/vllm-project/vllm.git vllm-pr
git -C vllm-pr fetch origin pull/52365/head
git -C vllm-pr checkout d996d76ec68e9f6a348b7b085ae1da61cb6095be
```

In each checkout, following vLLM's AGENTS.md: `uv venv --python 3.12`, then
`VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto` with that
checkout's `.venv/bin/python`. Pass each interpreter explicitly; the runner
verifies the checkout pin, a clean tracked tree, and the installed PR files
with the same interpreter it launches.

## Run

```bash
python runner.py sweep --python /path/vllm-base/.venv/bin/python \
  --vllm-src /path/vllm-base --model MODEL --model-revision MODEL_COMMIT_SHA \
  --max-model-len MODEL_NATIVE_CONTEXT --lengths TESTED_LENGTHS \
  --work-dir /private/p52365/sweep
python runner.py ab --base-python /path/vllm-base/.venv/bin/python \
  --base-src /path/vllm-base --pr-python /path/vllm-pr/.venv/bin/python \
  --pr-src /path/vllm-pr --model MODEL --model-revision MODEL_COMMIT_SHA \
  --max-model-len MODEL_NATIVE_CONTEXT \
  --short SHORT --long LONG --work-dir /private/p52365/ab
```

Before booking, record `nvidia-smi` memory, `command -v ninja`, the exact model
commit, the fixed prompt-length ladder, and why it fits on this GPU. Commit
the addendum, runner, timing hook and tests before collecting data. The
runner and hook digests go into `result.json`; the hook records only duration
and outcome of the real imported wait function. An absent trace is unscored.
Before startup, each interpreter is checked from a neutral directory for an
existing `sitecustomize` without this experiment on `PYTHONPATH`; if present,
the runner writes an unscored receipt and launches nothing.

Choose a public model whose **native** context covers every tested prompt
**plus one output token**, and use a fixed model commit SHA. Record the exact
length list before the run. Choose a model whose memory use is feasible at
that `--max-num-batched-tokens` setting and
record why the configuration is one a user could run; do not raise the
context limit with overrides. Check the model card and memory before booking.
The length ladder needs at least one step *after* the first >= 70 s screen;
if there is none or the next step fails, the result is `no_candidate`. Run
`ab` only if `sweep` returns `candidate_found`. Work directories must be
new or empty. Server logs and per-call JSONL traces stay private; `result.json`
holds settings, aggregate event-wait times/counts, booleans and outcomes.
`not_reproduced` does not refute the deadline; EngineCore health and exit are
recorded separately from the request outcome. Scoring is covered on CPU by
`tests/test_pr52365_event_timeout_runner.py`.
