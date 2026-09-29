---
model_cells: [M3, M4]
status: active
upstream_exit: none; exploratory CPU check is not a serving result or upstream post
last_scored: never
---

# PR #59179: DP dummy work versus request progress

This is a bounded CPU check against vLLM PR #59179 at
`b3687914aa7a1ded4bde8af67a35e17ac6ef2f95`. It calls the PR's actual
`DPEngineCoreProc.run_busy_loop`, `_process_engine_step`, progress publication,
and `AsyncMPClient` progress consumer. It mocks one unschedulable but unfinished
request and the distributed all-reduce. It does **not** reproduce a remote-KV
incident, run an HTTP server, or establish production impact.

Result: [English](RESULT_2026-09-29.md) ·
[中文](RESULT_2026-09-29.zh-CN.md).
The exact uncommitted runner used for the exploratory run is now retained as
[probe_exploratory_20260929.py](probe_exploratory_20260929.py) at its recorded
SHA-256. Use `probe.py` for a future receipt-bearing run; do not conflate them.

Prediction: each of two DP dummy batches advances `ready_progress_seq` despite
zero real request steps. The test ages both stored timestamps between cycles,
without waiting 60 wall-clock seconds. The frontend's previously stale BUSY
rank then ceases to be classified as stalled each time. A different result,
missing broadcast, changed source,
or broken controls is `unscored` or a reason to revisit the source hypothesis;
it is not evidence of a serving bug either way. Whether `/ready` promises GPU
execution capability or admitted-request progress is a contract choice.

Use a CPU-capable Linux venv that imports the pinned checkout and run:

```bash
/path/to/venv/bin/python experiments/pr59179-readiness/probe.py \
  --vllm-src /path/to/vllm-pr59179 \
  --receipt /path/to/new-result.json
```

The script refuses a different HEAD, changed relevant source files, or imports
from another checkout. It prints one JSON line with fixed, non-private facts.
If GitHub Git transport is unavailable, the fixed commit can instead be
downloaded from
`https://codeload.github.com/lio1226/vllm/tar.gz/b3687914aa7a1ded4bde8af67a35e17ac6ef2f95`.
The alternate command adds `--source-archive /path/to/source.tar.gz` and sets
`--vllm-src` to its extracted top-level directory. This route accepts only the
archive SHA-256 `e6dcdc927b75f9442019cd2fa5de1661637fc656dfdbead43cfe2ac1107d1f5f`
and verifies the four readiness source files byte-for-byte against it. The
result identifies which source route was used. The archive route was added
before the first exploratory run because GitHub Git transport timed out on the remote host;
it does not relax the source pin or the scoring conditions.
The remote prebuilt venv's Python modules and compiled extension were installed
for an earlier vLLM revision. If the pinned source lacks a compiled extension,
`--binary-package-dir /path/to/venv/site-packages/vllm` adds only that package
directory as a fallback import path; the checked readiness classes must still
come from the pinned source. Its stable-ABI extension digest is recorded. This
is an apparatus accommodation for a CPU-only method check, not a serving or
binary-compatibility validation. Import failure remains `unscored`.
The required `--receipt` must name a new file: the runner refuses to overwrite
an earlier result and writes one normalized JSON line before printing it.
The 2026-09-29 run predated this receipt writer and was not preregistered in
committed files. It remains exploratory; rerun with the committed runner and a
receipt before citing it as a Lab artifact.
No GPU booking, model download, server launch, or upstream post is authorized
by this check. If the PR head changes, review its source and tests before
repinning; never silently reuse this result.

中文：本测试只核对“DP dummy batch 是否刷新前端的进展计时器”。它不证明真实远端
KV 故障，也不预设 `/ready` 必须保证每个请求前进。只有结果可评分且契约目标明确后，
才考虑向作者提出问题；不据此直接提出新 RFC。
