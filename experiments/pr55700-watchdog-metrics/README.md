---
model_cells: [M3, M4]
status: active
upstream_exit: vLLM PR 55700 review; not posted
last_scored: never
---

# PR #55700 watchdog counter during a continuing hold

Draft probe for the [frozen selection record](../../docs/reviews/PR55700_FIRST_CANDIDATE_2026-09-28.zh-CN.md)
and its [addendum A1](../../docs/reviews/PR55700_FIRST_CANDIDATE_ADDENDUM_A1_2026-09-28.zh-CN.md),
plus [setup addendum A2](../../docs/reviews/PR55700_FIRST_CANDIDATE_ADDENDUM_A2_2026-09-28.zh-CN.md),
and [model-download addendum A3](../../docs/reviews/PR55700_FIRST_CANDIDATE_ADDENDUM_A3_2026-09-28.zh-CN.md),
which must be committed before any data is acquired. Nothing here has been run.
Pin: PR head `b274bf04dd4c6d54807a136babce5b5d17dd74be`.

`probe.py` starts a real `vllm serve` on the CPU backend, launched by the same
interpreter whose vLLM installation it verified. The only substitution is
`--worker-cls hold_worker.HoldingCPUWorker`, which holds one `execute_model`
call on request. The EngineCore loop, executor, watchdog, scheduler statistics,
output transport and frontend Prometheus logger are unchanged PR code.

| Run | Question scored | Hold |
| --- | --- | --- |
| `--tp 1` (uniproc) | EngineCore count during a continuing hold | rank 0, inside EngineCore |
| `--tp 2` (multiproc) | Worker count during a continuing hold | rank 0 (output rank) |
| `--tp 2` (multiproc) | Secondary: does a non-output rank's count reach `/metrics` | rank 1 |

Each run first performs control (c): three warm-up requests, after which the
required series must exist (absent is not zero), then an idle period in which
nothing may change. Episodes then cover (a) sampling `/metrics` every second
during the hold after a `feed timeout` witness, and (b) scraping after release.
Per addendum A1, `supported` requires that post-release export, and any failed
control, hold entry, held request, release, scrape, or missing series is
`unscored`. An unverified identity writes an `unscored` receipt and never starts
the server.

Source facts the parameters depend on, at the pin: an idle EngineCore feeds
every 5 s (`input_queue.get(timeout=5)`), an idle shm reader every 5 s
(`SHM_READER_RECHECK_INTERVAL_MS = 5000`), and the `execute_model` RPC deadline
is 300 s. Defaults: watchdog timeout 15 s, check interval 1 s, hold 45 s. The
probe refuses settings that would fire while idle or reach the RPC deadline.
The CPU server uses `--gpu-memory-utilization 0.5` (the flag's name also
applies to CPU memory) after a pre-hold startup failure at the 0.92 default.
If the direct Hugging Face endpoint is unreachable, set
`HF_ENDPOINT=https://hf-mirror.com` and record that environment change.

Prediction recorded before any run, source inference only: with TP=2 the
EngineCore waits for worker replies inside the shm reader, which feeds the
same process-wide watchdog, so an EngineCore witness is not expected during a
worker hold and that cell should be `unscored`.

## Run (Linux x86-64 CPU host, no GPU)

First run a read-only preflight: memory and free disk for a source build,
AVX512/AVX2 flags, Python and compiler versions. A container limited to about
2 GiB of memory is not expected to build or serve even a 0.5B model; record a
failed preflight as `NO-GO`, not as a probe result.

Build vLLM for CPU from the pinned PR head, following
`docs/getting_started/installation/cpu.x86.inc.md` **at that commit**
(`VLLM_TARGET_DEVICE=cpu` source build). Then, inside that environment:

```bash
python experiments/pr55700-watchdog-metrics/probe.py --tp 1 \
  --vllm-src /path/to/vllm --work-dir /private/pr55700/tp1
python experiments/pr55700-watchdog-metrics/probe.py --tp 2 \
  --vllm-src /path/to/vllm --work-dir /private/pr55700/tp2
```

`--vllm-src` must be a clean checkout at the pin; the probe also compares the
installed PR-changed files with it and scores nothing if they differ. The work
directory holds `server.log`, stack dumps and `result.json`. Keep the first two
private; `result.json` contains counts, relative times, configuration and
outcomes only. The scoring logic is covered on CPU by
`tests/test_pr55700_watchdog_probe.py`, which does not need vLLM.
