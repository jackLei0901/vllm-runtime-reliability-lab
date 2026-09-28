# K5 dual-GPU result: executor-dependent health under a worker hold

Status: **one scored TP=1 arm; TP=2 formally unscored, with a source-consistent
RPC-timeout observation**. This is a bounded injected-stall result, not a
root-cause finding for any cited user issue. The [original protocol](../../experiments/engine-liveness-contract/K5_PROTOCOL_2026-09-27.md)
and [environment amendment](../../experiments/engine-liveness-contract/K5_ENVIRONMENT_AMENDMENT_2026-09-27.md)
must be read together. The amendment was written and hashed before the
corrected round. No scoring predicate was changed after either arm ran.

## Frozen identity and evidence

- vLLM reported `0.1.dev586+gc8602c790.precompiled`; seven decisive installed
  Python-source SHA-256 values matched Git commit
  `c8602c79062440074a018c1d5f875a5571eb6881`. The model `config.json`
  SHA-256 was `5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba`.
- The unchanged runner SHA-256 was
  `9fb7f031f5ab36c0a0ffee98268b679763e4bf9f43163e5050307bb1ea0fd29d`;
  its [byte-identical v1 source](../../experiments/engine-liveness-contract/k5_health_stall_gate_v1.py)
  is retained separately from subsequent runner hardening. The latter was
  **not** used in this booking. The worker-hold plugin SHA-256 was
  `00d78fd1caacb20bdf48514a31e588f179702ee9ed008c3a93e8c52820a20fa2`.
  The original protocol SHA-256 was
  `ffa861ef61c79aba6dacd4e69677b16edf6a2869d7bb60ba937f6f2c733d9701`;
  the environment amendment SHA-256 was
  `f66473d8db545681bba68d3df2c8057e00d0f07c3ab8f8bf777ee136d5dc474c`.
- The host had exactly two RTX 4090 GPUs with 24,081 MiB free each at
  preflight. The transferred runner, plugin, protocol, amendment and tests
  matched their local digests; nine K5 CPU tests passed on the host.
- A private archive of the three cell directories and transferred materials
  had SHA-256 `6e82f22cc4f22063bf5ef4eeefccea5a82bfb32e3ba2dbe7c79b191b0b233665`
  on the host and after independent download. Its 32 members had no absolute
  or traversal names and no links. Raw logs and PID-bearing markers remain
  outside the public repository. The `K5_RESULT` lines were emitted to the
  terminal, not saved verbatim in the archive; the closed fields below are
  transcribed from those outputs and corroborated where possible by the
  retained logs and markers. Do not call the archive a complete transcript.

## Cells and scoring

| Cell | Observed result | Formal score | Private log SHA-256 |
| --- | --- | --- | --- |
| Original TP=1 | Startup failed before baseline or arm: FlashInfer could not find the already-installed `ninja` executable on `PATH`. | `unscored_server_not_ready` | `befbf61f0aa1c3a4a2d8be4c6e0cb0ee648386db83c8c7e3277b10ae28d826e3` |
| Corrected TP=1 / `uni` | Baseline passed; rank-0 hold marker was bound to a live descendant; every two-second `/health` sample through 48.05 s was 200; release, successful request completion and post-hold 200 were observed; no RPC-timeout marker. | `tp1_no_timeout_health_200` | `1310c7684c6c6a78cf7e062819e29629ac9146a77557b76b4c98b8b06f1e5e2f` |
| Corrected TP=2 / `mp` | Baseline and rank-0 identity binding passed; `/health` stayed 200 through 28.05 s, then returned 503 from 30.05 through 48.05 s; the request returned 500. The log names `RPC call to sample_tokens timed out.` rather than the preregistered `execute_model` timeout. | `unscored_timeout_attribution` | `70232d7d972955f3b838a01ff0c727184e30fe50c35dd5f010e0a7fa6eb28446` |

The correction was limited to prepending the existing vLLM environment's
`bin` directory to `PATH`. It did not change the 30-second RPC setting,
45-second hold, plugin or scoring code. The original TP=1 attempt remains an
unscored setup failure and is not pooled with the corrected round. No further
retry was made after the TP=2 mismatch.

## What the TP=2 mismatch means

At the pinned source, `MultiprocExecutor.execute_model` **and**
`MultiprocExecutor.sample_tokens` pass the same
`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` value to `collective_rpc`
(`multiproc_executor.py:321-342`). The worker process handles RPC methods
sequentially in `worker_busy_loop` (`:997-1018`). That explains why a hold in
rank 0's `execute_model` also prevents the later `sample_tokens` reply, but
not why the latter names the observed timeout.

The deciding order is in EngineCore. With a compatible default configuration,
`async_scheduling=None` (`config/scheduler.py:148`) is resolved to `True`
(`config/vllm.py:1095-1143`),
giving at least two concurrent batches (`:540-549`). EngineCore then selects
`step_with_batch_queue` (`core.py:231-233`). That path sends nonblocking
`execute_model` and `sample_tokens` RPCs (`:647-675`), but waits on the
`sample_tokens` future first (`:701`); it reaches `exec_model_fut.result()`
(`:705`) only if sampling returns `None`. With rank 0 blocked in the earlier
worker call, this wait order makes `sample_tokens` the reported deadline.
The synchronous `step` instead waits on `execute_model` first (`:596-604`).
The frozen protocol implicitly predicted that synchronous marker. The
effective `async_scheduling` value and `max_concurrent_batches` were **not
independently emitted by this run**, so their exact runtime values remain a
provenance limit; the default-config source path and named timeout agree.

The retained log's `sample_tokens` timeout and subsequent `EngineDeadError`
explain the observed 503. This is a source-based interpretation of the error
chain, **not the preregistered TP=2 score**, which required the exact
`RPC call to execute_model timed out.` marker. The scorer correctly refused
to promote a 503 alone.

The comparison supports a narrow conclusion: with this injected worker hold,
TP=1 remained health-green and recovered after release, while TP=2 reached an
RPC timeout and health 503 at the configured deadline. Because the timed-out
method differed from the frozen prediction, K5 does not yet meet its strict
two-arm acceptance condition. It neither establishes the root causes of
reported organic hangs nor measures false positives or production detection
latency. `VLLM_KEEP_ALIVE_ON_ENGINE_DEATH=1` was deliberately set so the API
could return 503; without it, a deployment may instead lose the connection.

## Disposition

Do not open an upstream bug solely from this result. The source inventory's
C3 evidence grade distinguishes scored TP=1 behavior from the unscored,
mechanism-consistent TP=2 observation. If K5 is ever repeated, derive the
expected marker from **recorded** scheduling mode: default async predicts
`sample_tokens`; explicitly disabled async predicts `execute_model`. This
run must not be rescored under either new prediction. The server processes were absent
after each cell; GPU memory was checked back at its preflight value after
TP=1 but was not re-queried after TP=2. A shutdown
command was issued after the archive passed local integrity checks; the
cloud platform's powered-off state requires separate console confirmation.

After the booking, the [current runner](../../experiments/engine-liveness-contract/k5_health_stall_gate.py)
was hardened for **future** use: preflight requires an available `ninja`, the
matching vLLM environment's `bin` directory is prepended to the child `PATH`,
and each post-preflight `K5_RESULT` is also written to a private `result.json`
using exclusive creation. Its SHA-256 is
`a350e9d5f20d0e59b3aaa521efe6fcc4b8f691aaba35abaf0fc2352f04f033b9`;
11 CPU tests pass in the updated [test file](../../tests/test_engine_liveness_k5.py).
This new version was **not** run on the dual-GPU host and does not change this
result. The historical runner and tests remain as hash-identical v1 snapshots.
