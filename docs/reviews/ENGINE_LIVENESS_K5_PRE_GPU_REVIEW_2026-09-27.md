# K5 pre-GPU review: TP=1/TP=2 health under a bounded worker hold

Historical preparation snapshot. The later [dual-GPU run record](ENGINE_LIVENESS_K5_DUAL_GPU_RESULT_2026-09-27.md)
supersedes its pending-run status; this page preserves the pre-run predictions
and frozen hashes rather than rewriting them after observing the outcome.

Status: **ready for external review, not a GPU result**. Start with the
[preregistered protocol](../../experiments/engine-liveness-contract/K5_PROTOCOL_2026-09-27.md),
then the [frozen v1 runner](../../experiments/engine-liveness-contract/k5_health_stall_gate_v1.py),
[test-only plugin](../../experiments/engine-liveness-contract/k5-worker-hold-plugin/llr_k5_worker_hold.py),
and [frozen v1 CPU tests](../../tests/k5_test_snapshot_v1.py). These files are local
and uncommitted. Do not call the experiment source-frozen on a cloud host
until its transferred files match the hashes below.

Post-run note: the unsuffixed runner and test files now include future-run
hardening. The linked `*_v1.py` snapshots are byte-identical to the files
used in the booking; the later versions did not generate the K5 result.

## Why these are the right two arms

At pinned vLLM `c8602c79062440074a018c1d5f875a5571eb6881`, the `uni`
executor's `execute_model` uses local `collective_rpc` with no deadline
(`uniproc_executor.py:108-123`), whereas `mp` passes
`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` into its worker RPC
(`multiproc_executor.py:321-331`). Both reach GPU
`Worker.execute_model` (`gpu_worker.py:1019`). The source's executor selector
maps `uni` and `mp` to those classes (`executor/abstract.py`), so the runner
sets each backend explicitly rather than inferring it from TP size after the
fact. Both arms pin the timeout to 30 seconds, with an identical 45-second
rank-0 host hold.

`AsyncLLM.check_health` tests its errored flag (`async_llm.py:920-923`), while
the API watchdog otherwise may exit after failure (`launcher.py:168-190`).
Keeping the API alive for this experiment makes an application-level 503
observable. It is a declared departure from default deployment behaviour,
not an assertion that a real health probe would always see 503 instead of a
closed connection.

## Controls and failure boundaries

The runner refuses a version mismatch, a difference in seven installed
Python-source hashes, a non-4090 or low-memory GPU pair, a missing cached
model, a used output directory, or an occupied port. Each fresh server must
answer `/health` and complete a request *before* the arm file is created.
The plugin only holds the first positive-token `Worker.execute_model` on rank
0 after arming; the entered marker is checked against a live `/proc` PID,
start ticks and ancestry. Without that binding, neither a persistent 200 nor
a missing timeout is scored.

The prediction is a paired contrast, not a standalone 503 claim:

- TP=1/`uni`: 200 throughout the hold; release, request completion and a
  subsequent 200 must all be observed.
- TP=2/`mp`: 200 before the 30-second deadline, then at least one 503 at
  32–44 seconds while the hold remains active, plus the named RPC-timeout log.
  A 503 without the timeout line is `unscored_timeout_attribution`.

An early unhealthy response is a contradiction, not a pass. A missing
marker, insufficient time window, transport failure in place of 503, or a
different failure chain is unscored. There is one attempt per arm; no live
adjustment of the hold point or thresholds. Raw logs and PID-bearing markers
remain under `0700` directories; public output contains only normalized
health samples, typed outcomes and hashes. The runner handles SIGTERM from
the outer timeout and attempts bounded process-group cleanup.

## Frozen local materials

| Artifact | SHA-256 |
| --- | --- |
| Protocol | `ffa861ef61c79aba6dacd4e69677b16edf6a2869d7bb60ba937f6f2c733d9701` |
| Runner | `9fb7f031f5ab36c0a0ffee98268b679763e4bf9f43163e5050307bb1ea0fd29d` |
| Plugin module | `00d78fd1caacb20bdf48514a31e588f179702ee9ed008c3a93e8c52820a20fa2` |
| Plugin distribution metadata | `9688835d87e1cd40227abf48bf9dc7ccb2fb7cffde9dcde03cd4c5d28b8bdaf1` |
| Plugin entry-point file | `710bf3c3c0dac7f01553b4089bf8672fd8921dc1284a88afb46c00fabc771b2a` |
| CPU tests | `32c83da760fb12bf2562c25055be3256bada4666aaf4dea67d6fb2867f5c6380` |

The seven vLLM Python hashes in the runner were extracted from a Git archive
of the pinned commit, not copied from the installed wheel. The model
`config.json` hash is recorded by the runner at booking time. The plugin
entry-point metadata is also part of the transferred package and must be
checked before launch.

## Checks performed without GPUs

- Nine pure activation/scoring tests passed, including wrong rank, missing
  binding, early unhealthy response, absent timeout attribution, and a flat
  TP=2 health series.
- Both Python modules compile; the runner's CLI help parses.
- The whole Lab suite passed on the latest run: **332 tests, OK, 15 skipped**.
  The immediately preceding run failed one unrelated client-probe progress
  assertion; its isolated rerun also failed with a different verdict. This
  apparent timing-sensitive test remains a separate follow-up, not K5
  validation. Ruff was unavailable here. No GPU-serving path, plugin loading
  inside vLLM, or CLI flag compatibility was executed locally.

## Review questions before booking

1. Is the rank-0 pre-`execute_model` hold an adequate common stall class for
   both executor paths, or does it cause an unrelated collective failure in
   TP=2 before the 30-second RPC deadline?
2. Does the pinned installed build accept the explicit `uni`/`mp` CLI backend
   flags and load the general plugin in the intended worker processes? Treat
   failure as apparatus, not a negative C3 result.
3. Is the keep-alive setting sufficient to retain `/health` long enough to
   observe 503 after EngineCore failure? If not, the result is unscored for
   this exact health-status claim; do not reinterpret connection refusal as
   503 after the run.

No upstream text follows from preparation alone. A scored two-arm run would
support a bounded C3 mechanism claim; cited user issues still require their
own root-cause evidence and a full duplicate check before an RFC.
