# vLLM #53859 Stage 1 R3 result

Execution date: 2026-09-16 UTC

Verdict: **PASS at the single-GPU serving boundary**

## Result

All four cells matched the behavioral contract.

| Cell | Source | Trigger | Outcome |
| --- | --- | --- | --- |
| 1 | base `22258a26` | control | 64 completion tokens; no stall or drop |
| 2 | base plus #53883 | control | 64 completion tokens; no stall or drop |
| 3 | base `22258a26` | paused consumer | stalled; health stayed 2xx; stack matched; recovered after release |
| 4 | base plus #53883 | paused consumer | no stall; 1 batch accepted and 4 dropped; 64 completion tokens |

In cell 3, 11 streaming progress events arrived before release. The consumer
was released 10.36 seconds after request start. The EngineCore stack sample
matched the allow-listed blocking chain through the lab's `observed_put`
wrapper, `Queue.put()` and `threading.Condition.wait()`. Progress resumed after
release and the request completed.

The cell 3 counters are cumulative at cell end. One batch was accepted before
the queue stalled; the blocked batch and the remaining batches were accepted
after the consumer was released, producing the final count of five.

In cell 4, the request completed before the delayed release could be relevant.
The EngineCore-local counter recorded one accepted non-null batch and four
`queue.Full` drops. No stack capture was taken because no stall was observed.

Both controls produced 64 progress events. Every cell removed the server
process group and the bound EngineCore PID/start-time identity during cleanup.

## Source and runtime identity

- Base commit: `22258a26bc090bccf5473cf681bbe9bac41bd035`
- Base tree: `b7061e73a6ed4773e16bd2ae3acf47aebfd1342d`
- #53883 head: `1a2b85306b6d13033bbecc693e6cb776acb4bcaa`
- Fix tree: `46bc6e191b14ce12a04827454b4588ea5d3a435f`
- Fix patch SHA-256:
  `ebf0e35f53e6e3a74c79d608f6a2656d3f7537bcb3c648ce6885a8eac3423dfc`
- GPU: NVIDIA GeForce RTX 4090, capability 8.9
- Python: 3.12.3
- PyTorch: 2.13.0+cu130
- CUDA runtime reported by PyTorch: 13.0
- Yama `ptrace_scope=1`, with scoped `PR_SET_PTRACER` authorization

The server command and request hashes are identical across all cells. Within
each source arm, the complete recorded runtime environment and build-identity
hash are identical between control and pause. Across arms, the common runtime
fields are identical; only the source-specific `kv_events.py` hashes and the
generated editable vLLM version differ.

The verifier's source-specific pins are not derived from the R3 outcome. The
base value `de08f01e...` and fix value `15c3038f...` already appear in the
committed Stage 1a summaries under
`results/vllm-zmq-backpressure-stage1a-20260915/` and its R2 rerun, before R3
was executed. The correction turns those earlier identities into explicit
per-cell requirements while allowing the expected editable-version difference.

The EngineCore independently reported the selected tree's `kv_events.py` hash
and a non-empty set of mapped worktree binaries. The campaign checked those
binaries against the per-arm identity record before sending the request.

## What this establishes

- The blocking queue path described in vLLM #53859 is reachable from a real
  model-serving EngineCore on one GPU.
- Under a paused KV-event consumer, the base server can remain health-responsive
  while token generation stops making progress.
- A privacy-bounded external stack sample identifies the blocked publisher path
  without retaining request or response content.
- Releasing the consumer restores progress in the base arm.
- Under the same trigger, #53883 preserves request progress by dropping new
  event batches instead of blocking EngineCore.

The result therefore supports **liveness restored with measured event loss**.
It does not establish reliable KV-event delivery.

## Limits

- This is a single-GPU, single-EngineCore experiment. It does not test data
  parallelism or the issue's reported DP-wide `shm_broadcast` consequence.
- The trigger is injected by a test-only plugin that pauses the real publisher
  thread and wraps the real queue's `put` method. The wrapper frame is present
  in the captured stack and is not part of upstream vLLM.
- The R3 identity record covers the exact source tree and installed worktree
  binaries used by EngineCore. Unlike the earlier R2 record, it is not a full
  published dependency manifest. The retained private audit records the
  self-contained environments and setup checks.
- The observed drop count belongs to this deterministic request and queue size;
  it is not a production loss-rate estimate.
- A dropped batch receives no publisher sequence number. This run does not show
  that downstream subscribers can detect the loss.

## Qualification and excluded attempts

R1 and R2 stopped at their first base control before any scored trigger. Their
records remain in `STAGE1_R1_CONTROL_FAILURE_2026-09-15.md` and
`STAGE1_R2_CONTROL_FAILURE_2026-09-15.md`; neither contributes to this verdict.

Before R3, Stage 1a passed against both exact R3 source environments with
PyTorch 2.13.0+cu130. The base stack matched on its first sample, and both arms
cleaned up without residual processes.

The first R3 no-hook server qualification failed before health readiness because
the absolute Python launch did not put the environment's existing `ninja`
binary on `PATH`. No package was added or changed. Repeating the qualification
with the arm environment's `bin` directory explicitly prepended passed. The
same explicit `PATH` rule was used for all four scored cells. This setup failure
is not evidence about #53859 or #53883.

The executed verifier originally required the generated vLLM version to be
identical across base and fix. It reached that final cross-cell comparison only
after every per-cell check passed. The verifier was corrected after execution
to require common runtime fields to match across all cells, exact full
environment equality within each arm, and fixed source-specific
`kv_events.py` hashes. No result field was changed.

## Public/private boundary

Only the four closed-shape summaries are published. They contain numeric
progress offsets, bounded classifications, source identities and SHA-256 values;
they contain no prompt, generated text, raw log line or raw stack frame.

Raw server logs, the full stack sample, private inputs, hook files and the R3
identity records remain in the ignored private audit archive. The public
summaries retain hashes of the raw server logs and stack sample.

The retained private archive `stage1-r3-evidence-20260916.tgz` has SHA-256
`4b4a5efcd9e14e2d392b600f80a255630dda456077a0f2080f947cb3d30d5801`.
A future redacted public dependency manifest would make the R3 environment
independently inspectable rather than only tamper-evident.

Verify with:

```bash
python experiments/vllm-zmq-event-backpressure/verify_stage1_results.py \
  results/vllm-zmq-backpressure-stage1-r3-20260916
```
