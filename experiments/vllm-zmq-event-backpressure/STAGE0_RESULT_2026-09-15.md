# vLLM #53859 Stage 0 result

Date: 2026-09-15

Verdict: **PASS at the component boundary**

## Result

The deterministic CPU-only experiment matched every frozen prediction.

| Observation | Base `22258a26` | Same base plus #53883 |
| --- | --- | --- |
| Consumer paused before publication | yes | yes |
| Second `publish()` returned within 1 s | no | yes |
| First queued batch retained | yes | yes |
| External stack located blocked call | yes | not required |
| Second `publish()` state after release | completed | already returned |

The base stack showed the allow-listed chain:

```text
publish_second (stage0_backpressure.py)
  publish (vllm/distributed/kv_events.py)
    put (queue.py)
      wait (threading.py)
```

The base call returned after the controlled consumer was released, and the queue
drained. This recovery observation distinguishes publisher backpressure from an
unrelated permanent failure.

## Source identity

- Base commit: `22258a26bc090bccf5473cf681bbe9bac41bd035`
- GitHub/base tree: `b7061e73a6ed4773e16bd2ae3acf47aebfd1342d`
- Reconstructed archive tree: the same value
- #53883 head: `1a2b85306b6d13033bbecc693e6cb776acb4bcaa`
- Ported fix tree: `46bc6e191b14ce12a04827454b4588ea5d3a435f`
- Ported changes: exactly `vllm/distributed/kv_events.py` and
  `tests/distributed/test_events.py`

The GitHub archive omitted five ignored blobs. They were fetched by blob SHA
from the pinned commit; after adding them, `git write-tree` exactly matched the
GitHub commit tree. No content or mode difference existed among the other 7,082
blobs.

## Environment

- CPU-only instance; no GPU present
- Python 3.12.3
- msgspec 0.21.1
- pyzmq 27.1.0
- Yama `ptrace_scope=1`
- scoped `PR_SET_PTRACER` authorization for the observer parent

## What this establishes

- The unbounded `Queue.put()` path blocks the caller under deterministic
  publisher backpressure.
- The blocking location is externally observable without reading request data.
- Releasing the consumer restores the base call.
- The #53883 change restores caller liveness under the same trigger by dropping
  the new batch.

## Important cost and limitation

The dropped batch never reaches the publisher thread, where sequence numbers
are assigned. A subscriber therefore sees neither the batch nor a sequence gap.
The `warning_once` log is also not a loss counter.

`new_batch_dropped` is derived rather than emitted by the runner: the queue has
capacity one, it still contained exactly the first object, and the second
`publish()` returned without an exception. The queue constraint leaves dropping
the new object as the only compatible outcome. The queue-full warning was
observed separately in the captured logger output (stdout in this run).

This result supports only **liveness restored with event loss**. It does not
establish reliable delivery, replay recovery, EngineCore behaviour, API health,
or DP-wide `shm_broadcast` propagation. Those remain Stage 1/2 questions.

## Excluded setup attempt

An earlier base attempt matched the queue behaviour but was not scored:

- `py-spy` was rejected under Yama because the subject had not authorized the
  observer;
- the standalone process retained the shared ZMQ context after producing its
  result.

The pre-correction protocol SHA-256 was
`c436462305d582e18c4d10fd7705d330e747e008877e092e26fd5472d88db23c`, and the
pre-correction runner SHA-256 was
`db6fee77c88466cd9d4a20d255e4fb10f216c846cee1b00afa431bb8fa79c4f1`.
The runner and protocol were corrected before the scored run. The changes only
added scoped `PR_SET_PTRACER` authorization and terminated the isolated ZMQ
context after production shutdown. Predictions, queue size, observation bound,
hold bound and source arms did not change.

## Public/private boundary

The public summary contains source identities, outcomes and allow-listed stack
frames. The runner directly emitted the queue and return observations. The
queue-full warning came from the captured logger output (stdout in this run),
while `stack_observed` and the call chain came from a separate `py-spy` capture.
Raw stdout, stderr and the full stack dump remain outside the public package;
their SHA-256 values are retained in `summary.json`.

The verifier checks this reviewed summary against the frozen expectations and
the protocol/runner hashes. It does not rebuild the summary from the private raw
files.

The exact executed runner is retained as `stage0_backpressure_executed.py`.
After the run, `stage0_backpressure.py` was hardened so a missing release raises
instead of silently returning after 300 seconds and so the derived completion
field is emitted directly. The exception occurs in the consumer thread; it does
not directly determine the process exit code. A missing release still makes the
run fail because the queue does not drain. No rerun was performed or claimed;
both runner hashes are recorded and checked.

Verify with:

```bash
python experiments/vllm-zmq-event-backpressure/verify_stage0_result.py
```
