# Stage 1a CPU preflight

Status: **prepared, not yet executed**

Stage 1a closes two integration questions before a GPU rental. On both the base
and same-base #53883 trees it must:

1. import `vllm` and `kv_events.py` from the selected clean worktree and match
   the imported file to the Git blob;
2. load the installed `vllm.general_plugins` entry point;
3. construct the real `ZmqEventPublisher` on an `inproc://` endpoint;
4. call the real `publish()` path from a scheduler-like thread;
5. observe at least one accepted batch in both arms and at least one rejected
   non-blocking batch in the fix arm;
6. have a separate parent process attach with real `py-spy` to the base arm and
   match `Condition.wait <- Queue.put <- publish`;
7. record the number of stack attempts needed;
8. confirm the kernel has restrictive Yama (`ptrace_scope >= 1`) and that
   thread-local setup produced `pr_set_ptracer_observer` authorization;
9. require the plugin's self-reported `__file__` hash to match the lab copy and
   record hashes for the plugin, preflight, shared campaign helpers and shared
   scoring contract;
10. require cleanup to remove both the process group and the PID/start-time
    identity that hosted the publisher.

The sampled base stack contains the lab wrapper between the production queue
and publisher frames:

```text
threading.Condition.wait
queue.Queue.put
dfx_stage1_backpressure.observed_put
ZmqEventPublisher.publish
```

The ordered matcher deliberately permits that disclosed instrumentation frame.
The preflight waits one fixed second after the scheduler-like caller starts,
before taking its first sample. This removes the caller-start race from the
attempt count. Formal Stage 1 may rely on one sample only if the base arm records
`stack_attempt_count == 1`; otherwise it adopts the same bounded retry and
discloses it.

This is an integration preflight, not scored Stage 1 evidence. It uses no model
and makes no inference-liveness claim.

Install the Stage 1 plugin without dependencies in each source environment,
then run from the lab checkout:

```bash
python experiments/vllm-zmq-event-backpressure/stage1a_cpu_preflight.py \
  --source-arm base \
  --server-workdir /absolute/path/to/base \
  --python /absolute/path/to/base/python \
  --private-dir /tmp/stage1a-base-private \
  --summary /tmp/stage1a-base.json

python experiments/vllm-zmq-event-backpressure/stage1a_cpu_preflight.py \
  --source-arm fix \
  --server-workdir /absolute/path/to/fix \
  --python /absolute/path/to/fix/python \
  --private-dir /tmp/stage1a-fix-private \
  --summary /tmp/stage1a-fix.json
```

Do not reuse either private directory. A public result is prepared only after
both summaries say `PASS` and their import identities have been reviewed.
Place the summaries together under the exact names `stage1a-base.json` and
`stage1a-fix.json`, then run:

```bash
python experiments/vllm-zmq-event-backpressure/verify_stage1a.py \
  /absolute/path/to/stage1a-results
```
