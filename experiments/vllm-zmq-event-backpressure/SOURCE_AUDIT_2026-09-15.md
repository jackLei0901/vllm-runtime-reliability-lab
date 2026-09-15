# vLLM #53859 source audit and bounded validation plan

Date: 2026-09-15

Status: source mechanism confirmed; Stage 0 component validation passed

## Question

Can the lab turn the existing vLLM report
[`vllm#53859`](https://github.com/vllm-project/vllm/issues/53859) into a
small, controlled before/after validation of an alive-but-not-progressing
EngineCore?

This is not an attempt to compete with the existing fix
[`vllm#53883`](https://github.com/vllm-project/vllm/pull/53883). The PR is the
candidate ground truth. The lab's job is to test whether its progress detector
and bounded external evidence identify the stall, and whether the same evidence
disappears with the fix applied.

## Source baseline

The audit used vLLM base commit
`22258a26bc090bccf5473cf681bbe9bac41bd035`. The local checkout also contains
the unrelated #52178 patch; none of the files below differ from that base.

The blocking path is directly reachable:

1. `KVEventsConfig.max_queue_size` is a public configuration field, with a
   default of 100,000 (`vllm/config/kv_events.py`). It is accepted through
   `--kv-events-config`; no source patch is required to make the queue small.
2. `ZmqEventPublisher` constructs
   `Queue(maxsize=max_queue_size)` (`vllm/distributed/kv_events.py`).
3. `ZmqEventPublisher.publish()` calls `self._event_queue.put(events)` with no
   timeout.
4. `Scheduler` calls `self.kv_event_publisher.publish(batch)` synchronously
   while preparing the outputs for a scheduling step
   (`vllm/v1/core/sched/scheduler.py`). The call is before construction of the
   `EngineCoreOutputs` returned for that step.

Therefore a full queue can block the EngineCore scheduling thread before the
step result is returned. This confirms the local mechanism claimed by #53859.
The further DP-wide `shm_broadcast` consequence is reported by #53859 and has
not yet been reproduced by this lab.

## Related fixes are distinct

- #53883 changes normal publication to `put_nowait()`, drops a new batch on
  `queue.Full`, and retains the already queued item. This addresses inference
  liveness under publisher backpressure.
- #52857 handles a full queue while `shutdown()` tries to insert its sentinel.
  It addresses cleanup, not the synchronous `publish()` stall.
- #56251 Bug 6 describes a third failure in the same class: shutdown may close
  a ZMQ socket after a bounded join while the publisher thread is still using
  it. It is a lifecycle race, not the #53859 queue-full mechanism.

These three observations should not be collapsed into one claim or one test.

## Cheapest useful validation

### Stage 0: deterministic component oracle (CPU only)

Use the real `ZmqEventPublisher` constructor, but override
`_publisher_thread()` with a test subclass whose consumer waits on a release
`Event`. Bind it to a unique temporary `inproc://` endpoint. This is a declared
fault-injection hook: the unmodified constructor always starts a consumer, so a
consumer that does not drain cannot otherwise be assumed.

Fill the one-slot queue, then invoke `publish()` from a separate thread. Hold
the consumer until after the observation bound and optional external stack
sample; release it explicitly and require the blocked base call to recover.

Pre-register:

- base: the call remains alive after a short bounded observation;
- base: after release, the call returns and the publisher drains normally;
- #53883: the call returns within the bound;
- #53883: the old queued batch remains and the new batch is dropped;
- neither arm relies on elapsed time to infer the internal blocking line; an
  external sample of the subject PID must show the chain
  `threading.Condition.wait` <- `queue.Queue.put` <-
  `ZmqEventPublisher.publish` for the base arm.

This overlaps intentionally with #53883's unit test for the liveness oracle,
but adds the external-observation half needed by the lab.

### Stage 1: single EngineCore, forced backpressure

Only proceed if Stage 0 passes. Start vLLM with KV events enabled and a tiny
`max_queue_size`. Load a test-only pause/release hook inside the EngineCore child
process, for example through a `VLLM_PLUGINS` general plugin. The pause/release
control must cross the process boundary through a file or socket; a
`threading.Event` in the API process cannot release an EngineCore-local thread.
The EngineCore child must also perform its own scoped `PR_SET_PTRACER`
authorization. Changing queue size alone is insufficient: with a functioning
consumer, filling a one-slot queue is still timing-dependent.

Pre-register:

- the API process remains alive;
- the health endpoint may still return success and is therefore not sufficient
  evidence of progress;
- the primary progress signal is the monotonic count of streamed response
  tokens observed by the external client;
- a stall is ten consecutive seconds with no new streamed token while the
  request remains open; before the scored run, the unpaused control must show a
  maximum inter-token gap below two seconds under the same model and request;
- a bounded external stack sample of the EngineCore PID (not the API-server PID)
  places its scheduling thread in the
  `Condition.wait` <- `Queue.put` <- `publish` chain;
- releasing the pause restores progress in the base arm;
- applying #53883 removes the stall under the same trigger;
- with #53883 applied, progress continues during the same pause and a hook-owned
  counter records dropped batches;
- request output correctness is compared only for completed requests.

Stage 1 may need one GPU for an end-to-end server, but it does not need the
field report's DP=8 setup. The exposed queue-size configuration and controlled
consumer pause make this a one-process backpressure experiment.

### Stage 2: DP propagation (optional)

Use two ranks only if the project needs to validate the reported
`shm_broadcast` consequence. Do not make this a prerequisite for showing that
the lab detects the EngineCore stall.

## Evidence contract

The comparison must use the same base source in both arms. Fetch #53883 head
`1a2b8530`, apply only that commit's diff onto `22258a26`, and record the
resulting Git tree hash. Do not compare against the PR's older source tree.

Retain only:

- source/build identity and applied-patch identity;
- declared queue size and trigger identity;
- process liveness and health result;
- monotonic progress samples;
- bounded, allow-listed stack frames;
- publisher pause/release markers and a hook-owned dropped-batch counter;
- per-arm termination class;
- completed-request comparison;
- artifact hashes.

Do not retain request text, model outputs, arbitrary environment variables, or
raw server logs in the public package.

## Stop rules

- Stop if the queue cannot be filled by a deterministic trigger.
- Stop if the base arm does not stall at the component level.
- Stop if the #53883 arm still blocks under the same component trigger.
- Treat a healthy `/health` response during the stall as a useful observation,
  not as a failed reproduction.
- Do not infer the DP-wide mechanism from a single-process result.

## Fix-arm interpretation boundary

#53883 restores inference liveness by dropping a new event batch when the queue
is full. Sequence numbers are assigned later, in the publisher thread, so a
dropped batch receives no sequence number. Subscribers cannot detect that loss
as a sequence gap, and `warning_once` cannot be used as a drop count.

The initial result may therefore claim only **liveness restored with measured
event loss**. It must not claim reliable delivery, replay recovery, or overall
correctness of the proposed policy. Any upstream comment about this trade-off
waits until the experiment has produced evidence.

## Current decision

**Stage 0 PASS, CPU only.** The base arm blocked at the predicted stack and
recovered after release. The same-base #53883 arm returned while retaining the
old batch and dropping the new one. See `STAGE0_RESULT_2026-09-15.md`.

The next step is to prepare the Stage 1 EngineCore-local hook and external
progress oracle. GPU rental is not justified until those pieces and their
pre-registered verifier are ready.
