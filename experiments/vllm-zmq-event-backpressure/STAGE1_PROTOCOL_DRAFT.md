# vLLM #53859 Stage 1 protocol draft

Date: 2026-09-15

Status: **pre-execution draft; no GPU result exists**

## Objective

Test whether bounded external evidence distinguishes a live API process from a
stalled EngineCore when the ZMQ KV-event publisher queue is full, and whether
the same trigger preserves inference progress after applying #53883.

This stage tests one EngineCore on one GPU. It does not test the reported
data-parallel `shm_broadcast` consequence.

## Source identity

- Base commit: `22258a26bc090bccf5473cf681bbe9bac41bd035`
- Candidate fix: the single-commit diff from #53883, head `1a2b8530`
- Queue size: 1
- Both arms must use the same model, request, launch arguments and observer.
- The base and fix Git tree hashes must match the Stage 0 identities before a
  scored run.

Both arms must also pass `verify_stage1_build_identity.py`. That record pins
the exact baseline wheel, PyTorch/CUDA pair, driver/GPU identity, dependency
manifest and every binary or executable member installed from the wheel. Each
installed binary must reside in its selected source tree and match the wheel
member byte for byte. The two arms must have identical binary identities.

Non-core packages are exposed through one reviewed
`stage1-dependency-pool.pth` file in each arm environment. Its sole line is the
absolute read-only pool path. The build record hashes both the file and its
value. For every visible distribution it records the normalized name, version,
SHA-256 of `RECORD`, count of content-verified files, and whether its metadata
came from the arm environment or the pool. Every hashed `RECORD` entry is
verified against its installed bytes. Duplicate normalized names fail
generation. Every pool top-level entry must be owned by a distribution
`RECORD`. The two manifests must match except for `vllm` and the test plugin.

## Fault hook

Install the package under `stage1_plugin/` into the vLLM environment and set:

```text
VLLM_PLUGINS=dfx_stage1_backpressure
DFX_STAGE1_ENABLE=1
DFX_STAGE1_CONTROL_DIR=<fresh absolute directory>
DFX_STAGE1_OBSERVER_PID=<campaign process PID>
```

The general plugin is loaded inside EngineCore before the scheduler constructs
the publisher. It wraps only `ZmqEventPublisher._publisher_thread`:

1. authorize the declared observer with `PR_SET_PTRACER` when Yama is present;
2. wrap the real queue's `put` method once, counting accepted non-null batches
   and failed non-blocking non-null batches separately;
3. atomically publish `ready-<pid>.json` with the EngineCore PID and process
   start time; more than one ready file fails the cell;
4. wait for a release file before entering the production publisher loop.

Setup failures produce `hook-error-<pid>.json` with a bounded error class.
They are scored as `hook_failed`, not conflated with server readiness.
The ready record also carries the plugin's own file hash and the observed Yama
scope. The campaign rejects a plugin hash that differs from its reviewed lab
copy.

The ready record also carries the SHA-256 of the EngineCore process's imported
`vllm.distributed.kv_events` file. It must equal the selected source tree's Git
blob before the request starts. This is independent of the campaign process's
import probe.

At ready time the hook also reads EngineCore's `/proc/self/maps` and retains
only source-tree-relative paths and hashes for mapped `.so` files. Formal Stage
1 requires a non-empty mapped set and requires every item to match the complete
installed-wheel binary identity. This distinguishes installation-time location
from actual load-time identity without retaining machine paths.

The campaign creates a fresh control directory per trial. A pre-existing
release file is therefore impossible. The control arm creates the release file
before server launch; the affected arm creates it only after its observation
and capture decision.

The hook does not change scheduler code, event generation, queue size, request
contents or the production publisher loop after release.

## Matrix

| Source arm | Trigger | Required outcome |
| --- | --- | --- |
| base | control | request completes; at least one batch accepted; no stall or drop |
| base | pause | at least one batch accepted; request stalls; stack matches; release restores progress then completion |
| fix | control | request completes; at least one batch accepted; no stall or drop |
| fix | pause | request completes while paused; at least one batch accepted and one dropped |

Run the two controls before either affected arm. A failed control stops the
campaign. Run each cell once for the boundary decision; repeat only after a
separately recorded protocol revision.

## Progress and liveness

The primary progress signal is the monotonic count of non-empty streamed
response chunks observed by the external client. Record only monotonic offsets
and counts, never request text or generated content.

- Control eligibility: maximum inter-chunk gap below 2 seconds.
- Stall: after at least two non-empty chunks, the request remains open and the
  count does not increase for 10 consecutive seconds.
- Recovery: after the release marker, the count increases again and the request
  completes within 30 seconds. The release offset and progress count at release
  are recorded; a failed stream cannot count as recovery.
- API liveness: sample `/health` during the 10-second no-progress interval. A
  200 response is an observation, not evidence of EngineCore progress.

Every arm must accept at least one event batch. The request must report at least
16 prompt tokens and 32 completion tokens: with the frozen block size of 16,
this covers one full prefill block and two output blocks. Otherwise the trial is
`trigger_not_reached`, not a pass.

## Stack evidence

For the base/pause arm only, sample the PID written by the EngineCore-local
plugin. The capture must occur after the 10-second stall decision and before
release. Retain publicly only whether this ordered allow-list was present:

```text
threading.Condition.wait
queue.Queue.put
ZmqEventPublisher.publish
```

Raw stack output remains private and is represented publicly only by SHA-256.
Failure to obtain the stack is `evidence_unavailable`; it is not equivalent to
the stack being absent.

## Event-loss evidence

The hook wraps only the queue instance's `put`. This avoids double counting
because `Queue.put_nowait` delegates to `put(block=False)`. A successful
non-null put increments the accepted count. A non-null `queue.Full` with
`block=False` increments the dropped count and is re-raised so #53883's
production handler still decides whether to drop the batch.

The fix/pause arm requires a count greater than zero. The count does not prove
subscriber-visible sequence gaps because sequence numbers are assigned later
in the publisher thread.

## Public result contract

Retain:

- source, patch, protocol, plugin and runner hashes;
- paired exact-wheel build-identity hashes;
- the dependency-pool attachment hash and per-distribution manifest;
- all installed wheel binary hashes and the EngineCore-mapped subset;
- Python, vLLM, CUDA and GPU identity;
- the imported vLLM and `kv_events.py` paths relative to the source tree, and a
  byte hash proving imported `kv_events.py` matches that Git tree;
- arm and trigger identity;
- EngineCore PID binding outcome, not command-line contents;
- streamed progress counts and monotonic offsets;
- health status classes;
- allow-listed stack match plus raw-artifact hash;
- accepted/drop counts plus pause and release observations;
- request completion and process-group cleanup classes.

Exclude request text, generated text, arbitrary environment variables and raw
server logs. A missing producer must remain distinct from a negative
observation.

## Stop rules

- Stop if either control fails or its maximum progress gap is at least 2 s.
- Stop if the plugin does not bind exactly one EngineCore PID.
- Stop if no KV-event batch reaches the hook.
- Stop if the base/pause arm does not meet the 10-second stall definition.
- Fail closed if the required stack is unavailable or mismatched.
- Do not infer DP-wide behaviour from this stage.

The server command explicitly enables prefix caching, fixes block size at 16,
and enables async scheduling. Defaults and warnings are not accepted as proof
of those conditions. Negated forms and duplicate block-size or KV-event
configuration entries are rejected.

The fix/pause cell runs the same live stall monitor as the base/pause cell; its
`false` result is observed rather than filled in by construction. Ready files
and bounded hook errors are enumerated again at cell end so a later EngineCore
cannot escape the one-instance rule.

Failure classification is ordered: hook integrity, stream failure, missing
release observation, then undersized request. A missing usage record after a
stream error therefore cannot relabel the primary failure as
`trigger_not_reached`.

## Decision

`PASS` requires all four cells to meet their required outcomes. A pass supports
only this claim: under a deterministic EngineCore-local backpressure trigger,
external progress and stack evidence identify the base stall, while #53883
preserves request progress by dropping measured event batches.
