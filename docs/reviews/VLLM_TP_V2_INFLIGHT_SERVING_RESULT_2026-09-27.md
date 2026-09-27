# TP=2 V2 in-flight serving gate: bounded run result

## Decision

**Unscored for the in-flight/export-policy question.** The healthy control
passed and the hold reached a real V2 cached-FULL replay on rank 1. During the
bounded wait, the request was pending and the rank/process identities were
stable; after release, the request completed. The callback-history verifier
nevertheless rejected the cell: one `(communicator, sequence)` key did not
retain a single channel count across the log. We do not select a favourable
subset of callbacks, infer a transport hang, or claim a stock-Inspector export
gap from this run. The offline identity-contract review below explains the
collision but cannot rescue the missing per-occurrence attribution.

This is a controlled host-side delay before a graph replay, not an organic
serving fault. The stock Inspector JSON is only a completed-record control;
its aggregate counts do not join to the pending communicator and sequence.

## Frozen inputs

- [Protocol and CPU gate](VLLM_TP_V2_INFLIGHT_SERVING_GATE_2026-09-27.md):
  `325f93c`, fork/request-window amendment `dc553d2`, and the exact Inspector
  enablement preflight `e17e933`. The previously unscored V1-wrapper run is
  [preserved separately](VLLM_TP_INFLIGHT_SERVING_RESULT_2026-09-27.md).
- vLLM `c8602c79062440074a018c1d5f875a5571eb6881`, PyTorch
  `2.13.0+cu130`, NCCL `2.29.7`, two RTX 4090 GPUs, the same local model
  checkpoint used for the prior serving cell, and forced
  `disable_custom_all_reduce=True`. The selected runner was observed as V2;
  `VLLM_USE_V2_MODEL_RUNNER` was unset. No default-route claim follows from
  the forced PyNccl setting.
- The runtime NCCL shared-library/wheel digest was **not retained**. The
  `v2.29.7-1` source analysis below is not an exact runtime-binary provenance
  claim; the next preflight must pin the loaded library's SHA-256 privately.
- Patched Inspector binary SHA-256:
  `e2f9acc8d985ab936503c58b89ac3965fe8ea2bb254a8ab76522538a0fd6c7ef`.
  The transferred V2 plugin, final runner, witness reader and callback parser
  SHA-256 values were respectively
  `37ebb91273d73285c2c2352a5f7c347a9ffa22c13c16b59183e71ec193ddc067`,
  `dc918c78f154950326fcc54830dc50ca24b429dd9f7bc9754b13d2d2df85b338`,
  `393a2d3e14442d70d97c4caef85aa8f14c15168b7de9e0c9353c74e73f3aa900`,
  and `214000cbe22c52ffdb2a2163646075f0175ec028a4079e17e3354ed3cb2fccd9`.
  The last digest is the frozen v1 parser at
  [`7d4f4ec:inflight_trace.py`](https://github.com/jackLei0901/vllm-runtime-reliability-lab/blob/7d4f4ec/experiments/vllm-tp-dfx/inflight_trace.py).
  The occurrence follow-up uses a separate `inflight_trace_v2.py`; it does
  not reinterpret the old logs or change the V1 runner's parser.
- The private archive of the four cells and transferred source has SHA-256
  `5bf9befc99a7a778a3d3e3341dbe6dad08e12b41c5d6b3a5e2b0cd3aa517d4a0`,
  matched before and after transfer to a local directory outside this repo.
  Raw logs, witness filenames, PIDs, hostnames, paths, communicator keys and
  model output are not public evidence.

## Cell inventory

| Cell | Observation | Status |
| --- | --- | --- |
| Initial load smoke | Engine startup failed because `ninja` was absent from `PATH`; the binary existed in the pinned dependency pool. | Apparatus failure; no request window. |
| Corrected load smoke | Both forked workers created distinct PID-keyed V2 manager witnesses, but Inspector was disabled by its default setting, leaving no rank-bound callback declaration. | `unscored / preflight_evidence_unavailable`. |
| Healthy control | Inspector enabled; both worker witnesses reported `ModelCudaGraphManager`, `FULL_AND_PIECEWISE`, non-breakable graphs and at least one cached FULL replay **after** the observation marker. Stable PID/start-tick identities, 16 tokens, and no hold marker. | `healthy_v2_full_replay_observed`. |
| One bounded hold | Rank 1 recorded one eligible replay and `hold_entered=true`; rank 0 did not hold. A snapshot was taken within the 3-second wait; identities stayed stable and 16 tokens completed after release. | `unscored / callback_history_invalid`. |

The apparatus corrections were made and checked before the scored healthy
control and hold: `ninja` was put on `PATH`, and `NCCL_INSPECTOR_ENABLE=1`, a
500-microsecond dump interval and verbose export were added to the runner's
exact environment check. Each attempt used a fresh private cell directory.
No second hold was run after the verifier rejected the first.
The fork-safe PID-keyed witnesses and post-`observe` cached-FULL replay
counter both worked; neither validates the callback key. The healthy-control
code did not run that validator, so it passed despite the same conflict in
its final logs.

## Why the hold cannot be scored

The frozen parser treats `(communicator, sequence)` as the identity of one
collective and requires its `CollStart` channel count to be stable. Reading the
retained final logs with that parser found **two conflicting keys per rank** in
both the healthy and hold cells. In the hold cell, one conflicting key had
1,460 `CollStart` records with channel counts of both 1 and 2; the other had
two such records. This is a closed aggregate of private data, not publication
of the keys. The offline check found the large key **1,460 times in each of
four logs** (two ranks in each of the healthy and hold cells), using the
[closed-shape legacy aggregate](../../experiments/vllm-tp-dfx/legacy_callback_aggregate.py).
Its consecutive
channel-count runs were `73, 219, 73, 1,095` in every log, all multiples of
73; 1,460 is `73 × 20`. The smaller conflicting key appeared twice in each
log. These counts support repeated graph replays of the same per-type
sequence, but do not prove which model steps or collective types produced
the two keys. No raw key, rank PID or log line is published.

The pinned [NCCL profiler source](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/plugin/profiler.cc)
indexes `seqNumber` by collective function and normally excludes persistent
graph-captured tasks from the increment. Its exception requires a Group or
CollApi handle together with KernelCh activation. The pinned
[Inspector plugin](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/plugins/profiler/inspector/inspector_plugin.cc)
enables only Coll and KernelCh events, so it does not supply those handles.
The [host-stream plan callback](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/enqueue.cc)
can issue profiler task events on each replay. Thus `(communicator, sequence)`
is not a unique replay occurrence; it also omits the collective function.
This is a source-supported explanation consistent with the retained
aggregates, subject to the missing runtime-binary pin above. The two-record
collision could be cross-function, but the old logs omitted `func`, so that
specific attribution remains unverified.

The old logs cannot be repaired honestly: the Nth `CollStart` can be counted,
but old `KernelChStart` lines carry no occurrence ID. Assigning them to a
particular replay would require a nearest-match guess. The in-memory
before/during snapshots were not separately retained either, so the final
logs cannot replay the original triplet verdict exactly.

Earlier synthetic two-collective and serving reports used aggregate callback
counts and known graph placement, not a `(communicator, sequence)` join to
name a serving collective. Their count observations stand; any stronger
collective-localization reading would exceed those records.

## Next gate

The [occurrence-identity follow-up](VLLM_TP_V2_OCCURRENCE_ID_GATE_2026-09-27.md)
specifies one further bounded cell only after a source-patch build and a
healthy callback-validity control. It keeps the original verdict `unscored`.
