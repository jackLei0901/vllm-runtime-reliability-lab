# NCCL profiler clock attribution: one-booking closeout gate

Chinese companion: [NCCL_PTIMER_CALL_HISTORY_CLOSEOUT_GATE_2026-09-27.zh-CN.md](NCCL_PTIMER_CALL_HISTORY_CLOSEOUT_GATE_2026-09-27.zh-CN.md). This protocol is frozen **before** the next GPU run. It is a closeout of the callback-clock attribution candidate, not a renewal of the Lab's in-flight TP-localization campaign.

## Question and prior evidence

The [completed standalone cell](NCCL_PTIMER_STANDALONE_RESULT_2026-09-27.md) captured three ungrouped `PyNcclCommunicator.all_reduce` calls and replayed twice. It found no equal graph-replay START/STOP clocks. The [post-hoc serving/standalone comparison](NCCL_PTIMER_GRAPH_COMPARISON_2026-09-27.md) found equal-clock classes of 73/78 for serving AllReduce, but class 1 for AllGather and standalone AllReduce. In serving, an eager AllGather on the same communicator follows the reductions. The [corrected NCCL source reading](NCCL_GRAPH_PROFILER_COUNTER_LIFECYCLE_SOURCE_REVIEW_2026-09-27.md) suggests that replay-side profiler counter advancement can depend on the communicator planner state left by its most recent call. This is a **hypothesis**: the retained logs do not expose the internal planner flag, counter or `sub->base`.

This test asks only whether inserting one eager AllReduce on the *same communicator* between capture and replay changes the callback-clock symptom, while an equivalent insertion on a distinct communicator does not. It does not test a TP hang, device progress, stock Inspector's timing fields or a corrective patch.

## Fixed matrix and identities

Run once each, in fresh two-rank processes and owner-only directories, in this order:

| Cell | Sequence after both communicators initialize | Prediction |
| --- | --- | --- |
| A `no_eager` | Capture three ungrouped AllReduces on primary → two numerically checked replays | Prior clean standalone result should repeat. |
| B `same_comm` | Same capture → one eager AllReduce on primary → two replays | If the call-history hypothesis is right, repeated callback clocks should appear on **both** ranks. |
| C `other_comm` | Same capture → one eager AllReduce on independently initialized secondary → two replays on primary | No reuse if the effect is communicator-local. |

Both communicators are constructed in **every** cell, and their constructor warm-up calls precede graph capture. The scorer requires exactly two distinct communicator identities, five pre-window AllReduce occurrences per rank (two warm-ups and three captured calls), the correct identity for the one inserted eager occurrence, six primary-communicator replay occurrences, complete channel markers and exact rank agreement. Before/during/after snapshots isolate the insertion from replay. Every eager result and both replay results are checked numerically. A channel clock is not assumed to be zero or nonzero; only distinct-occurrence equality is scored. No thresholds or scripts are changed after seeing A.

Runtime target: two RTX 4090 GPUs, two `torchrun` ranks, the previously validated vLLM `c8602c79062440074a018c1d5f875a5571eb6881` environment with Torch `2.13.0+cu130`, mapped NCCL `2.29.7` (both ranks SHA-256 `aa957cdfb91b516eae0d54a28e9ee5db52730d02e0ab45580efc3c19a68327a4`), and the occurrence+pTimer Inspector binary SHA-256 `ef734dc33acd60af9edc43b8ec84204c70e3d1fde53294478dc5b08e0d352c3b`. A changed binary or library identity is an apparatus stop until separately reviewed, not a silent substitution. The [driver](../../experiments/vllm-tp-dfx/run_ptimer_call_history.sh) hashes the binary and all analysis sources before running a cell; the runner also checks the vLLM, Torch and mapped NCCL identities. Frozen new source SHA-256: [runner](../../experiments/vllm-tp-dfx/ptimer_call_history.py) `61d939062274819f8f41ad330563ac45931f343524695190333196987d6d44d3`; [scorer](../../experiments/vllm-tp-dfx/ptimer_call_history_score.py) `f869729baf414b58545a1d7db435a525922498897addb1cce2674d08ad3bb59f`.

## Booking order and stop rules

1. Read-only preflight, at most 30 minutes: verify both GPUs, identity of the chosen venv and loaded NCCL, plugin digest, free space, `torchrun`, `ninja` and SSH host fingerprint independently supplied by the owner. Do not install into or delete an existing environment to make the gate pass.
2. Run the three fixed cells through `bash run_ptimer_call_history.sh VENV_DIR PLUGIN_SO PRIVATE_BASE LAB_ROOT`. Each cell has a 180-second process bound. A setup, timeout, numeric, rank-binding, occurrence, channel or digest failure makes the booking `unscored`; retain the failed cell and do **not** retune or rerun. Do not select a favourable subset of ranks or collectives.
3. Seal raw rank logs, snapshots, commands, runtime and score JSON in one owner-only private archive. Hash it on host and after download; check member names and links before extraction. Publish only closed counts, digests and version identities—not clocks, communicator hashes, PIDs, hostnames, IPs, raw traces or private host paths.
4. After evidence transfer and rehash, power off the instance under the user's authorization. SSH loss is not proof of cloud billing state; request platform confirmation separately.

Local preparation: the three new CPU scorer tests and the full `PYTHONPATH=src` suite passed (**318 tests, 15 skipped**); the new runner compiled under CPython 3.14, and the shell driver passed Git Bash `bash -n`. This does not validate the GPU route or profiler callback semantics.

## Frozen interpretation

| A / B / C result | Closeout decision |
| --- | --- |
| A no reuse; B reuse on **both** ranks; C no reuse | `conditional_callback_reuse_reproduced`. Draft a standalone, source-backed NVIDIA review/issue candidate. This is a symptom reproduction, not proof of `base=0` or a stock metric error. The original TP-localization line remains paused. |
| A no reuse; B no reuse; C no reuse | `candidate_not_reproduced`. Record the remaining serving/standalone difference as unresolved and stop this line. No second booking. |
| A reuse, C reuse, one-rank-only reuse, a structural check failure or any inconsistent pattern | `unscored_or_confounded`. Retain evidence and stop. Do not reinterpret a subset as support. |

The result is *not* admission of a Lab probe. Reopening real TP in-flight localization requires a separate organic fault, a new trustworthy producer capability, or explicit passage of the Lab's §5 probe gate.
