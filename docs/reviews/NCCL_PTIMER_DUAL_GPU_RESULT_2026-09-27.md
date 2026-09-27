# NCCL graph-profiler `pTimer`: one dual-GPU capability run

Chinese companion: [NCCL_PTIMER_DUAL_GPU_RESULT_2026-09-27.zh-CN.md](NCCL_PTIMER_DUAL_GPU_RESULT_2026-09-27.zh-CN.md). This is the result of the [pre-registered protocol](NCCL_PTIMER_DUAL_GPU_PROTOCOL_2026-09-27.md), not a change to the frozen v0.2 verdict or the earlier `unscored / target_collective_ambiguous` occurrence result.

## Outcome

The eager instrument control passed on both ranks. The healthy V2 serving control passed. The single hold entered, kept the request pending during the snapshot, and released to a complete 16-token request; the existing occurrence gate nevertheless returned **`unscored / target_collective_ambiguous`**. The independent `pTimer` postprocessor found repeated GPU-clock values within communicator/channel groups in both the healthy serving control and the held window. A post-run source check makes the healthy-control result stronger: ordinary forced-PyNccl TP calls are submitted separately, so a class of 73 distinct occurrences cannot all carry the device timestamp of their own separate executions. This is a source-to-callback inference, not proof of the `base=0` mechanism, a wrong stock metric, or a standalone NCCL defect.

There was one failed **setup attempt** before the scored serving control: the fixed virtual environment's `ninja` executable was not on `PATH`, so engine initialization failed before readiness. The `PATH` was corrected without changing code, thresholds or the plugin; a fresh control directory was used. No second hold was run.

## Frozen identity and evidence

| Item | Identity / status |
| --- | --- |
| GPUs | Two RTX 4090s; one run per cell. |
| vLLM / Torch / NCCL | vLLM `c8602c79062440074a018c1d5f875a5571eb6881`; Torch `2.13.0+cu130`; loaded NCCL `2.29.7`. The runner's software and mapped-library checks passed. |
| NCCL source | `v2.29.7-1`, `b91894bd5b190c874d98a017f93f5daa515b65d0`; fresh detached worktree. |
| Inspector patches | Occurrence patch SHA-256 `5cbaa80d97ecbaf79e388d74466b15522fd585c8b514953f289195c3cd2474aa`, then `pTimer` delta SHA-256 `8c1fd0860ff35996b6af70cd13722f48152c8d9b689fce9b47bbbc30f513aabe`. Both `git apply --check` calls and `git diff --check` passed. |
| Newly built Inspector | SHA-256 `ef734dc33acd60af9edc43b8ec84204c70e3d1fde53294478dc5b08e0d352c3b`; zero compiler warnings. The previously validated binary was not overwritten. |
| Postprocessor | [Frozen source](../../experiments/vllm-tp-dfx/ptimer_trace.py), SHA-256 `8400e4972e826a8b1b18633b7f298ad3e5673ba1e1da1f301af0cf3f2c255ebd`; transferred file rehashed identically. |
| Control receipt | SHA-256 `daa4513c622482c7cc378fc2033c184d6847abcc6ace825af80b5a339c44b2c0`; reports `healthy_v2_full_replay_observed` and pins the newly built Inspector. |
| Private evidence archive | SHA-256 `17d3c0a687e2d7f555591fbdc54dcc7512421fb1bb76305895420ba0b7849d3e`; 64 archived entries, owner-only mode `600`. Its digest was computed on the host; it was not independently rehashed after download. Raw logs, GPU clocks, communicator IDs and process identities remain private. |

The eager, healthy-control and hold postprocessor JSON digests are respectively `86fa4f61f1fb5136ee999ea549b864483af41dc23fc6cd22c765c37324d2d3d3`, `3669c55f0ac1691c03dfd240cacfbedef35200b2d90453eb904791051f506d37`, and `b619478e4fbece22391c27120bb0aacad30c00ec89791b880a5d9601f794a521`. Each summary was generated from two rank-bound private logs. The serving summaries also required digest-checked, prefix-consistent before/end snapshots.

This new archive contains the new patch, postprocessor, eager script, build log, binary and all four cell directories. The unchanged v2 runner, parser, stall plugin and activation witness are retained in the prior occurrence campaign's separate private archive; their fixed digests are in the protocol. The new archive alone is therefore not a self-contained source bundle.

## Closed observations

| Cell | Rank 0 | Rank 1 | Gate reading |
| --- | --- | --- | --- |
| Eager PyNccl | 6 paired start/stop channel events; no zero, repeated or nonincreasing clocks within its one communicator/channel group; no `stop <= start`. | Same. | Instrument positive control passed, including arithmetic. The six include the PyNccl initialization warm-up and five explicit calls. |
| Healthy V2, after minus before | 1,273 start and 1,273 stop channel events; largest within-group equal-clock class 73 for each kind; no zero or nonpositive paired duration. | 1,273 start and 1,273 stop; largest classes 78 start / 77 stop; no zero or nonpositive paired duration. | Runner reported `healthy_v2_full_replay_observed`, complete request and stable identity. Clock reuse is observed, not scored as wrong timing. |
| Single held window, during minus before | 164 start and 164 stop; largest within-group equal-clock class 58 for each kind; no zero or nonpositive paired duration. | 148 start and 148 stop; largest class 26 for each kind; no zero or nonpositive paired duration. | The rank-1 pre-replay hold was entered; the request was pending in the snapshot and completed after release. The runner still returned `unscored / target_collective_ambiguous`. These counts do not identify a common target collective. |

The grouped equality classes exclude accidental equality across different communicators or channels. [Batch stamping](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/device/common.h#L318-L335) is within a kernel launch; it does not explain the ordinary forced-PyNccl TP calls, which [invoke NCCL directly without an explicit group](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/distributed/device_communicators/pynccl.py#L166-L213) and are [individually submitted through NCCL's implicit group](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/enqueue.cc#L2786-L2823). The class of 73 comprises distinct callback occurrence IDs in one communicator/channel. For those ordinary TP calls, its values cannot all be their own execution's device start (or stop) clock. The published summary does not retain the class's function composition or stream mapping, so this is a **post-hoc, route-scoped inference**, not a retroactive preregistered score. Adjacent inversions remain unscored until occurrence order and effective stream are bound. No `stop <= start` pair was observed; this is compatible with START and STOP both reading a valid earlier slot pair. The hold's rank-1 callbacks cannot be interpreted as current-replay device progress merely because rank 1's host was held before that replay.

## Decision and next gate

The healthy control supports the narrower but stronger conclusion that **not every callback clock belongs to its named, separately submitted collective execution** on this forced-PyNccl route. It does not establish which other work supplied the clock, that `base=0` occurred, that stock Inspector's `coll_exec_time_us` or bandwidth is wrong, or that an existing producer cannot expose a correct device-progress signal. The existing hold occurrence result remains `unscored / target_collective_ambiguous`: a repeated clock cannot select its target collective. No Lab probe is admitted and no upstream NCCL issue is justified by this serving run alone.

Next, recover and independently rehash the private archive before a **post-hoc** offline audit: within each communicator/channel, count distinct occurrences sharing START or STOP clocks (stratified by function), clocks in the held window below the pre-window maximum, and adjacent inversions only where order/stream are established. These counts have not been computed and must not be reported as observed. The archive remains on the powered-off host; its remote digest above is not a post-download check.

Before another NCCL-core build, attempt a standalone two-rank reproduction: ungrouped all-reduces separated by bounded device work, once eagerly and once in a CUDA graph, with the same Inspector `pTimer` instrumentation. Distinct increasing eager clocks versus repeated graph-replay clocks would establish the callback symptom without vLLM. A separately identified NCCL-core instrument is **optional mechanism confirmation** if that reproduction leaves `base`, planner state or slot identity unresolved. A stock-metric claim would additionally require a completed record bound to an occurrence and an independent timing oracle. For the next booking, preflight must include `command -v ninja` in the selected venv's `PATH` before the first serving cell.
