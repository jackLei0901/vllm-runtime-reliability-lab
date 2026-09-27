# TP=2 V2 occurrence-identity capability run — result

## Decision

**Unscored: `target_collective_ambiguous`.** The Linux patch/build gate and the healthy serving control passed. The one preregistered rank-1 hold entered a V2 cached-FULL replay window; the request was pending during capture, rank/process identities remained stable, and 16 tokens completed after release. The frozen scorer found no **unique** target occurrence. Offline replay of the retained exact snapshots found 74 rank-asymmetric occurrences during the same window: 73 `AllReduce` and one `AllGather`. Rank 0 emitted `KernelChStart` callbacks for all 74 while rank 1 emitted none. This is a callback observation, **not** evidence that rank 0's GPU started or completed 74 collectives. It explains the unscored result without replacing its score or selecting a favourable callback.

This controlled host-side delay is not an organic NCCL transport hang. Neither `CollStart` nor `KernelChStart` proves collective completion or token progress. The result does not establish that stock Inspector exports incomplete collectives.

## Frozen inputs and provenance

- Preregistered [gate](VLLM_TP_V2_OCCURRENCE_ID_GATE_2026-09-27.md) and [one-booking runbook](VLLM_TP_V2_OCCURRENCE_ONE_BOOKING_RUNBOOK_2026-09-27.md). One healthy control and one hold were run; there was no parser change or second hold on the host.
- NCCL Inspector source tag `v2.29.7-1`, commit `b91894bd5b190c874d98a017f93f5daa515b65d0`. The corrected [local occurrence patch](../../experiments/vllm-tp-dfx/inspector-inflight-occurrence-v2.29.7-1.patch) SHA-256 was `5cbaa80d97ecbaf79e388d74466b15522fd585c8b514953f289195c3cd2474aa`. Both local and remote `git apply --check` passed; remote compilation and `git diff --check` passed. New Inspector binary SHA-256: `c6abacf8c7aee7d08a63f23d0efbee8e06fc4e4e2e1be618cf682044dae93053`. It was built in a fresh checkout; earlier Inspector binaries were not overwritten.
- Two RTX 4090 GPUs, pinned vLLM source `c8602c79062440074a018c1d5f875a5571eb6881` (`0.1.dev586+gc8602c790.precompiled`), PyTorch `2.13.0+cu130`, CUDA reported by Torch as `13.0`, and NCCL `2.29.7`. The same local Qwen3-4B-Instruct-2507-v2 checkpoint and forced `disable_custom_all_reduce=True` were used. Both workers mapped one identical runtime NCCL library, SHA-256 `aa957cdfb91b516eae0d54a28e9ee5db52730d02e0ab45580efc3c19a68327a4`. Selected module digests and the full environment are retained privately; the printed version is not a complete wheel attestation.
- The transfer archive matched SHA-256 at both ends: `51c3c88df650735a28c7e0a4b6aeaf228fbd110a2fd0082f72419694a51fab73`. The final private evidence archive matched SHA-256 on the remote and local hosts: `97062797e921cdf4ad53ae15816470549bc7c50afce27acba887d08eb0ff5f7b`. It is stored outside the public repository. Raw NCCL logs, callback snapshots, witness filenames, PIDs, communicator IDs, host paths and model output are not published.

## Cell results

| Cell | Required observations | Frozen outcome |
| --- | --- | --- |
| Build/load gate | Corrected patch applied at the pinned source; `.so` compiled and contained `LLR_TP_EVT_V2`. The control then bound exactly two rank logs, distinct worker identities, V2 graph-manager witnesses and one mapped NCCL library. | Pass; eligible for healthy control. |
| Healthy control | Both ranks reported `ModelCudaGraphManager`, `FULL_AND_PIECEWISE`, non-breakable cached-FULL replay after `observe`; no hold entered; 16 tokens completed; callback history validated. A one-use receipt was written with SHA-256 `aaee549613570d2ee69830f56d1f9a2786dec22adee5ba5b0f3efdfa20dff10f`. | `healthy_v2_full_replay_observed`; eligible for one hold. |
| One bounded hold | Rank 1 had one eligible call and `hold_entered=true`; rank 0 did not hold. A pending request and held snapshot were observed; identities stayed stable; 16 tokens completed after release. The receipt matched and was consumed. | `unscored / target_collective_ambiguous`; no rerun. |

The hold's exact private snapshot SHA-256 values were `9f631a2023deb794edab7731ba5908678df9525aa4ffe61442960bd8c340d5aa` (before), `6c3303cc6d2cfdf090e9cd88dcd96ffaf1adb210212c62843945fdaae89135aa` (during), and `dd71ffeb0dbad2e69a7e41b663114be40bd89bbeeb17385691d83f61be9e0097` (after). They permit offline reproduction of the frozen scorer and the closed-shape callback audit without using final logs as a substitute for mid-window evidence.

## Post-run offline callback audit

This audit reads the same retained, digest-checked **before** and **during** snapshots; it does not rerun or rescore the hold. It selects occurrence keys with a new rank-0 `kernel_ch_start` and no new rank-1 start, then counts new stops on those **same** keys. The 74 asymmetric occurrences are contiguous `CollStart` positions 75–148 of the 1,184 request-window occurrences, aligned to a 74-collective forward-step boundary. For those 74 exact occurrence IDs, rank 0 gained **75 `kernel_ch_stop` events covering all 74 occurrences** during the hold (one occurrence had two channels); rank 1 gained **zero**. The 74 include 73 `AllReduce` and one `AllGather`. Rank 1 was held on the host before its graph replay, so these stop callbacks cannot certify completion of the corresponding current-replay collectives on rank 0's GPU. The exact occurrence join is stronger than an aggregate callback count, but it validates callback identity only, not device truth.

Stock Inspector's aggregate completed-`AllReduce` record counts were 265/264 before and 286/276 during for ranks 0/1: deltas **+21/+12**. These records are asynchronously exported and are **not joined to the 74 occurrence IDs**; they cannot show that any of the 74 completed. They also prevent treating an increasing aggregate as a current-step completion witness. Raw records and identifiers stay private.

## Interpretation and next gate

The earlier `(communicator, sequence)` collision has been removed as an identity problem: per-callback occurrence IDs passed the healthy control, including cross-rank descriptor checks. The present obstacle is **callback semantics**, not merely the unit of discrimination. The pre-replay hold made rank 0 emit start and stop callbacks for an entire 74-collective step while rank 1 emitted none; it did not show that rank 0's device traversed that step. In NCCL `v2.29.7-1`, the [proxy-side start/stop tests](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/transport/profiler.cc#L23-L48) compare expected values with device-written counters, and [proxy counter assignment](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/proxy.cc#L549-L566) has a persistent-plan branch. A stale-counter match under graph replay is a **source-backed hypothesis**, not an established root cause of these callbacks.

A subsequent [counter-lifecycle source review](NCCL_GRAPH_PROFILER_COUNTER_LIFECYCLE_SOURCE_REVIEW_2026-09-27.md) traces a concrete reachable zero-base path through capture, planner cleanup and replay. The experiment still did not record the runtime branch and counter values needed to confirm that path for each callback.

Do not promote this to `start_asymmetry_observed` retroactively. A graph-step rule over these callbacks would establish host-side replay/callback asymmetry, not device progress; the V2 activation witness already supplies a host-layer fact. Moving the hold inside the graph would not repair a callback that can report a stop for a collective that cannot have completed. Before another GPU booking or an Inspector export design, finish the source-level counter-lifetime analysis and preregister a minimal standalone two-rank, two-collective graph reproducer that checks whether a later collective receives `KernelChStart` or `KernelChStop` while its peer has not replayed. No NVIDIA issue or new Lab probe is admitted from one serving hold. The v0.2 progress verdict remains independent of this native evidence.

## Cloud state

After archive transfer and hash verification, `shutdown -h now` reset the SSH session and the 2222 tunnel returned connection refused. The user then confirmed that the AutoDL console displays the instance as powered off. The SSH observation alone would not have established the control-plane billing state.
