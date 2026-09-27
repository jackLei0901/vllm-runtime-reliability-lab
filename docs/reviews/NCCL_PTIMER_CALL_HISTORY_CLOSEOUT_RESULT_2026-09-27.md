# NCCL profiler clock call-history closeout: apparatus did not reach the score

Chinese companion: [NCCL_PTIMER_CALL_HISTORY_CLOSEOUT_RESULT_2026-09-27.zh-CN.md](NCCL_PTIMER_CALL_HISTORY_CLOSEOUT_RESULT_2026-09-27.zh-CN.md). This run followed the [frozen closeout gate](NCCL_PTIMER_CALL_HISTORY_CLOSEOUT_GATE_2026-09-27.md) at Lab commit `38cee45`. It produced **no scored A/B/C comparison**.

## Decision

Cell A (`no_eager`) stopped at its first, pre-replay callback snapshot. Both rank logs contained exactly **two** completed collective occurrences, on **two distinct communicator identities**: the constructor warm-ups. The runner required five there, counting the three collectives captured into the graph. Those three capture-time occurrences had not appeared in the callback logs at that point. After the bounded snapshot wait, rank 0 raised `bounded profiler snapshot unavailable`; rank 1 then lost its Gloo peer. The driver returned nonzero. Neither the intervening-call step nor any graph replay ran; cells B and C were not started.

The five-occurrence pre-replay requirement was an **apparatus mistake in the new gate**, not evidence about clock attribution. This observation only establishes that the expected capture-time callback records were absent from this pre-replay window. It does not establish their general delivery timing, `comm->planner.persistent`, `op.workCounter`, `sub->base`, a stock Inspector metric error, or whether a same-communicator eager call changes replay clocks. The prior [serving/standalone comparison](NCCL_PTIMER_GRAPH_COMPARISON_2026-09-27.md) and its call-history hypothesis remain unproven, not refuted.

| Cell | Result | Scope |
| --- | --- | --- |
| A `no_eager` | `unscored / pre_replay_callback_assumption_failed` | Two rank-bound warm-up occurrences each; no replay or clock score. |
| B `same_comm` | `not_run / gate_stopped` | No observation. |
| C `other_comm` | `not_run / gate_stopped` | No observation. |

## Evidence and limits

The two-GPU host and SSH fingerprint matched the previously confirmed instance. The transferred runner and scorer matched the hashes frozen in the protocol; the selected Inspector binary matched SHA-256 `ef734dc33acd60af9edc43b8ec84204c70e3d1fde53294478dc5b08e0d352c3b`. The selected venv reported Torch `2.13.0+cu130` and vLLM `0.1.dev586+gc8602c790.precompiled`. The runner reached communicator construction and CUDA graph capture before the snapshot failure. The prior NCCL-library identity check in the runner did not reject the mapped library; this run did not emit a final runtime manifest because it stopped before that print.

Raw rank logs, runner output, frozen scripts and dependencies were sealed privately. The archive SHA-256 was `853d3f276a3d6a11ebe68b3276a4b27ca8a72617ed4a1e309ceda845866872ce` on the host and after download; it has 22 members, with no absolute paths, traversal members or links. Do not publish its raw logs, communicator identifiers, clocks, PIDs or host paths. The only published event counts above came from a read-only parse of the retained logs. The run did not meet the scorer's input conditions, so no equality class or hypothesis outcome is reported.

The `shutdown -h now` command was issued after archive transfer and independent rehash. SSH reset immediately afterward. Cloud-platform power and billing state were **not** observed through that disconnect and require the owner's console confirmation.

## Closure

Per the preregistered one-booking rule, the script was not retuned or rerun on this instance. The NCCL clock-attribution candidate is parked as **unresolved**; no NVIDIA issue, Lab probe or stock-metric claim follows. Any future attempt requires a separately reviewed protocol whose pre-replay snapshot does not require graph-replay callback occurrences, plus a new explicit investment decision. This failure does not reopen the broader real-TP in-flight localization line.
