# NCCL `pTimer` standalone replay: one-booking result

Chinese companion: [NCCL_PTIMER_STANDALONE_RESULT_2026-09-27.zh-CN.md](NCCL_PTIMER_STANDALONE_RESULT_2026-09-27.zh-CN.md). This follows the [frozen one-booking gate](NCCL_PTIMER_STANDALONE_ONE_BOOKING_GATE_2026-09-27.md) at Lab commit `c0bd1e0d5f10421ffe7d6f50de3c895950d492e2`. One eager cell and one graph cell were run, without retuning or repetition.

## Decision

The eager instrument control passed on both ranks. The standalone graph cell returned **`graph_reuse_not_observed`** on both ranks: six new, fully paired AllReduce occurrences per rank, with no equal START or STOP clocks among those occurrences. This is a negative result for this *small standalone workload*, not a refutation of the previous forced-PyNccl vLLM serving observation. The frozen interpretation requires an offline comparison of graph structure, callback coverage, and route before another GPU booking. It does not justify an NCCL issue, a stock-Inspector metric claim, an NCCL-core build, or a Lab probe.

## Identity and evidence

| Item | Recorded result |
| --- | --- |
| Host hardware | Two RTX 4090s. |
| Software | vLLM `0.1.dev586+gc8602c790.precompiled` from source revision `c8602c79062440074a018c1d5f875a5571eb6881`; Torch `2.13.0+cu130`; NCCL `2.29.7`, mapped-library SHA-256 `aa957cdfb91b516eae0d54a28e9ee5db52730d02e0ab45580efc3c19a68327a4` on both ranks/cells. The reused Inspector was built at NCCL `v2.29.7-1`. |
| Inspector binary | SHA-256 `ef734dc33acd60af9edc43b8ec84204c70e3d1fde53294478dc5b08e0d352c3b`, matching the prior booking. |
| Prior private archive | SHA-256 `17d3c0a687e2d7f555591fbdc54dcc7512421fb1bb76305895420ba0b7849d3e` on the host and after independent download. The archive member list had 64 entries and no absolute or traversal paths. |
| This booking's private archive | SHA-256 `fb393ef17415ff3d9c663b3d838c98c9760a3c10ac4f91aabeab0a16767dd757`, matching after independent download; 37 archived entries. Raw logs, clocks, communicator identifiers, PIDs, and host paths remain private. |

The driver verified the frozen source hashes before running either cell. Both cells used fresh two-rank processes and separate owner-only directories; the eager score gated entry into graph. The runner reported `numeric_result: pass` after checking each eager call and both graph replays. Private before/after snapshots and rank-log digests are retained in the archive. The published numbers below come from the frozen [standalone scorer](../../experiments/vllm-tp-dfx/ptimer_standalone_score.py), not manual selection of records.

| Cell | Rank 0 | Rank 1 | Outcome |
| --- | --- | --- | --- |
| Eager, three explicit AllReduces | Three new occurrences, three paired channels; largest equal-clock class 1 for START and STOP; zero ordering, zero-clock, or nonpositive-pair violations. | Same. | `eager_instrument_pass` |
| CUDA graph, three ungrouped AllReduces per replay, two checked replays | Six new occurrences, six paired channels; largest equal-clock class 1 for START and STOP; zero zero-clock or nonpositive-pair violations. | Same. | `graph_reuse_not_observed` |

The graph score does not apply the eager stream-order predicate; its recorded `eager_order_violations` is null, not zero. The standalone graph contains three direct AllReduce calls and bounded device sleeps. It is not the same graph, tensor shapes, callback volume, or serving route as the Qwen3 TP workload. The result therefore narrows the reproduction boundary instead of cancelling the serving observation.

## Post-hoc audit of the previous serving evidence

After independently checking the prior archive, the frozen [post-hoc auditor](../../experiments/vllm-tp-dfx/ptimer_posthoc_audit.py) read digest-pinned snapshots and unchanged rank logs. These counts were **not** preregistered scoring thresholds for the old campaign.

| Previous window | Rank 0 | Rank 1 |
| --- | --- | --- |
| Healthy serving, after minus before | 1,273 START and 1,273 STOP; largest equal-clock class 73 for each kind after grouping by communicator, channel **and function**; 14 of 2,546 new clocks below their pre-window group maximum. | 1,273 each; largest classes 78 START / 77 STOP; 14 of 2,546 below history. |
| Held serving, during minus before | 164 each; largest classes 58/58; 62 of 328 below history. | 148 each; largest classes 26/26; 14 of 296 below history. |

The old hold remains **`unscored / target_collective_ambiguous`**. These post-hoc counts support a route-scoped callback-clock attribution concern, but do not establish `base=0`, identify the target collective, or prove a stock timing/bandwidth metric wrong. The standalone non-reproduction means any proposed NCCL report needs a tighter reproducer or a source-grounded explanation of the serving/standalone difference.

## Next bounded work

Compare the retained serving and standalone graphs **offline first**: graph construction/replay path, number and ordering of captured collectives, tensor sizes and channel counts, callback timing/coverage, and which occurrence/function forms each repeated-clock class. Keep the analysis tied to the two pinned archives. If a concrete difference predicts clock reuse, preregister one follow-up reproducer before another GPU booking; otherwise record the result as unresolved rather than adding a probe or filing an issue.

The remote `shutdown -h now` command was issued after both archives were downloaded and rehashed; the SSH connection reset immediately afterward. Cloud-platform power/billing state requires separate confirmation and is not inferred from the disconnect.
