# TP=2 serving callback and two-collective graph result (2026-09-27)

## Scope and identity

One dual-RTX-4090 host, vLLM source `c8602c79062440074a018c1d5f875a5571eb6881`, PyTorch `2.13.0+cu130`, NCCL `2.29.7`, and the lab-local Inspector plugin whose SHA-256 is `e49215462fb80f58d27c39155a5fc8008178e1b11a579c3ee95d259a1b81a315`. The serving runner explicitly set `disable_custom_all_reduce=True`; it tested a forced PyNccl route, **not** vLLM's default route on a P2P-capable host. Both tests are producer-capability checks, not fault reproductions or reliability estimates. There was one successful run per cell.

The [serving runner](../../experiments/vllm-tp-dfx/serving_callback_gate.py) and its [preregistration](VLLM_TP_SERVING_CALLBACK_PROTOCOL_2026-09-27.md) were frozen in commit `8237a18`; the runner's SHA-256 is `4a6c7da440331addda9b32f4c32ddcbeccdefdfee3ccb8327d54d412883ebeaf`. The separate [two-collective runner](../../experiments/vllm-tp-dfx/two_collective_graph_gate.py) was frozen in commit `1063dd2`; its SHA-256 is `7c475c5a4e09c26e329bce6d8f1c6ab2a3513845c45634b4a1a8f2d8f720dc74`. The transfer copy of each runner was hash-checked before execution.

Raw NCCL debug logs, Inspector JSON, and complete process output remain private. The private archive was copied off the host before shutdown, and its SHA-256 matched at both ends: `176eac46216ff1e00cb8b4f729c51d01dcfff20ee66a56ed60c77b71f46c8743`. The serving and two-collective process-log SHA-256 values are respectively `639ba8cbb8335842d6818576d6e5ff14ae2a4d34c734591f646470bedd69837b` and `793fac4b013b6c2d6a838fd99e70090abb4921b77d8e2188c237e5280e34d77c`. No raw hostnames, PIDs, paths, or trace contents are published here.

## Serving cell

The exact-source Qwen3-4B-Instruct-2507-v2 run completed a 16-token generation at tensor parallel size 2. The startup log reported the TP all-reduce backend set as `['PYNCCL']` and reported FULL and PIECEWISE CUDA-graph capture. The runner took rank-bound callback snapshots **after** initialization/capture and again **after** generation:

| Rank | `CollStart` before → after | `KernelChStart` before → after |
| --- | ---: | ---: |
| 0 | 737 → 1921 (+1184) | 962 → 2235 (+1273) |
| 1 | 737 → 1921 (+1184) | 962 → 2235 (+1273) |

Both callback kinds increased on both ranks during real serving. The `CollStart` delta is exactly `1184 = 16 × 74` per rank. A source-level candidate decomposition for each forward is 36 layers × two row-parallel reductions ([`o_proj`](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/model_executor/models/qwen3.py#L114) and [`down_proj`](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/model_executor/models/qwen2.py#L97)) = 72, plus one [vocabulary-embedding reduction](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/model_executor/layers/vocab_parallel_embedding.py#L506) and one [TP logits gather](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/model_executor/layers/logits_processor.py#L87-L96) = 74. The [published Qwen3 configuration](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blame/main/config.json) names 36 layers. Sixteen output tokens normally entail one prefill and fifteen decode forwards. The equality is **consistent with** that route and rules out an entirely absent NCCL path in these forwards; without per-call markers it does not prove the 74-way attribution, the exact logits branch used, or the graph mode of any individual step.

`KernelChStart` rose by 1273 per rank, 89 more than `CollStart`. It is a **channel-start** count, not a collective count: multi-channel collectives can produce multiple markers. Equality between ranks is informative in this cell, but a raw `KernelChStart` count cannot be converted into a collective position inside a serving graph without the collective descriptors and their channel counts. Neither callback kind means completion or token progress. No hang was injected in this cell.

## Separately scored graph cell

In a small PyNccl test, each rank captured two all-reduces in one CUDA graph. Rank 1 captured a bounded device sleep **between** them. Both host replay calls returned before the middle snapshot; the rank-1 completion event was pending before and after that snapshot. The replay's output tensors were checked after synchronization.

| Snapshot | Rank 0 (`CollStart`, `KernelChStart`) | Rank 1 (`CollStart`, `KernelChStart`) |
| --- | ---: | ---: |
| Before replay | (2, 2) | (2, 2) |
| During rank-1 delay | (4, 4) | (4, 3) |
| After completion | (4, 4) | (4, 4) |

Thus, in this one graph, the second rank-1 `CollStart` was present before its second `KernelChStart`. The rank-1 kernel-channel marker caught up after the delay. Both `CollStart` markers being present while the second kernel had not started shows that, here, `CollStart` witnesses graph-replay issuance rather than progress to each collective's device execution. The single-channel `KernelChStart` markers distinguish how far the device reached in this synthetic graph. This does not establish the callback's implementation thread, extend that one-channel positional inference to multi-channel serving collectives, or make either marker a completion signal. The cell is synthetic, not a vLLM serving fault.

## Decision and next gate

No new Lab C++ probe is admitted. The existing NCCL profiler interface supplied a rank-specific kernel-start distinction that RAS replay counts did not, but stock Inspector does not export incomplete collectives. The remaining question is whether an existing, privacy-bounded producer can **export** per-rank in-flight channel-start state during a real serving stall, with per-call route/graph and channel-count witnesses. If none can, the gap is export policy rather than acquisition. That is a new, separately preregistered gate, not a conclusion of this run.
