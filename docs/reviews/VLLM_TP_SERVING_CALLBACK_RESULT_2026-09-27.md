# TP=2 serving callback and two-collective graph result (2026-09-27)

## Scope and identity

One dual-RTX-4090 host, vLLM source `c8602c79062440074a018c1d5f875a5571eb6881`, PyTorch `2.13.0+cu130`, NCCL `2.29.7`, and the lab-local Inspector plugin whose SHA-256 is `e49215462fb80f58d27c39155a5fc8008178e1b11a579c3ee95d259a1b81a315`. Both tests are producer-capability checks, not fault reproductions or reliability estimates. There was one successful run per cell.

The serving runner was frozen in commit `8237a18`; its SHA-256 is `4a6c7da440331addda9b32f4c32ddcbeccdefdfee3ccb8327d54d412883ebeaf`. The separate [two-collective runner](../../experiments/vllm-tp-dfx/two_collective_graph_gate.py) was frozen in commit `1063dd2`; its SHA-256 is `7c475c5a4e09c26e329bce6d8f1c6ab2a3513845c45634b4a1a8f2d8f720dc74`. The transfer copy of each runner was hash-checked before execution.

Raw NCCL debug logs, Inspector JSON, and complete process output remain private. The private archive was copied off the host before shutdown, and its SHA-256 matched at both ends: `176eac46216ff1e00cb8b4f729c51d01dcfff20ee66a56ed60c77b71f46c8743`. The serving and two-collective process-log SHA-256 values are respectively `639ba8cbb8335842d6818576d6e5ff14ae2a4d34c734591f646470bedd69837b` and `793fac4b013b6c2d6a838fd99e70090abb4921b77d8e2188c237e5280e34d77c`. No raw hostnames, PIDs, paths, or trace contents are published here.

## Serving cell

The exact-source Qwen3-4B-Instruct-2507-v2 run completed a 16-token generation at tensor parallel size 2. The startup log reported the TP all-reduce backend set as `['PYNCCL']` and reported FULL and PIECEWISE CUDA-graph capture. The runner took rank-bound callback snapshots **after** initialization/capture and again **after** generation:

| Rank | `CollStart` before → after | `KernelChStart` before → after |
| --- | ---: | ---: |
| 0 | 737 → 1921 (+1184) | 962 → 2235 (+1273) |
| 1 | 737 → 1921 (+1184) | 962 → 2235 (+1273) |

Both callback kinds increased on both ranks during real serving. This establishes that the profiler interface is usable on this TP=2 route. It does **not** identify which named layer or individual decode step caused a callback, establish that a particular step used FULL graph replay, or turn callback counts into completion/progress counts. No hang was injected in this cell.

## Separately scored graph cell

In a small PyNccl test, each rank captured two all-reduces in one CUDA graph. Rank 1 captured a bounded device sleep **between** them. Both host replay calls returned before the middle snapshot; the rank-1 completion event was pending before and after that snapshot. The replay's output tensors were checked after synchronization.

| Snapshot | Rank 0 (`CollStart`, `KernelChStart`) | Rank 1 (`CollStart`, `KernelChStart`) |
| --- | ---: | ---: |
| Before replay | (2, 2) | (2, 2) |
| During rank-1 delay | (4, 4) | (4, 3) |
| After completion | (4, 4) | (4, 4) |

Thus, in this one graph, the second rank-1 `CollStart` was present before its second `KernelChStart`. The rank-1 kernel-channel marker caught up after the delay. This supports interpreting `CollStart` as launch/issuance-level evidence and `KernelChStart` as a more specific in-flight kernel-start witness; it does not by itself prove the callback's implementation thread or that either marker means collective completion. The cell is synthetic, not a vLLM serving fault.

## Decision and next gate

No new Lab C++ probe is admitted. The existing NCCL profiler interface supplied a rank-specific kernel-start distinction that RAS replay counts did not. The remaining work is to determine whether an existing, privacy-bounded producer can export those in-flight facts during a **real** serving stall with a per-call route/graph witness. That is a new, separately preregistered gate, not a conclusion of this run. The host OS shutdown command was issued after the archive transfer; the SSH connection reset. Cloud control-plane power state was not independently observed.
