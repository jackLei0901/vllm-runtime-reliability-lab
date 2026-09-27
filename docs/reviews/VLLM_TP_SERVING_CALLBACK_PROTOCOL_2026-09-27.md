# TP=2 serving callback gate — preregistered check (2026-09-27)

Scope: one Qwen3-4B serving run at vLLM `c8602c79062440074a018c1d5f875a5571eb6881`, two RTX 4090s, PyTorch `2.13.0+cu130`, NCCL `2.29.7`, and the already pinned lab-local Inspector build. This is a capability check, not a hang reproduction or reliability estimate.

The runner [serving_callback_gate.py](../../experiments/vllm-tp-dfx/serving_callback_gate.py) takes one snapshot of rank-bound `CollStart` and `KernelChStart` debug-marker counts **after engine initialization and graph capture**, then generates 16 tokens, then snapshots again before shutdown. A rank log must declare its NCCL rank; duplicate or missing ranks fail closed. Only counts and deltas may be published. Raw logs stay private and are identified by digest.

The scored claim is narrow: both markers increasing on both ranks during generation, together with a logged `PYNCCL`-only TP backend set and FULL graph capture, is evidence that the profiler interface sees collectives in a real serving workload. It does **not** by itself prove that a specific decode step used FULL replay, that the markers belong to one named model-layer all-reduce, or that a fault can be attributed in serving. Those require a per-call route/graph witness. Zero delta, ambiguous rank binding, or a failed generation is `unscored`, not evidence of an absent collective.

The follow-up two-collective graph cell is separately gated: it must place a bounded device sleep between two captured collectives and sample callbacks while that sleep is pending. The question is whether the second `CollStart` precedes its second `KernelChStart`; it must not be merged into the serving result.
