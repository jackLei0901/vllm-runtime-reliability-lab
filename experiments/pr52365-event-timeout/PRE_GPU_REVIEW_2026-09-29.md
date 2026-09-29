# PR #52365: pre-GPU configuration review

Status: **not booked after offline feasibility review**, 2026-09-29. This is not an amendment to A1 and no vLLM cell has run. The [Chinese companion](PRE_GPU_REVIEW_2026-09-29.zh-CN.md) records the same decision.

## Reviewed but unused 24 GiB single-GPU cell

- Model: `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`. Its [published config](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/config.json) declares BF16, 36 layers, 8 KV heads, head dimension 128, and a native 262,144-token context. The [model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) documents vLLM serving. The Lab has served this architecture before, but not at these prompt lengths or the #52365 pins.
- `--max-model-len 65536`, `--gpu-memory-utilization 0.9`, one GPU; no quantization, context override, artificial sleep, or modified vLLM source.
- Fixed sweep ladder: `8192 16384 24576 32768 40960 49152 57344 65535`. The last prompt plus one output token fits 65,536. A first >=70 s screen at the final step is `no_candidate`, not permission to extend the ladder. The runner stops after exactly one step beyond the first screen.
- Memory lower bound, not a fit guarantee: BF16 parameters are about 7.49 GiB ([4,022,468,096 parameters](https://huggingface.co/api/models/Qwen/Qwen3-4B-Instruct-2507) x 2 bytes); full-context BF16 KV cache is 9 GiB (`36 layers x 2 x 8 KV heads x 128 x 2 bytes x 65,536`). At 90% of 24 GiB, this leaves about 5.1 GiB for CUDA context, activations, graph capture, workspaces and other allocations. The large `--max-num-batched-tokens 65536` may make startup fail despite the lower bound. An 81,920-token context would require 11.25 GiB of KV cache and leave only about 2.9 GiB, so it is *not* a silent fallback on a 24 GiB card.

## Compute screen and decision

For the final 65,535-token rung, a coarse prefill estimate is about `2 x 4.022e9 parameters x 65,536 = 5.27e14` linear FLOPs plus `2 x 65,536^2 x (32 query heads x 128) x 36 = 1.27e15` causal-attention FLOPs, or **1.79e15 FLOPs total**. The linear term is deliberately approximate: not every parameter is multiplied at every prompt token. NVIDIA lists [165.2 TFLOPS peak BF16 Tensor compute with FP32 accumulation for the RTX 4090](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf). At hypothetical sustained rates of 130, 100 and 50 TFLOPS, the arithmetic gives about 14, 18 and 36 seconds. Reaching a 70-second whole-request screen from these FLOPs alone would require about 25.6 TFLOPS sustained; 80 seconds would require about 22.4 TFLOPS.

This is **source-and-arithmetic reasoning, not a latency measurement or a hard upper bound**. Effective throughput, memory traffic, graph/capture overhead and implementation details are unknown, and a request's wall time is not its `wait_for_gpu_event` time. Nevertheless, the estimate provides no credible margin for A1's completed **single wait >=65 s** on this 24 GiB cell. The additional risk of startup OOM makes a three-hour booking poor value. **Decision: do not book this cell; predict `no_candidate`, but do not record an actual Stage-1 outcome.** This decision does not show that the PR's default timeout is harmless for all workloads.

Reopen GPU consideration only for a *different, naturally occurring* single-step configuration whose pre-booking source/shape and hardware estimate reaches roughly **80 seconds or more**, whose native context and weight/KV/activation budget fit the chosen GPU, and whose purpose is worth the cost. Write and commit a new, fixed model/hardware/length selection before booking. Do not use shared or throttled hardware to manufacture a long wait, nor modify A1's scoring after observing data. If no such configuration is available, keep #52365 as a source-backed ordinary review, not a Lab delivery.

Had this cell run, A1 would have required `no_candidate` if the fixed ladder found no completing request with the required margin. The code and command below are retained as an **unused proposal**, not evidence that any stage completed. Even a whole request over 70 seconds would only have screened for the measured per-call wait in the disabled PR arm.

If a future, separately approved review reopens this exact cell, the proposed Stage 1 invocation would be:

```bash
python experiments/pr52365-event-timeout/runner.py sweep \
  --python /path/vllm-base/.venv/bin/python --vllm-src /path/vllm-base \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --model-revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --max-model-len 65536 \
  --lengths 8192 16384 24576 32768 40960 49152 57344 65535 \
  --work-dir /private/p52365/sweep
```

Stage 2 is conditional on `candidate_found`; use exactly the `short` and `long` lengths in that receipt with the README's `ab` command. No manual choice among successful lengths is allowed.

## Conditions if a future booking is proposed

1. Confirm the actual GPU memory class. If it is not 24 GiB, select and commit a separate configuration **before** opening the instance; do not treat this ladder as transferable. Confirm that one GPU, not TP=2, is allocated.
2. Confirm the exact model revision remains available, the full BF16 weights are cached or can be downloaded within the three-hour total cap, and that the host has adequate free disk/RAM. Do not print tokens or credentials.
3. On the host, record `nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader`, `command -v ninja`, `df -h`, `free -h`, Python version, and the base/PR checkout SHAs. The runner's own identity and pre-existing `sitecustomize` checks remain hard gates; do not work around them after data appear.
4. Commit this chosen configuration before any GPU booking. Confirm the Lab commit is public and that the runner, hook, tests and A1 are the versions reviewed. Then use a fresh private work directory for each stage and keep server logs/traces private.
5. Budget one session, at most three hours including installation. If startup cannot fit the fixed model/configuration, record an apparatus `unscored` outcome and stop. Do not turn an OOM into a smaller, post-hoc configuration.

No GPU booking is requested or approved by this review.
