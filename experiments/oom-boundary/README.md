# OOM boundary map

## Goal

Separate "not enough GPU memory" into distinct regions: startup rejection, CUDA Graph capture OOM, runtime KV pressure, preemption degradation, and runtime OOM.

## Variables

- `max_model_len`
- `max_num_seqs`
- `max_num_batched_tokens`
- `gpu_memory_utilization`
- input/output length
- concurrency and arrival rate

Hold the model, dtype, quantization, attention backend, and CUDA Graph mode fixed. Change one group of variables at a time. Use binary search near a boundary instead of an unbounded Cartesian sweep.

## Minimum evidence per point

- startup outcome and peak startup GPU memory
- warmup outcome
- request success rate and error type
- KV-cache usage and preemption counter
- TTFT, TPOT, E2E latency, and throughput
- failure stage and the last incident snapshot

## Regions

| Region | Criterion |
| --- | --- |
| safe | no errors, no sustained preemption, stable latency |
| degraded | requests finish, but TTFT/TPOT or queuing degrades materially |
| preemption | preemptions keep increasing and recomputation cost is measurable |
| OOM | explicit OOM during startup, graph capture, or runtime |

OOMs from different stages must not be merged into one result.

## Hardware result from 2026-08-27

On one RTX 4090 with Qwen3-4B BF16, eager mode, and an 8K context, `gpu_memory_utilization=0.40` was rejected by the startup KV-capacity check. At 0.41, the server started and completed a 7,000-input + 512-output-token request. This locates a startup capacity boundary; it is not a runtime CUDA OOM. See [`../../results/VALIDATION_SUMMARY_2026-08-27.md`](../../results/VALIDATION_SUMMARY_2026-08-27.md).
