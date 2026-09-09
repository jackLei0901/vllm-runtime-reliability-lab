# Soak test

## Goal

Check whether a sustained mixed workload produces memory growth, latency drift, throughput decay, handle leakage, or periodic stalls.

## Suggested protocol

1. Fix the model, server arguments, workload seed, and request set.
2. Warm up CUDA Graphs, allocators, and caches.
3. Run two hours for a smoke soak; use at least six hours for a long-run claim.
4. Collect the runtime timeline every second and aggregate latency/throughput each minute.
5. Preserve allocator-trim measurements when applicable, GPU memory, and the process tree at shutdown.

## Interpretation

- RSS growth is not automatically a leak; correlate it with PSS, Private Dirty, allocator trimming, and live objects.
- Stable GPU reserved memory with changing used memory is not sufficient evidence of a GPU-memory leak.
- Start/end points alone hide periodic sawtooth behavior; retain the full time series.
- Exclude warmup from trend fitting and report the window, peak, and final plateau with any slope.

## Hardware result from 2026-08-27

A ten-minute smoke soak completed 1,200 requests without failure. GPU memory stayed at 20,445 MiB because vLLM preallocated the KV pool; no visible latency drift or preemption occurred. The short window did not establish a reliable long-term RSS trend and does not replace the planned 2–6-hour run. See [`../../results/VALIDATION_SUMMARY_2026-08-27.md`](../../results/VALIDATION_SUMMARY_2026-08-27.md).
