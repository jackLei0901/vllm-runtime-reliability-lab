# Preemption cost

## Goal

Measure not only how often preemption occurs, but also its recomputation, queuing, and throughput cost.

## Minimum controls

- low-pressure workload: should not preempt
- boundary workload: intermittent preemption
- high-pressure workload: sustained preemption
- candidate: change only the scheduler or KV behavior under test

## Measurements

- change in `num_preemptions_total`
- running/waiting request timeline
- TTFT, TPOT, and E2E P50/P95/P99
- output throughput and success rate
- recomputed tokens when the code path exposes them; preemption count is not a substitute

## Interpretation

Eliminating a preemption storm does not guarantee every latency metric improves. A conservative stop-and-wait policy may reduce wasted state transitions while increasing individual request wait time. Report mechanism and user-visible metrics separately.

## Hardware result from 2026-08-27

For the same `8 x (7,000 input + 512 output)` workload, reducing the KV cache from 11.59 GiB to 6 GiB produced one preemption. Output throughput fell by about 38%, P99 TTFT was close to 4x, and mean TPOT was nearly unchanged. Each condition currently has one run, so this establishes the mechanism and collection path rather than a stable performance mean. See [`../../results/VALIDATION_SUMMARY_2026-08-27.md`](../../results/VALIDATION_SUMMARY_2026-08-27.md).
