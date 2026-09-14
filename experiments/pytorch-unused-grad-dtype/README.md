# Two-rank FSDP2 unused-gradient dtype experiment

Status: **mechanism isolated; Gate 1e strict capture failed closed; Gate 1f
shutdown-stage diagnostic frozen but not executed.**

This experiment tests whether a simplified two-rank unused-parameter case is
sufficient to reproduce the mixed-gradient-dtype assertion observed during the
four-GPU organic-hang control. It cannot establish whether pipeline parallelism
is required.

## Arms and predeclared interpretations

| Arm | Rank behavior | Required result on affected PyTorch |
| --- | --- | --- |
| control | both ranks use both linear layers | completes |
| affected | rank 0 uses both; rank 1 skips the conditional layer | exact FSDP mixed-gradient-dtype assertion |

The pre-execution plan was to run three trials per arm on PyTorch 2.13.0+cu130,
then repeat on a current nightly without changing the scripts. The first lane's
declared expectation was `--expected-affected reproduced`. The first lane did
not reproduce, so the nightly lane is now deferred: a passing nightly cannot be
interpreted until the affected-version reproducer has a positive control.

## First-lane result (2026-09-12)

On two RTX 4090 GPUs with PyTorch 2.13.0+cu130, all three control trials and all
three affected trials completed. The predeclared `reproduced` verification
therefore failed, while a separate descriptive integrity check confirmed a
complete 3+3 `not-reproduced` matrix. This is a falsified hypothesis, not a
passing reproduction campaign.

The result shows that a single backward pass in plain two-rank FSDP2 is
insufficient to reproduce the four-GPU assertion. It does not identify which
remaining condition is necessary. The next experiment should isolate
the dtype mechanism before spending time on accumulation, four GPUs or a
nightly lane.

See `GPU_RESULT_2026-09-12.md` and the structured records under
`results/pytorch-unused-grad-dtype-20260912/`.

The frozen next-step order is documented in `PHASE2_PROTOCOL.md`. Its Gate 0
observes the exact gradient dtype list entering `foreach_reduce` and includes a
known mixed-dtype positive control. Accumulation and four-GPU runners are
prepared but must not run unless the preceding gate justifies them.

Gate 0 later confirmed that the ordinary unused-parameter path is uniformly
BF16 and that the probe detects a forced BF16+FP32 list. Gate 1e then reproduced
the accumulated-gradient rank-local assertion plus peer wait in three trials,
but only rank 0 produced a Flight Recorder dump. The strict two-rank capture
gate therefore failed closed. Gate 1f is a one-trial, pre-frozen diagnostic that
retains only allow-listed per-rank shutdown-stage flags and records the NCCL
version; it does not reinterpret Gate 1e. See `REVIEW_PHASE2_RESULTS_CN.md`.

## Environment

- Linux with `/proc`
- exactly two visible CUDA GPUs
- PyTorch with NCCL and
  `FSDPModule.set_reduce_scatter_unused_params`
- no vLLM checkout or model download

## Commands

```bash
export CUDA_VISIBLE_DEVICES=0,1
python experiments/pytorch-unused-grad-dtype/campaign.py \
  --output /tmp/unused-grad-torch-2.13 \
  --trials 3
python experiments/pytorch-unused-grad-dtype/verify_results.py \
  /tmp/unused-grad-torch-2.13 \
  --trials 3 \
  --expected-affected reproduced
```

The runner retains structured JSON only. Raw stdout and stderr are classified
in memory, hashed, and discarded. A timeout triggers PID-identity-checked
cleanup inherited from the organic-hang campaign.

## Decision boundary

- Reproduction in the affected arm plus clean controls shows that PP and
  microbatch accumulation are not required.
- Clean affected trials on a fixed nightly, with clean controls, show that the
  minimal case no longer reproduces there.
- An unexpected failure, timeout, incomplete PID record, or orphan is a failed
  campaign, not evidence for either conclusion.
- This experiment does not itself prove a production impact or propose a fix.
