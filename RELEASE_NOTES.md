# v0.1.0-alpha.4

This release adds the first lab case that was not designed by the project: a
four-GPU reconstruction of the known FSDP2 conditional-parameter hang tracked
by pytorch/pytorch#158719 and pytorch/torchtitan#2747.

## Delivered evidence

- One host with 4 x RTX 4090, PP=2 and DP=2.
- Three PyTorch 2.11 `DebugLevel.DETAIL` oracle trials reported a stable
  `_REDUCE_SCATTER_BASE` input-shape mismatch (`[1024]` versus `[960]`).
- Three normal hung runs with automatic ProcessGroupNCCL Flight Recorder
  capture reconstructed the same semantic mismatch in 3/3 trials.
- The public verifier checks the expected stage-0 DP group `[0, 2]`, uniform
  dtype family, completed collectives before the hang, and every retained
  lifecycle cleanup record.
- Raw stderr, prepared source and private audit material are excluded from the
  public result directory.

## Decision boundary

This is a known-answer reconstruction, not discovery of an unknown root cause,
and the campaign remains short of GO. A same-version no-divergence control is
still required. The attempted PyTorch 2.13 opt-in control hit a separate
mixed-gradient-dtype assertion in all three trials and was recorded as blocked,
not counted as evidence of success. Protocol revision `2026-09-12.6` was
untracked at execution time, so its freeze timing remains author-declared.

The result validates an external c10d-visible evidence workflow. It does not
validate the proposed vLLM EngineCore incident Snapshot or establish its
diagnostic utility.

## Review entry

- [Review entry](https://github.com/jackLei0901/vllm-runtime-reliability-lab/blob/v0.1.0-alpha.4/experiments/organic-hang/REVIEW_RESPONSE_2026-09-10.md)
- [GPU result](https://github.com/jackLei0901/vllm-runtime-reliability-lab/blob/v0.1.0-alpha.4/experiments/organic-hang/GPU_RESULT_2026-09-12.md)
- [Derived-only public evidence](https://github.com/jackLei0901/vllm-runtime-reliability-lab/tree/v0.1.0-alpha.4/results/organic-hang-20260912)

Wheel SHA-256:

```text
67fb31e293901b3e9f6159ff0d5edaa9b9f916bad3d7862697b01cb7f5fed0e4
```

Derived-only evidence archive SHA-256:

```text
a7d0e4d3c90324f7e5d9c35f9f3e25ef43d7da1a56c65b4bb514f35172e820f8
```
