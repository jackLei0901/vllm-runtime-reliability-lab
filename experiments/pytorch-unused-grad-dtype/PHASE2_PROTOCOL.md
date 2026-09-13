# Phase 2 protocol: isolate dtype before topology

Status: **design and scripts complete; no Phase 2 GPU result exists.**

This protocol was written after the 2026-09-12 two-rank null result. It does
not alter that result or its failed predeclared hypothesis.

## Question

Why did the four-GPU organic workload produce a mix of bf16 and fp32 gradients
at FSDP2's uniform-dtype assertion while the simplified unused-parameter model
completed?

The protocol tests mechanisms in increasing order of cost. It does not treat
GPU count, pipeline parallelism or accumulation as the cause until a narrower
gate supports that conclusion.

## Gate 0 — exact dtype mechanism, two GPUs

Run `dtype_probe_campaign.py` for three trials per case on the same PyTorch
2.13.0+cu130 environment:

| Case | Change | Frozen expectation |
| --- | --- | --- |
| `unused-parameter` | rank 1 skips one parameter group; unused reduction enabled | `foreach_reduce` sees bf16 only and the run completes |
| `forced-mixed-gradient` | one real gradient is intentionally produced in fp32 | the probe records bf16+fp32 and the exact assertion fires |

The probe wraps the Python symbol called by `FSDPParamGroup.post_backward` and
records only rank, gradient count and dtype names. It does not persist tensors,
parameter values or raw stderr.

Gate 0 passes only if all six trials match their frozen classifications, both
rank records are present, and no tracked process remains. This confirms both
the null's dtype mechanism and the assertion classifier against a positive
control.

If the unused-parameter case already contains mixed dtypes, stop: the current
mechanism hypothesis is wrong. Do not reinterpret a failure as support for
accumulation or topology.

## Gate 1 — pipeline-style accumulation, two GPUs

Run `campaign.py --execution-mode accumulated --work-units 4` for three control
and three affected trials. The affected arm keeps rank 1's conditional
parameters unused across all four microbatches. FSDP synchronization is disabled
for the first three backwards and enabled for the final backward using:

- `set_is_last_backward`;
- `set_requires_gradient_sync`; and
- `set_reshard_after_backward`.

Frozen expectations:

- control: 3/3 complete;
- affected: 3/3 exact mixed-gradient-dtype assertion.

If affected completes, accumulation in this simplified grouping is
insufficient. Stop and publish the null. Do not run a nightly lane.

## Gate 2 — original four-GPU topology matrix

Run only after Gate 0 passes and either Gate 1 reproduces or the remaining
question specifically requires the original pipeline implementation.

`prepare_four_gpu_matrix.py` derives all variants from the pinned upstream
reproducer. Every variant enables `set_reduce_scatter_unused_params(True)`.

| Case | Topology | Conditional behavior | Frozen expectation |
| --- | --- | --- | --- |
| `pp2-dp2-divergent` | PP=2, DP=2 | rank/microbatch-dependent | exact assertion |
| `pp1-dp4-divergent` | PP=1, DP=4 | rank/microbatch-dependent | completes |
| `pp2-dp2-uniform` | PP=2, DP=2 | conditional path always used | completes |

Run three trials per case, 200 training steps, DETAIL disabled, on one exact
PyTorch version and four identical GPUs. A timeout or unexpected failure is not
accepted as the expected assertion.

This matrix separates two questions:

- PP=2/DP=2 versus PP=1/DP=4 while preserving the divergent workload; and
- divergent versus uniform execution while preserving PP=2/DP=2.

It does not vary GPU count: every Gate 2 case uses four GPUs. The existing
two-GPU result is a separate experiment, not a matrix cell.

## Gate 3 — fixed nightly

Nightly testing is allowed only after an affected-version arm has a positive,
non-circular reproduction. Freeze the exact nightly wheel URL/version and the
expected outcome before execution. A nightly completion without a positive
affected-version control is uninterpretable and must not be called a fix.

## Stop rules

- Stop after any incomplete PID record or orphan.
- Stop after an unexpected exception; do not broaden the accepted marker.
- Do not change a case expectation after reading output.
- Do not add a fourth matrix cell to rescue a failed gate.
- Retain only structured summaries publicly; raw output remains excluded.

## Commands (not yet executed)

```bash
# Gate 0
CUDA_VISIBLE_DEVICES=0,1 python dtype_probe_campaign.py \
  --output /tmp/dtype-gate0 --trials 3
python verify_dtype_probe.py /tmp/dtype-gate0 --trials 3

# Gate 1
CUDA_VISIBLE_DEVICES=0,1 python campaign.py \
  --execution-mode accumulated --work-units 4 \
  --output /tmp/dtype-gate1 --trials 3
python verify_results.py /tmp/dtype-gate1 --trials 3 \
  --expected-mode accumulated --expected-affected reproduced

# Gate 2 source freeze and execution
python prepare_four_gpu_matrix.py --output /tmp/dtype-four-gpu-sources
CUDA_VISIBLE_DEVICES=0,1,2,3 python four_gpu_campaign.py \
  --prepared-dir /tmp/dtype-four-gpu-sources \
  --output /tmp/dtype-gate2 --trials 3
python verify_four_gpu.py /tmp/dtype-gate2 --trials 3
```
