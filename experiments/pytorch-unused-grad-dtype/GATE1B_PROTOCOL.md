# Gate 1b protocol: rank-local assertion followed by a global timeout

Status: **pre-execution; no Gate 1b GPU result exists.**

This is a new diagnostic revision written after Gate 1 stopped on its first
affected timeout. It does not replace the original `PHASE2_FREEZE.json`, change
the Gate 1 result, or reinterpret the discarded raw output.

## Question

Does four-microbatch FSDP2 accumulation produce a rank-local dtype assertion
on rank 1 while rank 0 enters reduce-scatter and remains blocked until the
campaign wall-clock bound?

The question is deliberately per-rank. A trial-level `timeout` is expected only
when the structured rank records establish the earlier local assertion. A
timeout by itself is not evidence for the mechanism.

## Frozen rank matrix

Both arms use two identical GPUs, PyTorch `2.13.0+cu130`, one root FSDP group,
`param_dtype=bf16`, `reduce_dtype=fp32`, unused-parameter reduction, and four
microbatches. Synchronization is disabled for the first three backwards and
enabled for the final backward.

The prediction is source-derived: `to_accumulated_grad_if_needed` converts an
existing real gradient to `reduce_dtype`, the unused slot is created with
`zeros_like(unsharded_param)`, and the uniformity assertion runs before the
collective conversion. See PyTorch v2.13.0
[`_fsdp_param.py` lines 755-766](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/_fully_shard/_fsdp_param.py#L755-L766),
[`_fsdp_param.py` lines 878-880](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/_fully_shard/_fsdp_param.py#L878-L880),
and
[`_fsdp_collectives.py` lines 484-520](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py#L484-L520).

| Arm | Rank | Dtypes at `foreach_reduce` | Frozen outcome |
| --- | ---: | --- | --- |
| control | 0 | fp32 only | reduce returns; process completes |
| control | 1 | fp32 only | reduce returns; process completes |
| affected | 0 | fp32 only | enters reduce-scatter; does not return before the wall bound |
| affected | 1 | fp32 + bf16 | exact local uniform-gradient-dtype assertion |

The affected trial-level classification is
`rank1_assertion_rank0_wait`, and its launcher return code is recorded as 124.
The control classification is `completed_symmetric_fp32` with return code 0.

Each rank must emit exactly one reduce-entry record. Control must emit exactly
one reduce-return and one completed outcome per rank. Affected must emit no
reduce-return; its only terminal rank outcome must be rank 1's exact assertion.
Duplicate records fail closed rather than allowing the classifier to choose a
convenient call.

## Timing and capture contract

- process-group timeout: 30 seconds;
- external rank-stack capture: 20 seconds, before the process-group timeout;
- campaign wall bound: 45 seconds;
- stack-command bound: 8 seconds per live rank;
- cleanup grace: bounded by the shared lifecycle helper.

The runner sets `TORCH_NCCL_ASYNC_ERROR_HANDLING=3` because PyTorch's own
TorchTitan integration documents skip-cleanup mode as necessary for timeout
dumps. This is a declared diagnostic-setting change from Gate 1, not an
uncontrolled environment difference.

The 20-second stack capture is intended to observe rank 0 while it is still
blocked in the collective rather than after watchdog teardown has begun.
`watchdog_marker_seen` is recorded but is not an acceptance criterion because
message text is not a stable interface.

For every affected trial:

1. rank 0 must yield a parsed `py-spy --native --json` sample containing at
   least one frame from `gate1b_reproducer.py`;
2. at least one same-run ProcessGroupNCCL Flight Recorder dump must be decoded;
3. the bounded normalizer must expose a `missing_member` or `all_pending`
   secondary divergence;
4. raw stack, Flight Recorder and launcher output must not be persisted.

Flight Recorder pickle decoding is performed only on same-run files inside a
private temporary directory. Public results retain hashes, file sizes and the
allow-listed normalized summary. They do not retain the pickle or stderr.

Capture is a separate gate from mechanism. A mechanism match with a capture
failure supports the dtype/timeout sequence but does not support a claim that
the lab produced usable joined incident evidence.

## Lifecycle and stop rules

- Record launcher and rank identities with Linux PID start times.
- Always attempt bounded cleanup, including identity-discovery and runner
  failures.
- Require no tracked orphan for every retained trial.
- Persist each structured JSON before deciding whether to continue.
- Stop automatically after the first mechanism-classification mismatch.
- Do not widen markers, alter the rank matrix, or accept timeout alone after
  seeing output.
- Publish capture misses and unexpected watchdog behavior.

## Commands after review

```bash
CUDA_VISIBLE_DEVICES=0,1 python gate1b_campaign.py \
  --output /tmp/dtype-gate1b --trials 3

python verify_gate1b.py /tmp/dtype-gate1b --trials 3
```

Do not run the four-GPU matrix unless Gate 1b leaves a topology-specific
question that cannot be answered with this two-rank setup.
