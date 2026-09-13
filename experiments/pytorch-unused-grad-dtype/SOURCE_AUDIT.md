# Phase 2 source audit

Checked against PyTorch v2.13.0 on 2026-09-12.

## Assertion boundary

In `_fsdp_param_group.py:574-594`, `post_backward` appends one of three values for each
trainable FSDP parameter: an accumulated unsharded gradient, a current
unsharded gradient, or `unsharded_zero_grad_data` when unused-parameter
reduction is enabled.

In `_fsdp_param.py:878-880`, `unsharded_zero_grad_data` is produced by:

```python
torch.zeros_like(self.unsharded_param)
```

With `param_dtype=torch.bfloat16`, this placeholder is bf16.

In `_fsdp_collectives.py:484-520`, `foreach_reduce` constructs the set of input
gradient dtypes and raises before allocating/copying into the `reduce_dtype`
buffer when the set has more than one member. The exact v2.13.0 marker is:

```text
FSDP reduce-scatter expects uniform gradient dtype but got
```

This matches `ASSERTION_MARKER` in every campaign classifier.

Primary sources:

- https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/_fully_shard/_fsdp_param.py#L878-L880
- https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py#L574-L594
- https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py#L484-L520

## Independent positive control

Garrett Goon published a two-rank FSDP2 reproducer that intentionally produces
one bf16 gradient and one fp32 gradient in the same FSDP group. Its captured
output shows the same assertion on both ranks. Phase 2's
`ForcedMixedGradientModel` adapts only that dtype-producing mechanism and
credits the source in code; it does not claim the gist is evidence about the
organic pipeline bug.

- source gist: https://gist.github.com/garrett361/c2338998a96ad2ab73aeec61d458e541

## Claim boundary

The source audit explains why the simplified unused-parameter case is expected
to remain uniform when all real gradients are bf16. It does not establish what
caused fp32 real gradients in the original four-GPU workload. Gate 0 measures
the simplified case; later gates test accumulation and topology only if needed.
