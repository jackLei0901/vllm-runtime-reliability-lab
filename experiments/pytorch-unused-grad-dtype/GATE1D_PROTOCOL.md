# Gate 1d: three repeated mechanism observations with independent termination

Status: **pre-execution**

Gate 1d supersedes the withdrawn, never-executed Gate 1c campaign contract. The
Gate 1c mechanism matrix is retained unchanged; the wall bound and stop rule are
corrected before any GPU result exists.

## Question and mechanism matrix

Can a two-rank accumulated-gradient run retain this sequence three times?

| Arm | Rank | Reduce dtype list | Reduce return | Barrier | Outcome |
| --- | ---: | --- | --- | --- | --- |
| control | 0 | all FP32 | yes | enter and return | completed |
| control | 1 | all FP32 | yes | enter and return | completed |
| affected | 0 | all FP32 | yes | enter, no return | none before cleanup |
| affected | 1 | FP32 + BF16 | no | not entered | exact uniformity assertion |

The mechanism classification remains
`rank1_assertion_rank0_barrier_wait`. Duplicate or missing marker records fail
closed. The runner stops after a mechanism mismatch, because later repetitions
cannot repair a failed predeclared mechanism.

## Independent termination observation

The affected termination prediction remains
`wall_bound_rank1_teardown_wait`: rank 1 enters `destroy_process_group()` without
returning, rank 0 remains before teardown, and the launcher reaches the external
wall bound.

This prediction is less certain than the source-grounded mechanism. A
termination mismatch is retained and reported, but it does not stop the
campaign. All three affected trials run when the mechanism continues to match.
The verifier recomputes each termination class from the markers and reports the
number of prediction mismatches separately.

Recognized alternatives include torchrun teardown, watchdog teardown and a wall
bound with a different teardown-marker pattern. None is silently relabeled as
the frozen prediction.

## Timing and Flight Recorder margin

- concurrent rank-stack capture starts at T+20 seconds;
- each py-spy subprocess has an eight-second bound;
- process-group timeout is 30 seconds from the queued work;
- external wall bound is 60 seconds;
- trials are control x3 followed by affected x3.

The previous 45-second wall bound left only a small margin after startup,
microbatches, the 30-second NCCL timeout and dump-store polling. Sixty seconds
allows both plain-file pickle writes to finish while remaining below the
approximately 90-second earliest rank-0 watchdog teardown expected for this
sequence.

The capture gate remains strict. Both rank dumps must exist and decode before a
missing rank at one collective sequence can mean non-participation. The matching
entry must be a non-completed rank-0 `REDUCE_SCATTER` in group `[0, 1]`. Missing
dump ranks are recorded separately. The rank stacks must show rank 0 at
`dist.barrier()` and rank 1 at `dist.destroy_process_group()`.

## ptrace portability

Preflight distinguishes two modes:

- `pr_set_ptracer_parent`: the Yama `ptrace_scope` file exists, its value is
  recorded, and each rank authorizes the campaign parent and its descendants;
- `yama_absent`: the Yama file does not exist and `PR_SET_PTRACER` is not called.

Each rank emits its selected mode, and the verifier requires both rank records
to agree with preflight. Permission denial remains an explicit capture failure.

## Source identity and privacy

Each result stores the SHA-256 of `gate1d_reproducer.py`. Before deriving stack
line numbers, the verifier hashes the current file and requires an exact match.
An edit after execution therefore fails rather than silently moving the expected
stack locations.

Raw torchrun output, py-spy JSON and Flight Recorder pickle files remain in the
temporary directory and are deleted. Retained files contain structured markers,
bounded project frames, hashes, derived Flight Recorder summaries and lifecycle
results.

## Frozen source basis

- [`_fsdp_param.py` accumulated-gradient conversion](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/_fully_shard/_fsdp_param.py#L802-L813)
- [`_fsdp_param_group.py` unused-gradient path](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py#L631-L636)
- [`_fsdp_param.py` zero placeholder](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/_fully_shard/_fsdp_param.py#L931-L932)
- [`_fsdp_collectives.py` uniformity check](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py#L552-L558)
- [`_fsdp_collectives.py` reduce-scatter call](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py#L182-L189)

## Stop and interpretation rules

- stop and retain the first mechanism mismatch;
- do not stop for a termination-prediction mismatch;
- never interpret a single-rank dump as collective non-participation;
- mechanism success without capture success is not joined incident evidence;
- publish misses and lifecycle failures;
- do not expand to four GPUs to reinterpret this two-GPU contract.

Gate 1d does not establish root cause, general hang diagnosis, vLLM hot-path
coverage, cross-host behavior, production safety or unknown-issue resolution.

## Commands after review

```bash
python experiments/pytorch-unused-grad-dtype/verify_gate1d_freeze.py
python experiments/pytorch-unused-grad-dtype/gate1d_campaign.py \
  --output results/pytorch-unused-grad-dtype-gate1d-YYYYMMDD
python experiments/pytorch-unused-grad-dtype/verify_gate1d.py \
  results/pytorch-unused-grad-dtype-gate1d-YYYYMMDD
```

