# Gate 1c: per-rank assertion, barrier wait and teardown capture

Status: **pre-execution**

Gate 1c supersedes the withdrawn, never-executed Gate 1b contract. It does not
overwrite the Gate 1b freeze. No GPU result may be attributed to Gate 1c until
the new freeze and independent verifier both pass against retained results.

## Question

Can a two-rank accumulated-gradient run retain enough bounded evidence to
separate this sequence?

1. rank 1 reaches FSDP's uniform-gradient-dtype assertion before issuing its
   reduce-scatter;
2. rank 0 issues the reduce-scatter, returns from the CPU call and later waits
   in `dist.barrier()`;
3. rank 1 enters process-group teardown and may remain there until the external
   wall bound;
4. two complete Flight Recorder dumps show a pending rank-0 reduce-scatter with
   no corresponding rank-1 entry.

The mechanism, termination and evidence-capture claims are scored separately.
A termination mismatch does not retroactively change a mechanism observation.

## Source-grounded mechanism

The PyTorch 2.13 source establishes the order being tested:

- accumulated real gradients may be converted to the reduce dtype before the
  final synchronized backward;
- an unused slot is populated with `zeros_like(unsharded_param)`;
- the uniform-dtype assertion runs before the collective;
- the NCCL reduce-scatter is issued with `async_op=False`, but the CPU call may
  return after installing stream ordering rather than waiting for remote
  completion.

Pinned source references:

- [`_fsdp_param.py` accumulated-gradient conversion](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/_fully_shard/_fsdp_param.py#L802-L813)
- [`_fsdp_param_group.py` unused-gradient path](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py#L631-L636)
- [`_fsdp_param.py` zero placeholder](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/_fully_shard/_fsdp_param.py#L931-L932)
- [`_fsdp_collectives.py` uniformity check](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py#L552-L558)
- [`_fsdp_collectives.py` reduce-scatter call](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py#L182-L189)

## Frozen mechanism matrix

| Arm | Rank | Reduce dtype list | Reduce return | Barrier | Outcome |
| --- | ---: | --- | --- | --- | --- |
| control | 0 | all FP32 | yes | enter and return | completed |
| control | 1 | all FP32 | yes | enter and return | completed |
| affected | 0 | all FP32 | yes | enter, no return | no terminal outcome before cleanup |
| affected | 1 | FP32 + BF16 | no | not entered | exact local uniformity assertion |

The affected mechanism classification is
`rank1_assertion_rank0_barrier_wait`. Duplicate or missing marker records fail
closed.

## Frozen termination prediction

| Arm | Expected launcher termination | Teardown markers |
| --- | --- | --- |
| control | natural exit 0 | both ranks enter and return |
| affected | 45-second external wall bound | rank 1 enters, no return; rank 0 does not enter |

The affected termination classification is
`wall_bound_rank1_teardown_wait`. This is a separate, falsifiable prediction
about the unexplained Gate 1 timeout. If torchrun or the NCCL watchdog ends the
job first, the mechanism result remains separately reportable, but this
termination gate fails and the campaign stops after retaining that trial.

`TORCH_NCCL_ASYNC_ERROR_HANDLING=3` remains explicit for reproducibility. Mode 3
is already the PyTorch default in this environment; it is not described as a
new diagnostic-setting change.

## Capture contract

At 20 seconds, before the 30-second process-group timeout, the campaign starts
`py-spy dump --native --json` concurrently for both recorded rank identities.
Each subprocess has an eight-second bound, so one slow attach cannot delay the
other rank's capture until after the process-group timeout.

Expected affected-arm stack sites:

- rank 0: the `dist.barrier()` call in `gate1c_reproducer.py`;
- rank 1: the `dist.destroy_process_group()` call in the same file.

The Flight Recorder join may claim non-participation only when both rank dumps
exist and decode. A passing candidate must have:

- group members exactly `[0, 1]`;
- only rank 0 present at that collective sequence;
- an operation containing `REDUCE_SCATTER`;
- rank 0 state other than `completed`.

Missing dump ranks are recorded separately. A single dump can never satisfy
this capture gate, even if the generic normalizer reports `missing_member`.

## ptrace and privacy boundary

Preflight records the effective UID and Linux `ptrace_scope`. Each rank calls
`PR_SET_PTRACER` for the campaign parent; the spawned py-spy process is its
descendant. Permission denial remains an explicit capture failure.

Raw torchrun output, py-spy JSON and Flight Recorder pickle files remain inside
the temporary directory and are deleted. Retained files contain only structured
markers, bounded stack frames from the reproducer, hashes, derived Flight
Recorder summaries and lifecycle results.

## Timing and stop rules

- stack capture: T+20 seconds;
- process-group timeout: 30 seconds;
- external wall bound: 45 seconds;
- trials: control x3, affected x3;
- stop after the first mechanism or termination expectation mismatch;
- retain the mismatching structured result;
- do not run a four-GPU matrix to reinterpret a failed two-GPU contract.

## Non-claims

Gate 1c does not establish root cause, general hang diagnosis, PyTorch version
scope, vLLM hot-path coverage, cross-host behavior or production safety. It
tests one source-grounded two-rank sequence and whether the lab can preserve the
evidence needed to distinguish its local assertion, later CPU wait and teardown.

## Commands after review

```bash
python experiments/pytorch-unused-grad-dtype/verify_gate1c_freeze.py
python experiments/pytorch-unused-grad-dtype/gate1c_campaign.py \
  --output results/pytorch-unused-grad-dtype-gate1c-YYYYMMDD
python experiments/pytorch-unused-grad-dtype/verify_gate1c.py \
  results/pytorch-unused-grad-dtype-gate1c-YYYYMMDD
```
