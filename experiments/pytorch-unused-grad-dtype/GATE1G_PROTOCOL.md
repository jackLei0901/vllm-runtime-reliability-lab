# Gate 1g: FSDP-free ProcessGroupNCCL reproducer

Status: **pre-execution**

Gate 1f showed that, on PyTorch 2.13.0 with NCCL 2.29.7, rank 1 stopped its
Flight Recorder responder before communicator destruction completed. Gate 1g
tests whether that diagnosability gap is reproducible without FSDP, model code,
mixed precision, unused parameters, or gradient accumulation.

This is a new experiment. It does not reinterpret Gate 1e or Gate 1f.

## Controlled changes

The affected arm uses two ranks in one ProcessGroupNCCL group:

1. both ranks first complete one same-size `all_reduce` followed by
   `torch.cuda.synchronize()`; this creates the communicator and its connections
   before the paths diverge;
2. rank 0 enqueues an asynchronous NCCL `all_reduce`, records that event through
   an atomic state marker, and waits on the work handle;
3. rank 1 waits until that marker exists, records that it observed the marker,
   raises a declared local exception, and enters `destroy_process_group()` from
   `finally`;
4. the external observer samples both rank stacks at 20 seconds and retains only
   project-frame summaries;
5. the ProcessGroupNCCL timeout is 30 seconds, the campaign wall bound is 60
   seconds, and `Work.wait()` has an explicit 180-second timeout.

The explicit `Work.wait()` timeout is required for CPU blocking. The no-timeout
form only adds a GPU-stream dependency and returns. The 180-second value is
deliberately above the campaign wall bound, so the observer remains in control
of termination.

The control arm runs the same asynchronous `all_reduce` on both ranks and then
performs normal teardown. No stack or Flight Recorder capture is expected.

## Frozen mechanism prediction

Control:

- both ranks record warm-up entry and return;
- both ranks record `all_reduce` enqueue and return;
- both ranks record `completed`;
- both ranks enter and return from teardown;
- the launcher exits zero.

Affected:

- both ranks record warm-up entry and return before divergence;
- rank 0 records enqueue but not return and is sampled in `work.wait()`;
- rank 1 records observation of rank 0's atomic marker, records the injected
  local failure, enters teardown but does not return, and is sampled in
  `destroy_process_group()`;
- the watchdog marker is observed and the launcher remains alive to the 60-second
  wall bound.

The marker orders the two user-space events. It does not claim that rank 1
participated in the collective.

## Frozen Flight Recorder and shutdown prediction

The affected arm is expected to retain one decodable rank-0 Flight Recorder
dump and no rank-1 dump. The rank-0 artifact must contain **exactly one** local,
non-completed `ALLREDUCE` entry for global group `[0, 1]`. The completed warm-up
entry must not be selected; zero or multiple pending candidates fail closed.
This is local rank-0 evidence only. The function never reads rank-1 entries when
constructing candidates, and the summary records
`peer_participation_inferred: false`.

The fixed per-rank ProcessGroupNCCL log flags are expected to match Gate 1f:

- rank 0 broadcasts the dump request successfully and writes its own dump, but
  has no shutdown-stage flags;
- rank 1 reaches `Watchdog joined, destroying NCCL communicators.` but not
  `Destroy complete.`, and neither observes the later dump request nor writes a
  dump.

The control must show all four normal shutdown stages on both ranks and no dump
flags.

## Fail-closed and privacy rules

- marker and library-log parsing failures are stored as bounded error codes;
- raw torchrun output, raw py-spy output, and raw Flight Recorder pickles are
  hashed and deleted with the temporary directory;
- the result is written before a declared mismatch returns non-zero;
- the control arm fails fast; an affected termination mismatch is recorded but
  does not erase the mechanism result;
- PID identity and orphan checks remain mandatory;
- the exact reproducer and all runtime dependencies are hash-pinned before GPU
  execution.

## Decision rule

A PASS requires both arms to satisfy the frozen mechanism, provenance,
lifecycle, privacy, stack, Flight Recorder, and per-rank log contracts. A match
supports reporting a general ProcessGroupNCCL diagnosability gap upstream: a
rank using the documented teardown path can retire its dump responder before a
later shutdown step blocks. It does not establish the exact NCCL internal call,
other PyTorch/NCCL versions, or a safe fix.

If the affected mechanism is established but rank 1 completes teardown, or the
launcher terminates before the wall bound, Gate 1g has **not** reproduced the
missing-rank dump gap without FSDP. That negative result does not invalidate
Gate 1f; it shows only that the stall depends on state created by the removed
workload, without identifying which part of that state is causal.

## Commands after review

```bash
python experiments/pytorch-unused-grad-dtype/verify_gate1g_freeze.py
python experiments/pytorch-unused-grad-dtype/gate1g_campaign.py \
  --output results/pytorch-unused-grad-dtype-gate1g-YYYYMMDD
python experiments/pytorch-unused-grad-dtype/verify_gate1g.py \
  results/pytorch-unused-grad-dtype-gate1g-YYYYMMDD
```
