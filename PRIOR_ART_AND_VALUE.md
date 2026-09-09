# Prior art and value cases

[中文](PRIOR_ART_AND_VALUE.zh-CN.md)

This document records public before/after evidence for linked runtime data. It
is not a claim that training failure distributions transfer to inference
serving.

## PyTorch Flight Recorder

### Before

PyTorch describes NCCL watchdog timeout as a catch-all symptom. The rank that
first reports a timeout is rarely the culprit, the collective visible at timeout
may be downstream of the initiating fault, and the watchdog-thread stack omits
the main-thread scheduling context. The published account states that debugging
without Flight Recorder can take hours and can require a rerun with additional
debug settings.

Source: [Flight Recorder: A New Lens for Understanding NCCL Watchdog
Timeouts](https://pytorch.org/blog/flight-recorder-a-new-lens-for-understanding-nccl-watchdog-timeouts/).

### Correlation mechanism

Each rank keeps a bounded CPU-side record of collective type, lifecycle state,
dtype, size, call stack and a process-group-local sequence. Dumps are aligned
offline across ranks and process groups. The resulting mismatch is a relation:
a missing peer, different lifecycle state, incompatible operation metadata or
inconsistent arguments cannot be defined from one rank alone.

PyTorch uses a separate TCPStore-based control path to request best-effort dumps
from rank monitor threads. External orchestration later gathers the local files.
The authors report near-complete full-dump coverage in Meta's fleet and prefer
offline analysis because timeout state is already fragmented.

### After

One published case initially appeared to involve different `all_to_all`
variants. Cross-rank call-stack and scheduling-order alignment showed that some
ranks had advanced to a later collective while peers remained in the preceding
one. Linked evidence rejected the initial line of investigation and exposed the
execution divergence.

The public article does not report a controlled MTTR reduction or an avoided
GPU-hour total. This project therefore treats the case as qualitative evidence
that a join can correct a hypothesis, not as a numeric benefit baseline.

### Join-key lesson

PyTorch later documented that one sequence ID was not sufficient to align
truncated buffers when process groups mix collectives and point-to-point
operations. That issue proposes additional identity so an analyzer can recover
a reference frame after the beginning of execution has fallen out of the ring.

Source: [pytorch/pytorch
#125173](https://github.com/pytorch/pytorch/issues/125173).

### What transfers to this project

- bounded per-producer capture before failure;
- best-effort local persistence;
- no required failure-time collective;
- explicit capture-coverage measurement;
- offline identity and sequence alignment;
- semantic comparison after syntactic linkage;
- hypothesis elimination as a useful outcome.

### What does not transfer

- Meta's training failure percentages do not describe inference serving;
- inference replicas and roles do not follow symmetric SPMD execution in every
  topology;
- external process/GPU observations cannot reproduce c10d collective metadata;
- a missing producer or state mismatch narrows categories but does not always
  identify one root cause;
- near-complete dump coverage in Meta's environment is not evidence for this
  implementation.

## Product implication

The first v0.2 value test will use identical underlying producer artifacts in
two arms:

1. unlinked files without topology, clock alignment or a normalized view;
2. a closed manifest plus the vLLM process/progress semantic join.

The join must create checkable relationship facts rather than merely collect
more files. Structural measures are capture coverage, join coverage, missing
producer detection, hash verification and first-observed-divergence accuracy
where clock precision permits. Human diagnostic utility remains a separate
measure: hypotheses eliminated and time to the next action.
