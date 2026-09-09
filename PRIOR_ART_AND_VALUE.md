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

### A published gap this lab can test

The same PyTorch account says Flight Recorder analysis must be coupled with a
distributed view of CPU main-thread stacks to distinguish CPU work, barriers,
CPU-GPU synchronization and exception handling. It also states that PyTorch does
not currently provide this diagnostic tool and points to `py-spy` as one
possible telemetry source.

That is a bounded adjacent problem for this lab: collect per-process stack
snapshots externally, bind them to declared rank/process identities, and join
them with existing Flight Recorder artifacts after failure. The first experiment
must first determine whether a genuinely blocked NCCL rank can be sampled at all.
Only after that gate passes will controlled divergence compare the same stack/FR
inputs with and without linkage. It will not claim arbitrary hang detection or
add `py-spy` as a mandatory runtime dependency.

FR already identifies missing or mismatched ranks at a collective's logical
position. The added stack answers a narrower question: what was that rank's CPU
thread doing instead? Matching normalized frames can support some associations,
but it is not universal when the rank stalled before scheduling the missing
collective. Unsynchronized wall time is not used to infer which rank failed
first.

Raw stack output remains private by default and outside the current shareable
schema because it may expose source paths or application-specific symbols.

### What transfers to this project

- bounded per-producer capture before failure;
- best-effort local persistence;
- no required failure-time collective;
- explicit capture-coverage measurement;
- offline identity and sequence alignment;
- semantic comparison after syntactic linkage;
- optional external CPU main-thread stack capture correlated across ranks;
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
producer detection, hash verification and logical-mismatch localization. Human
diagnostic utility remains a separate
measure: hypotheses eliminated and time to the next action.
