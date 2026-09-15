# Fix feasibility for pytorch/pytorch#196968

Status: **design review only; no implementation is proposed yet**

This note evaluates possible ways to preserve Flight Recorder evidence from a
rank that has entered legacy `ProcessGroupNCCL::shutdown()` while a peer later
requests a timeout dump. It is a local review artifact, not an upstream fix
proposal.

## Evidence boundary

The standalone two-rank reproducer established the following on PyTorch 2.13
with NCCL 2.29.7 and on the selectable legacy backend in a current nightly with
NCCL 2.30.7:

1. rank 1 raises locally and enters `destroy_process_group()`;
2. rank 1 logs `Watchdog joined, destroying NCCL communicators.` but not
   `Destroy complete.`;
3. about 30 seconds later, rank 0 times out, successfully broadcasts the dump
   signal, and writes its own Flight Recorder dump;
4. rank 1 neither observes the signal nor writes a dump before the external
   bound.

This proves a diagnostic availability gap. It does not identify the exact NCCL
call in which rank 1 blocks. `ncclCommDestroy` remains a source-based inference,
not an observation.

Upstream issue:
[pytorch/pytorch#196968](https://github.com/pytorch/pytorch/issues/196968).

## Current-source audit

Audited on 2026-09-15 against PyTorch `main` commit
`bea1fae02008a8483354d6fdccadd2067f1702c6`.

The relevant ordering in legacy `ProcessGroupNCCL::shutdown()` remains:

1. finalize communicators and wait for outstanding operations;
2. stop and join the watchdog;
3. stop the heartbeat monitor at `ProcessGroupNCCL.cpp:1592`;
4. destroy communicators at lines 1594-1601.

The heartbeat monitor polls the global store for a peer's dump signal at lines
1865-1931. Its `stop()` method sets
`terminateHeartbeatMonitorThread_` and wakes the thread, causing the loop to
return before another poll.

The responder owner and ordinary all-group shutdown order are already defined
by the current source. Only the monitor whose process-group UID is zero polls
for the global dump signal (`checkDumpSignal = dumpOnTimeoutOrEx_ &&
pg_->getUid() == 0`, line 1826). Python's `destroy_process_group()` shuts down
all process groups in reverse name order, leaving the default group, and thus
that process-level responder, until last. A proposed fix should preserve this
single-responder ownership rather than adding one store poller per group.

The monitor is not a dump-only listener. The same loop also:

- checks the watchdog heartbeat and may classify a watchdog hang;
- handles local and remote dump requests;
- performs Flight Recorder, GIL, and C++ stack diagnostics;
- may terminate the process after an unclean-shutdown diagnosis.

Therefore, merely moving `heartbeatMonitor_->stop()` below communicator
destruction would keep unrelated monitor behavior active during teardown and
needs a concurrency and lifecycle review.

## Required behavior contract

During legacy `ProcessGroupNCCL` shutdown, a live rank should retain a bounded
ability to answer an already-configured peer `exception_dump` request until one
of these terminal conditions occurs:

- communicator teardown completes;
- the process exits or is externally terminated;
- the global store becomes unavailable;
- a declared diagnostic deadline expires.

Retaining that ability must not imply that the monitor may continue every
normal watchdog action. In particular, a shutdown-time diagnostic path must
not race communicator destruction by aborting, querying, or otherwise using a
communicator whose teardown is in progress.

## Non-goals

- Identifying or fixing the underlying NCCL teardown stall.
- Changing the default `nccl2` backend.
- Guaranteeing a dump after the process or store has disappeared.
- Turning normal `destroy_process_group()` calls into unconditional dump
  events.
- Inferring peer participation from the presence or absence of one rank's
  artifact.

## Design options

### Option A: explicit shutdown dump-only mode

Before communicator destruction, transition the monitor into a state that:

- continues polling only for the global dump signal;
- suppresses watchdog-heartbeat timeout classification;
- suppresses communicator abort or other communicator-dependent actions;
- permits one bounded, idempotent Flight Recorder dump;
- exits when communicator destruction completes or a diagnostic deadline is
  reached.

After successful communicator destruction, `shutdown()` stops and joins the
monitor as it does today.

This is the most direct expression of the required contract, but it is not yet
an implementation plan. The existing post-loop code performs more than writing
the Flight Recorder artifact, including GIL checks, C++ stack dumping, waiting,
and possible process termination. A safe implementation must define which of
those actions remain valid in dump-only mode.

The communicator-dump behavior is build-dependent. On standard CUDA builds,
`getNCCLCommDumpMap()` returns an empty map and the Flight Recorder dump does
not call into each communicator. On ROCm or NCCLX builds compiled with
`NCCL_COMM_DUMP`, it iterates over every communicator and invokes
`ncclCommDump()`. Dump-only mode must therefore suppress the communicator
portion of the dump while communicator destruction is in progress; the
standard-CUDA result is not enough to justify concurrent dumping on every
supported build.

Questions for maintainers:

1. Should dump-only mode retain today's post-dump termination behavior? The
   current monitor waits for up to `heartbeatTimeoutInSec_` (480 seconds by
   default) after a dump and calls `terminateProcess()` unless its stop flag is
   set. If communicator teardown never returns, reusing that path aborts the
   rank after the same long post-dump wait discussed in #169943. The alternative
   is for dump-only mode to emit evidence without owning process termination.
2. What deadline should bound the dump-only lifetime, and should it reuse
   `TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC`?

### Option B: separate dump-signal responder lifetime

Separate global-store dump-signal polling from watchdog-heartbeat monitoring.
A minimal responder could remain alive through process-group teardown while
the watchdog and heartbeat timeout logic stop at the existing point.

This gives the cleanest ownership boundary, but it is a larger architectural
change. It must also preserve today's one-responder-per-process behavior and
avoid accessing a destroyed `ProcessGroupNCCL` object.

### Option C: proactive final dump before stopping the monitor

Write a local final Flight Recorder artifact before communicator destruction.
This is mechanically simpler, but `shutdown()` does not necessarily know that
the process is in an abnormal distributed state. Dumping on every normal
shutdown adds noise and I/O, while dumping only for a known local NCCL error
does not cover the reproduced rank-local Python exception.

This is useful only if maintainers can identify a narrow abnormal-shutdown
predicate that includes the reproducer. It is not sufficient as a generic
replacement for the peer-request path.

### Option D: document the limitation and an application-level workaround

Applications may explicitly request or write a Flight Recorder dump before
calling `destroy_process_group()` after a local exception. This avoids changing
the C++ lifecycle but depends on every caller doing the right thing and cannot
recover failures that enter cleanup through generic exception handling.

Because `nccl2` is now the default backend, maintainers may decide that an
explicit legacy-backend limitation plus this workaround is the appropriate
final resolution. The lab should treat that as a valid upstream disposition,
not only as a temporary step, provided the supported boundary is documented
clearly.

## Recommended upstream discussion

Ask maintainers to choose between Option A and Option B before implementation.
Do not propose a patch that only moves `heartbeatMonitor_->stop()` below the
communicator loop.

Do not post the design options while #196968 has no maintainer response. Once a
maintainer engages, the smallest useful reply should state the required
contract, name the two options, and offer to prepare a regression test and
implementation after the lifecycle owner confirms which behavior is intended.

## Regression matrix

| Case | Required observation |
| --- | --- |
| Normal two-rank shutdown | both ranks return; no unsolicited dump; no material teardown delay |
| Reproduced peer timeout | rank 0 and rank 1 both produce decodable, separately named dumps |
| Dump disabled | shutdown behavior remains unchanged; no artifact required |
| Global store unavailable | bounded warning/failure; teardown is not blocked by diagnostics |
| Multiple process groups | exactly one process-level responder handles the global signal |
| Signal races monitor transition | at most one dump per rank; no use-after-free or duplicate termination |
| Dump exceeds its deadline | failure is bounded and observable; teardown does not wait indefinitely |
| Default `nccl2` backend | unaffected by the legacy-backend change |

The affected integration test must distinguish `producer_missing` from
`member_missing`: no rank-1 artifact is a producer failure, not proof that rank
1 did not participate in the collective.

The proposed in-tree regression should be a sibling of
`test_timeout_dumps_on_stuck_ranks` in
`test/distributed/test_c10d_nccl.py`. Instead of leaving rank 1 in a generic
sleep, rank 1 should call the backend's `shutdown()` path after rank 0 has
enqueued an unmatched collective. The test should require separately named,
decodable dumps from both ranks and retain the existing bounded process-exit
checks. The lab's standalone two-GPU gate remains the external acceptance test;
it is not a substitute for this in-tree regression.

## GPU acceptance gate

Do not rent GPUs until maintainers agree on a direction or request a proposed
patch. A candidate fix is acceptable only if:

1. the unchanged standalone reproducer is run against a pinned current nightly;
2. the legacy backend is explicitly selected;
3. both rank dumps are present and decodable in three affected trials;
4. three normal controls exit cleanly without extra dumps;
5. process cleanup leaves no orphaned launcher or worker;
6. loaded PyTorch, CUDA, NCCL, driver, source revision, and patch revision are
   recorded;
7. failure to obtain either producer artifact fails closed.

Minimum hardware: two CUDA GPUs on one host. Consumer GPUs are sufficient for
the existing reproducer.

## Stop rules

- Stop if the current nightly no longer reproduces before applying the patch.
- Stop if the proposed change requires keeping communicator-dependent monitor
  actions active without a reviewed synchronization argument.
- Stop if the normal control emits dumps or regresses teardown completion.
- Do not reinterpret a single-rank dump as a successful cross-rank diagnostic.
