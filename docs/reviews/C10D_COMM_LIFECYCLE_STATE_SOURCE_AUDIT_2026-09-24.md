# c10d communicator lifecycle-state source audit — 2026-09-24

Decision: **a real representational limit is visible in the pinned wrapper,
but an added `CommState` enum or upstream PR is not yet justified.** This is
the second read-only c10d candidate audit, not a new reproduction or a
change to #197232.

Source: PyTorch candidate head
`10909dd81964c2497f1f2657c1132dbc1991e582`, legacy
`torch/csrc/distributed/c10d/NCCLUtils.{hpp,cpp}` and
`ProcessGroupNCCL.cpp`. `nccl2` and the internals of NCCL RAS were not
audited. A bounded GitHub search for `NCCLComm` lifecycle/state/teardown
found adjacent [shutdown-race #124468](https://github.com/pytorch/pytorch/issues/124468)
and [process-group revoke RFC #188317](https://github.com/pytorch/pytorch/issues/188317)
discussions, not a verified exact duplicate; repeat the check before any
proposal. The RFC's API/lifecycle distinctions are prior art, not evidence
that a new wrapper enum is needed.

## State actually held by the wrapper

`NCCLUtils.hpp:362,371,375` declares `aborted_`, `initialized_`, and
`nonBlocking_`. The third is a **mode**, not a lifecycle stage; counting three
state booleans overstates the model. The first two are not a closed lifecycle
enum. `NCCLUtils.cpp:570-572` `repr()` prints a pointer, not a stage.

| Path and source anchor | Wrapper state relevant to the question | Supported observation | Missing or misleading inference |
| --- | --- | --- | --- |
| Ready communicator, `NCCLUtils.cpp:135-160` | Normally `initialized_=true`, `aborted_=false` after readiness. | `getNcclComm()` can return a usable handle. | The flags do not name a later finalize/destroy transition. |
| `finalize()`, `NCCLUtils.cpp:331-341` | No lifecycle field changes after `ncclCommFinalize()`. | Source order proves finalize was called on that path. | The two flags alone do not distinguish pre- from post-finalize. |
| `destroy()` in progress, `NCCLUtils.cpp:343-352` | `aborted_` changes only after `ncclCommDestroy()` returns. A recursive mutex is held through the call (`NCCLUtils.hpp:237`). | Source proves a destruction interval exists. A native stack or external stage marker may sample/mark it. | It is **not** valid to claim that `isAborted()` concurrently reads `false` and misclassifies it as healthy: `isAborted()` also locks (`NCCLUtils.cpp:435-437`) and may block until destruction ends. No safe public in-progress stage was shown. |
| Successful clean destroy, `NCCLUtils.cpp:352-355` | `aborted_=true`; the wrapper comments say this poisons future `getNcclComm()`. | The communicator is invalid for further use. | `isAborted()==true` does not distinguish orderly destroy from failure abort. The name serves invalidation semantics, not cause attribution. |
| Failure abort, `NCCLUtils.cpp:357-427` | `aborted_=true` after a successful abort; `ncclComm_` becomes null and async error can be poisoned. | This is a different control-flow path from clean destroy. | The public `isAborted()` result is still `true`; the optional reason may be absent. |

If `ncclCommDestroy()` throws before line 354, this table makes no claim about
the underlying NCCL object's recoverability. The wrapper assignment has not
run, but source order alone cannot characterize the external library state.

## Existing observations and the Stage C limit

`ProcessGroupNCCL::shutdown()` logs before its communicator-destruction loop
and after completion (`ProcessGroupNCCL.cpp:1601-1611`). If retained and
correctly ordered, those markers already bracket the *process-group* destroy
interval; a bounded external stack may sharpen that observation. The loop
can visit multiple entries in `devNCCLCommMap_`, however, and those PG-wide
markers do not identify which communicator is inside `ncclCommDestroy()`.
They also do not give a durable *per-communicator* transition or a cross-rank
causal join.

The Flight Recorder shape retains collective and PG-sequence states, but the
legacy shutdown peer dump path deliberately avoids calling native
communicator dump APIs while `destroy()` may be active
(`ProcessGroupNCCL.cpp:1705-1715`). A comm-state enum in the wrapper alone
would not fix #196968's more basic risk that the entire rank dump producer
may be unavailable. It also would not supply Stage C's missing rank/PID
binding, closed artifact manifest, responder-stage provenance, or same-clock
deadline relation. See `docs/STAGE_C_RETAINED_INPUT_AUDIT_2026-09-23.md`.

## Discrimination gate before an enum or probe

Two named questions are worth testing, separately:

1. Given the retained PG-wide start/end markers for a bounded destroy
   interval, can existing logs, native stacks, or other retained evidence
   identify **which communicator** in `devNCCLCommMap_` is being destroyed,
   versus another communicator that is still ready? If the claim needs only
   PG-wide stage, the existing markers may already settle it: **no probe
   needed**.
2. Does an external investigator ever need to distinguish **cleanly
   destroyed** from **failure-aborted** after the process or communicator has
   gone, when both report `isAborted()==true`?

For each, first establish a positive case, a negative control, retained
existing-DFX artifacts, and the decision that changes if the distinction is
known. If current logs, Flight Recorder, NCCL RAS, and stack evidence already
answer the question, conclude **no probe needed**. Otherwise propose a
minimal closed transition at the state owner, with explicit lifetime,
thread-safety, overhead and dump-availability semantics. An enum without a
durable producer/consumer path is merely a new in-memory label.

No test, GPU run, issue or PR was created by this source audit. Hold any
second upstream proposal in this seam until #197232 has substantive reviewer
engagement and a fresh duplicate check.
