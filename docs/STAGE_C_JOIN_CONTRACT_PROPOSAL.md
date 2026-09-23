# Stage C identity/window join — review proposal

Status: **proposal only**. Stage C remains `join_contract_required`; this
document does not admit a cross-producer attribution, authorize a C++ probe, or
change the v0.2 verdict. The target is the existing #196968 two-rank teardown
case, not a new fault campaign. The
[`2026-09-23 retained-input audit`](STAGE_C_RETAINED_INPUT_AUDIT_2026-09-23.md)
found that existing artifacts cannot retroactively satisfy this proposal.

## Question and source boundary

The bounded question is whether rank 1 was still a participant while its
Flight Recorder peer-dump producer was unavailable during communicator
destruction. A stack snapshot, lifecycle-stage flags, a peer dump request, and
a Flight Recorder artifact are four independently produced facts. No single
producer digest may vouch for all four.

In the reviewed legacy c10d source, `ProcessGroupNCCL::shutdown()` joins the
watchdog before communicator destruction. The unpatched `origin/main` source
then calls `heartbeatMonitor_->stop()` before `NCCLComm::destroy()`, which calls
`ncclCommDestroy`. The proposed #197232 branch instead retains a narrowly
scoped dump-signal responder during destruction and stops it afterward. See
the source-level review in
[`reviews/C10D_SHUTDOWN_SOURCE_REVIEW.md`](reviews/C10D_SHUTDOWN_SOURCE_REVIEW.md).
These are source facts, not an inference that `ncclCommDestroy` caused the
original hang or that the proposed fix has been accepted upstream.

## Proposed join inputs

| Input | Subject and proof | What it cannot prove alone |
| --- | --- | --- |
| Rank-1 process sample | PID and Linux start ticks before and after bounded capture; rank is only an operator declaration until independently bound | Collective participation, progress, or rank identity |
| Launcher rank binding | Separately produced and digested rank-to-PID/start-ticks record derived from the launched child's rank assignment, not copied from the capture operator's declaration | Runtime progress or dump-responder state |
| Rank-1 stack capture | Its own typed outcome, capture interval and raw digest; normalized frames only | Responder state or peer request reception |
| Rank-1 lifecycle flags | Separately digested producer record, rank identity, ordered stage sequence | The stack's exact blocking frame |
| Rank-0 dump request | Rank-0 identity, successful request marker and its own decodable dump | Rank 1 received the request |
| Rank-1 dump absence | Separately digested, closed expected-file manifest at the pre-registered observation end | Rank 1 was absent or did not participate |

The join key is **pre-registered run identity plus PID and process start
ticks**, not a declared rank, filename, or matching wall clock. The declared
rank must be cross-checked against an independently produced launcher binding
for that same PID/start identity; a rank-0 Flight Recorder artifact's own rank
field can corroborate rank 0, but an absent rank-1 artifact cannot bind rank
1. A missing or mismatched independent binding makes the join
`join_not_scorable`. Each input retains its own producer kind, source digest,
and outcome. If any required process identity is missing or changes, the join
is likewise `join_not_scorable`.
Digest equality establishes byte identity, not trust in the original producer.

The manifest is closed only if the harness creates the expected output
directory empty, controls its writers for the entire run, fixes the expected
rank/file set before execution, and enumerates all entries (including partial
or unexpected files) at a pre-registered instant after the bounded dump wait.
The enumeration is canonicalized and digested. An externally shared directory,
an unbounded listing, or an unknown writer invalidates the absence claim.
"Absent" means no complete, decodable rank-1 dump existed **by that instant**;
it does not make a claim about later files.

For the unpatched arm, the retained shutdown marker records that
`heartbeatMonitor_->stop()` was called, **not** that the monitor thread had
already exited. Source ordering plus this marker and bounded
non-response can support the missing-producer observation for this run; they
do not establish the precise exit instant or prove that the responder was
unavailable at every instant of communicator destruction.

## Time and causality

Within one process, monotonic capture bounds and lifecycle logical sequence
may order local events. Monotonic timestamps from different hosts are not
directly comparable. Cross-rank ordering requires an explicit experiment
barrier or request/acknowledgment marker with a pre-registered causal meaning;
wall-clock proximity alone is insufficient. A rank-0 broadcast marker proves
that the request was issued, not that rank 1 observed it.

For the producer-loss claim, require the held rank-1 teardown marker to be
established before the rank-0 request, rank-1 process identity to remain stable
through the bounded observation, and no complete rank-1 dump in the closed
artifact set. A stack taken only before the held marker cannot be joined as
fault-time attribution. If the marker is released or its causal relation to
the request is unknown, retain the individual observations and return
`join_not_scorable`.

In the proposed-fix arm, the responder has a finite deadline computed as
`options_->timeout + getDumpTimeout()` from its shutdown-time enablement. The
joined observation must end before that deadline, using a bound derived from
the same process's monotonic clock and retained as separate lifecycle
provenance. If the deadline or its clock binding is unavailable, or the
observation crosses it, the fix-arm join is `join_not_scorable`: an absent dump
after intentional responder expiry is not producer loss. Being inside the
deadline is necessary, not proof that the responder was still running; store
failure or another early exit must be checked independently. The unpatched arm
does not have this proposed responder deadline.

## Closed join results and precedence

The Stage C join has its own closed result set:

```text
producer_loss_supported | producer_loss_refuted | evidence_incomplete |
join_not_scorable
```

`join_not_scorable` takes precedence for invalid subject/rank binding,
unclosed manifests, missing causal ordering, or out-of-window observations.
For a valid join, an observed request response or complete rank-1 dump
refutes the proposed loss mechanism. If no contradiction exists but required
producer facts are unavailable, the result is `evidence_incomplete`. Only all
required, non-contradictory facts support `producer_loss_supported`.

`producer_loss_supported` maps to Block 4's bounded `producer_missing`
claim, never `member_missing`; `evidence_incomplete` maps to Block 4
`insufficient_evidence` for that claim. `producer_loss_refuted` and
`join_not_scorable` make no Block 4 producer-loss claim. These are **not**
v0.2 verdicts or Block 5 A/B/A2 pairing results. Every join result leaves the
separately recomputed v0.2 verdict unchanged.

## Decision table for review

| Condition and source producer | Join result |
| --- | --- |
| Stable rank-1 identity (external process sampler), independently verified rank (launcher binding), valid causal marker (harness), rank-0 request and dump (request marker and Flight Recorder), monitor-stop-requested and destruction-started flags without completion (lifecycle-stage producer), rank-1 dump absent (closed manifest) | `producer_loss_supported` for the tested legacy backend and interval; no exact monitor-exit claim |
| Rank-1 process absent or start identity changed (external process sampler), or declared rank mismatches the launcher binding | `join_not_scorable`; investigate participant state separately |
| Rank-1 responder observed the request (lifecycle-stage producer) or a complete rank-1 dump exists (Flight Recorder and manifest) | `producer_loss_refuted` for the proposed mechanism |
| Lifecycle flags absent, unordered, or falsely attached to a stack record (lifecycle-stage producer versus stack producer) | `evidence_incomplete` if genuinely missing; invalid provenance or order makes the join `join_not_scorable` |
| Cross-rank causal marker absent (harness), capture outside held state (stack and lifecycle windows), unclosed manifest (harness), or fix-arm observation outside/without the responder deadline (lifecycle provenance) | `join_not_scorable`; source-specific facts remain visible |

The public join projection should carry the run/subject binding digest, each
input's independent digest and typed outcome, the tested source/backend
identity, the causal-marker result, and the bounded conclusion. It must not
contain raw frames, logs, file paths, addresses, arguments, or free-text
errors. The v0.2 verdict is recomputed independently and remains unchanged
when the entire Stage C sidecar is removed.

## Acceptance and stop rules

Before implementing a join verifier, review: (1) the exact retained-case
marker and its cross-rank causal semantics; (2) independent rank binding,
closed input schema, and manifest-closure procedure; (3) healthy and fault
negative controls; (4) the fix-arm responder-deadline bound. A causal
base/fix comparison is admitted only if source revision (apart from the
declared patch), toolchain, PyTorch/CUDA/NCCL binaries, backend selection,
runtime flags, topology, harness, and marker protocol are pinned equal.
Otherwise the two arms may support separate bounded observations, as in the
existing validation, but not an isolated patch-effect or performance claim.
A join that only aligns timestamps or merges observations into one
synthetic producer is rejected. Until this review closes, Stage C's exit is
`join_contract_required`, not `existing_tools_sufficient` or `probe_needed`.

Minimum future verifier vectors: a swapped declared rank with otherwise
matching timestamps is `join_not_scorable`; a fix-arm request after the
responder deadline is `join_not_scorable`; an unexpected writer or late
manifest enumeration is `join_not_scorable`; missing lifecycle flags on an
otherwise valid join are `evidence_incomplete`; a complete rank-1 dump
refutes producer loss. A stop-requested marker alone never proves monitor
thread exit. These are contract expectations, not tests claimed to exist yet.

Even after a valid join, a lab-local C++ probe requires the separate LLR-009
admission test: show that PyStack, Flight Recorder, NCCL RAS and existing
external evidence cannot supply the named transition. Any upstream-facing
instrumentation remains gated on an explicit #197232 maintainer outcome.

## 中文审阅摘要

本文件只提出 Stage C 的跨 producer join 契约，不实施 join，也不新增 C++ probe。
join 只用预注册 run、PID 和 start ticks 作主体键；声明的 rank 必须由独立 launcher
记录验证。stack、stage flags、peer request 和 dump manifest 分别保留来源与 digest。
dump 目录须由 harness 清空并独占，预先固定预期文件集，在预注册时刻完整枚举。
不同主机的 monotonic 时间不能直接排序，跨 rank 先后关系须由受控 marker 建立。
fix arm 还必须在有证据绑定的 responder deadline 内观察；超过 deadline 的缺失
dump 不能算故障。join 的关闭结果是 `producer_loss_supported`、
`producer_loss_refuted`、`evidence_incomplete`、`join_not_scorable`；仅第一项映射
到 Block 4 的有界 `producer_missing`，所有结果均不改变 v0.2 verdict。未修复版本的
stage marker 只证明调用了 `stop()`，不证明 monitor 线程已经退出。
