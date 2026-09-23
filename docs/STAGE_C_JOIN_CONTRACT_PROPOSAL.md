# Stage C identity/window join — review proposal

Status: **proposal only**. Stage C remains `join_contract_required`; this
document does not admit a cross-producer attribution, authorize a C++ probe, or
change the v0.2 verdict. The target is the existing #196968 two-rank teardown
case, not a new fault campaign.

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
| Rank-1 process sample | Operator-declared rank, PID and Linux start ticks before and after bounded capture | Collective participation or progress |
| Rank-1 stack capture | Its own typed outcome, capture interval and raw digest; normalized frames only | Responder state or peer request reception |
| Rank-1 lifecycle flags | Separately digested producer record, rank identity, ordered stage sequence | The stack's exact blocking frame |
| Rank-0 dump request | Rank-0 identity, successful request marker and its own decodable dump | Rank 1 received the request |
| Rank-1 dump absence | Closed expected-file manifest and bounded observation end | Rank 1 was absent or did not participate |

The join key is the **pre-registered run identity plus declared rank and
process start identity**, not a shared PID, a filename, or a matching wall
clock. Each input retains its own producer kind, source digest, and outcome.
If any required identity is missing or changes, the join is `not_scorable`.
Digest equality establishes byte identity, not trust in the original producer.

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
`not_scorable`.

## Decision table for review

| Condition | Permitted result |
| --- | --- |
| Stable rank-1 identity; valid causal marker; rank-0 request and dump; responder-stopped and destruction-started flags; destruction not completed; rank-1 dump absent in closed manifest | `producer_missing` for the tested legacy backend and interval; never `member_missing` |
| Rank-1 process absent or start identity changed | No producer-loss claim; investigate participant state separately |
| Rank-1 responder observed the request or a complete rank-1 dump exists | Contradicts the missing-producer explanation |
| Lifecycle flags absent, unordered or sourced only from a stack record | `insufficient_evidence` for producer lifecycle |
| Cross-rank causal marker absent or capture intervals do not cover the held state | `not_scorable` join; source-specific facts remain visible |

The public join projection should carry the run/subject binding digest, each
input's independent digest and typed outcome, the tested source/backend
identity, the causal-marker result, and the bounded conclusion. It must not
contain raw frames, logs, file paths, addresses, arguments, or free-text
errors. The v0.2 verdict is recomputed independently and remains unchanged
when the entire Stage C sidecar is removed.

## Acceptance and stop rules

Before implementing a join verifier, review: (1) the exact retained-case
marker and its cross-rank causal semantics; (2) the closed input schema and
digest manifest; (3) healthy and fault negative controls; (4) whether the
legacy and proposed-fix build identities are comparable for the particular
claim. A join that only aligns timestamps or merges observations into one
synthetic producer is rejected. Until this review closes, Stage C's exit is
`join_contract_required`, not `existing_tools_sufficient` or `probe_needed`.

Even after a valid join, a lab-local C++ probe requires the separate LLR-009
admission test: show that PyStack, Flight Recorder, NCCL RAS and existing
external evidence cannot supply the named transition. Any upstream-facing
instrumentation remains gated on an explicit #197232 maintainer outcome.

## 中文审阅摘要

本文件只提出 Stage C 的跨 producer join 契约，不实施 join，也不新增 C++ probe。
join 必须绑定预注册 run、rank、PID start ticks，并分别保留 stack、stage flags、
peer request 和 dump manifest 的 digest。不同主机的 monotonic 时间不能直接排序；
跨 rank 的先后关系必须由受控 marker 建立。只有身份稳定、marker 有效、rank 0
确实发出请求且产生 dump、rank 1 的 responder 已停止而 communicator destruction
未完成、封闭文件集内没有 rank-1 dump，才允许限定地说 `producer_missing`。任何缺口
返回 `not_scorable` 或 `insufficient_evidence`，绝不推成 `member_missing`。
