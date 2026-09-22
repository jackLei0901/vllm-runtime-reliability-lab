# Low-level evidence capability requirements

Status: requirements baseline for Block 5. These requirements authorize
capability experiments, not production integration or a v0.2 schema change.

## 1. Objective

Lower-level evidence is useful only when it distinguishes two already named
explanations that the current evidence cannot separate. The target is not
"more native data." The target is a smaller set of defensible claims.

The immediate questions are:

1. Is a selected thread executing Python, native runtime code, or a transition
   between them?
2. Which closed lifecycle stage has a component entered and not completed?
3. Is a diagnostic producer unavailable, or is its subject absent?
4. Can an existing Flight Recorder, NCCL RAS, or external stack producer answer
   the question without a source patch?

## 2. Scope

### In scope

- capability checks against existing #196968, #53859, and #196996 reproducers;
- attach-based Python/native stack capture on an explicitly supplied PID;
- producer availability and failure semantics;
- version-constrained normalization into closed attribution facts;
- evaluation of existing Flight Recorder and NCCL RAS outputs;
- one minimal lifecycle stage probe only if mature producers cannot expose the
  required fact.

### Out of scope

- automatic root-cause classification;
- arbitrary native frame publication;
- general eBPF, CUDA, NCCL, or NVML instrumentation frameworks;
- continuous hardware telemetry as a progress signal;
- automatic restart or remediation;
- a new v0.2 verdict or a change to its precedence;
- dynamic plugin discovery or a generic adapter SDK.

## 3. Functional requirements

### LLR-001 — explicit subject binding

Every capture shall bind to an operator-supplied process identity and record a
start identity strong enough to reject PID reuse. Rank, worker, process-group,
and service roles shall remain declarations unless independently verified.

**Acceptance:** replacing the target process under the same PID invalidates the
capture relation rather than inheriting the old evidence.

### LLR-002 — bounded capture interval

Every producer invocation shall have a monotonic start, monotonic end, timeout,
and maximum output budget. A point stack shall not be represented as continuous
state.

**Acceptance:** timeout returns a typed producer outcome and never blocks the
collector past the declared bound.

### LLR-003 — mixed Python/native thread observation

The capability experiment shall determine whether the selected producer can
return a thread inventory with Python and native frames for the same target.
Thread identifiers and frame order must be retained privately long enough to
validate normalization.

**Acceptance:** healthy and fault captures state whether mixed frames were
available; absence of native frames is not interpreted as Python execution.

### LLR-004 — GIL state observation

When supported by the producer, the normalized private record shall distinguish
`holding`, `waiting`, `dropping`, and `unknown`. Unsupported or ambiguous state
shall become `unknown`, not an inferred value.

**Acceptance:** the same raw state maps deterministically; a producer without
the capability yields weaker attribution without changing the primary verdict.

### LLR-005 — typed producer outcome

Producer execution shall distinguish at least:

```text
produced
unsupported
binary_missing
permission_denied
timeout
feature_disabled
empty_output
execution_failed
capture_occupied
rate_limited
```

The existing v0.2 stack adapter uses the narrower public set
`binary_missing | permission_denied | timeout | empty_output |
execution_failed`. Block 5 may evaluate a richer experimental vocabulary, but
must not silently rewrite the frozen v0.2 schema.

**Acceptance:** every failed capability check has exactly one bounded outcome;
missing output never becomes a negative target-state observation.

### LLR-006 — closed native attribution vocabulary

Any publishable native interpretation shall use a closed value:

```text
python
queue_wait
ipc_wait
collective_wait
communicator_destruction
watchdog_or_monitor
cuda_synchronization
unknown
```

The vocabulary describes the sampled execution point, not root cause. New
values require the taxonomy admission gate.

**Acceptance:** an unmatched frame shape returns `unknown`; no fuzzy or
nearest-category match is allowed.

### LLR-007 — version-constrained rules as data

Frame and symbol interpretations shall be data records with explicit producer,
vLLM, PyTorch, NCCL, platform, and symbol-shape constraints. Verifier code shall
not branch on producer implementation names.

**Acceptance:** changing an applicable version outside the declared range turns
the attribution into `unknown` without changing the primary verdict.

### LLR-008 — producer interchangeability

Two tools that normalize to the same observation shall produce the same
attribution. A tool with less information may only produce a weaker attribution
or `unknown`.

**Acceptance:** fixture vectors from two named mock producers are identical
after normalization and never require a tool-specific verifier branch.

### LLR-009 — lifecycle transition evidence

For #196968, the required reusable fact is the relation among:

```text
dump_responder_active
dump_responder_stopped
communicator_destroy_started
communicator_destroy_completed
peer_dump_request_observed
dump_completed
```

Each transition shall be process/rank bound and ordered within one monotonic or
logical sequence domain. Free-text logging is insufficient for a public claim.

**Acceptance:** invalid order, duplicate terminal stages, or missing identity
fails closed. The record can distinguish `producer_missing` from
`participant_missing` without reading arbitrary logs.

### LLR-010 — existing communicator evidence first

Flight Recorder or NCCL RAS shall remain authoritative for the logical
collective/communicator facts they already expose. External stacks may
corroborate CPU execution state but shall not synthesize collective membership.

**Acceptance:** a stack with a collective-looking frame cannot create a
collective sequence number, rank membership, or completion state.

### LLR-011 — primary verdict isolation

Native state, GIL state, communicator context, and hardware state shall not
change the v0.2 primary verdict. They may select a separately versioned
attribution or reduce its coverage.

**Acceptance:** mutating or removing all native attribution while preserving the
v0.2 decisional observations leaves the semantic verdict unchanged.

### LLR-012 — privacy-bounded publication

Raw frames, instruction addresses, argument values, source paths, command
lines, environment variables, and process memory remain private. Public output
may retain the closed attribution, rule-set identity, producer outcome, subject
binding, and raw-content digest.

**Acceptance:** schema tests reject raw frame text, absolute paths, arbitrary
stderr, URLs, prompts, or credentials in the public shape.

### LLR-013 — observer safety and flow control

Capture shall be out of process by default, opt-in, time bounded, and protected
by an occupancy flag plus cooldown/quota. Repeated no-progress evaluations shall
not repeatedly attach to the target.

**Acceptance:** a second trigger during an active capture returns
`capture_occupied`; a trigger inside cooldown returns `rate_limited`; neither
starts another attach.

### LLR-014 — healthy/fault pairing

Every native interpretation admitted from Block 5 shall have at least one
healthy or negative-control capture using the same producer and version family.

**Acceptance:** a frame pattern seen in both healthy and fault windows cannot be
used as the sole discriminator.

### LLR-015 — hardware context remains non-decisional

Hardware counters may be retained only when a named hypothesis requires them.
GPU utilization, memory use, clocks, power, temperature, ECC, or Xid state shall
not be interpreted as useful inference progress.

**Acceptance:** high utilization with flat token progress remains no-progress;
missing hardware telemetry does not weaken an otherwise sufficient primary
verdict.

## 4. Case-to-capability traceability

| Requirement | #196968 | #53859/#53883 | #196996 |
| --- | --- | --- | --- |
| LLR-001/002 subject and bounds | required | required | required for multi-rank manifestation |
| LLR-003/004 mixed stack and GIL | useful attribution | useful attribution | not required |
| LLR-005 typed failure | required for degraded operation | required for optional stack | optional |
| LLR-006/007 closed versioned attribution | `communicator_destruction` candidate | `queue_wait` candidate | no native category required |
| LLR-009 lifecycle transitions | required reusable fact | not required | not required |
| LLR-010 communicator evidence | Flight Recorder first | not required | only for competing distributed explanations |
| LLR-011 verdict isolation | required | required | required |
| LLR-012 privacy | required | required | required |
| LLR-013 capture flow control | required if attach is automated | required if attach is automated | not required |
| LLR-014 paired control | required | required | existing dtype controls already satisfy the principle |
| LLR-015 hardware context | no current discrimination | no current discrimination | no current discrimination |

## 5. Minimal C++ probe admission gate

A source-level probe may be designed only when all of the following are true:

1. two competing failure modes are named in
   [`EVIDENCE_TO_CLAIM_BLOCK4.md`](EVIDENCE_TO_CLAIM_BLOCK4.md);
2. PyStack, Flight Recorder, NCCL RAS, existing counters, and external process
   evidence cannot distinguish them;
3. the missing fact is a closed lifecycle transition, not a stack string or
   free-text error;
4. a healthy/fault or base/fix pair can validate the transition;
5. the probe has a bounded overhead and privacy surface;
6. removing the probe demonstrably weakens the attribution to `unknown` or
   `insufficient_evidence`.

The only current candidate is the #196968 dump-responder/shutdown-stage
boundary. #53859 and #196996 do not currently satisfy this gate.

## 6. Block 5 entry and exit criteria

### Entry

- the three evidence-to-claim tables are accepted;
- the target PID, environment, producer version, timeout, and output budget are
  frozen before execution;
- healthy and fault runs use the same normalized observation shape.

### Exit

- PyStack capability is recorded for mixed frames, GIL state, attach permission,
  timeout, empty output, and subject binding;
- each result states which Block 4 row it strengthens;
- equivalent normalized observations are producer-independent;
- unsupported and permission-denied paths are first-class results;
- the review concludes either `existing_tools_sufficient` or names exactly one
  irreducible lifecycle fact for probe design;
- no code is merged merely because more native data was collected.

## 中文审阅摘要

底层能力不是“采更多 C++ 信息”，而是补齐 Block 4 已命名的判定缺口。当前只有
#196968 的 dump responder 与 communicator destruction 生命周期关系可能需要最小
C++ probe；#53859 的主 verdict 已由 process/health/demand/progress 决定，native stack
只能增强归因；#196996 已在本地 dtype contract 边界完成机制证明，不需要 native
采集。

所有 Block 5 producer 必须满足：显式主体绑定、有界执行、失败原因类型化、原始栈默认
私有、规则带版本约束、未匹配输出 `unknown`、工具实现身份不进入判定、缺失 producer
只能降低 attribution coverage，不能改写 v0.2 verdict。只有成熟工具无法提供一个已命名
的关闭状态转换时，才允许设计最小 C++ probe。
