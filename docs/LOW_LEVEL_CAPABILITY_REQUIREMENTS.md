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

### LLR-005 — typed attempt stage and outcome

The record shall separate where an attempt stopped from why it stopped. The
closed pair is:

```text
attempt_stage = not_requested
  outcome = disabled

attempt_stage = coordinator
  outcome = capture_occupied | rate_limited

attempt_stage = preflight
  outcome = unsupported | binary_missing | feature_disabled

attempt_stage = execution
  outcome = produced | timeout | output_budget_exceeded |
            permission_denied | empty_output | execution_failed
```

The producer is invoked only for `attempt_stage = execution`. Coordinator and
preflight outcomes therefore make no statement about attach behavior or target
state. `not_requested × disabled` records explicit operator opt-out and is not
missing evidence from a requested capture. The verifier shall reject every
stage/outcome pair outside this matrix.

Permission is established only by an execution attempt. Preflight observations
such as Linux Yama `ptrace_scope` are provenance and shall not gate the producer
or emit `permission_denied`: a target may grant scoped authorization even when
the system setting looks restrictive.

The existing v0.2 stack adapter has a separate stack state and the narrower
flat `error_kind` set `binary_missing | permission_denied | timeout |
empty_output | execution_failed`. Block 5 may evaluate the staged experimental
shape, but must not silently reinterpret or rewrite the frozen v0.2 schema.

**Acceptance:** every capture plan has exactly one valid stage/outcome pair;
`capture_occupied` cannot be confused with an attach attempt, and missing
output never becomes a negative target-state observation.
`raw_output_sha256` is null for `not_requested`, coordinator, and preflight;
`execution × produced` requires a non-null digest. A failed execution may retain
a digest of bounded partial output, but that digest does not make the attempt
`produced`.

### LLR-006 — closed native attribution vocabulary

Only the values required by current Block 4 rows are emittable:

```text
queue_wait
communicator_destruction
unknown
```

`ipc_wait`, `collective_wait`, `watchdog_or_monitor`, and
`cuda_synchronization` are reserved names and are not valid outputs until a
Block 4 evidence row and taxonomy admission record require them. `python`
belongs to `execution_domain`, not `blocked_in`. The vocabulary describes the
sampled execution point, not root cause.

**Acceptance:** an unmatched frame shape returns `unknown`; no fuzzy or
nearest-category match is allowed.

### LLR-007 — version-constrained rules as data

Frame and symbol interpretations shall be data records with explicit producer
**kind**, vLLM, PyTorch, NCCL, platform, topology, and normalized symbol-shape
constraints. Attribution rules shall never constrain or read producer
implementation name or implementation version. Implementation-specific version
support belongs to the normalizer and is provenance, not an attribution input.

**Acceptance:** changing a target-runtime version outside the explicit reviewed
allowlist turns the attribution into `unknown` without changing the primary
verdict; changing producer implementation identity without changing normalized
facts does not change attribution.

### LLR-008 — producer interchangeability

Two tools that normalize to the same observation shall produce the same
attribution. A tool with less information may only produce a weaker attribution
or `unknown`.

**Acceptance:** mock vectors prove only evaluator purity. The real comparison
uses a deterministic held target and sequential captures because the occupancy
rule forbids simultaneous attach. A same-tool capture before and after the
other tool must satisfy the same rule predicates and emit the same
`blocked_in`; otherwise the experiment reports `target_not_stable` and makes no
interchangeability claim. If the stability control passes, both tools must
satisfy the same applicable rule predicates and emit the same `blocked_in`.
Their normalized frame sequences need not be identical, and tool-only detail
may change declared coverage. No tool-specific verifier branch is allowed.
`target_not_stable` is a pairing-level experiment result, not a producer
stage/outcome code. A triplet with no admitted rule, or with any unusable
capture, is `not_scorable`; three `unmatched` results shall never be called
`interchangeable`.

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
Lifecycle flags and stack facts remain separate producer records. A
single-producer stack observation cannot carry lifecycle facts, and a future
joined attribution must retain every source binding and raw-content digest.

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

A post-capture process-identity recheck is provenance only. If it fails, the
native capture binding is invalid and its attribution becomes unavailable; it
does not emit `process_missing`. The v0.2 process producer remains the only
source of primary process state.

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
| LLR-005 staged outcome | required for degraded operation | required for optional stack | optional |
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

The gate has two different outputs:

- **Lab-local measurement:** after mature producers are proven insufficient,
  Block 5 may specify one bounded experiment-only stage probe to test the
  evidence relation. This does not make the probe a product dependency.
- **Upstream-facing instrumentation:** no probe issue or PR is proposed until
  #197232 has an explicit maintainer outcome. If the lifecycle framing is
  rejected or superseded, the probe remains lab-local unless a maintainer
  explicitly requests it. An open PR, triage label, CI state, or silence is not
  an outcome.

Block 5 exits with a probe decision and design boundary; it does not need to
implement the expensive probe before that upstream gate closes.

## 6. Block 5 entry and exit criteria

### Entry

- the three evidence-to-claim tables are accepted;
- the target PID, environment, producer implementation/version as provenance,
  timeout, and output budget are frozen before execution;
- healthy and fault runs use the same normalized observation shape.

### Exit

- PyStack capability is recorded for mixed frames, GIL state, attach permission,
  timeout, empty output, and subject binding;
- each result states which Block 4 row it strengthens;
- sequential `py-spy`/PyStack captures are bracketed by a same-tool stability
  control on a deterministic held target;
- after that control passes, both tools satisfy the same rule predicates and
  emit the same `blocked_in`, while frame sequences and coverage may differ;
- unsupported and permission-denied paths are first-class results;
- before an identity/window join contract is reviewed, the review concludes
  `join_contract_required`; only after that gate may it conclude
  `existing_tools_sufficient` or name exactly one irreducible lifecycle fact
  for probe design;
- no code is merged merely because more native data was collected.

## 中文审阅摘要

底层能力不是“采更多 C++ 信息”，而是补齐 Block 4 已命名的判定缺口。当前只有
#196968 的 dump responder 与 communicator destruction 生命周期关系可能需要最小
C++ probe；#53859 的主 verdict 已由 process/health/demand/progress 决定，native stack
只能增强归因；#196996 已在本地 dtype contract 边界完成机制证明，不需要 native
采集。

所有 Block 5 producer 必须满足：显式主体绑定、有界执行、coordinator/preflight/
execution 结果分层、原始栈默认私有、规则带目标版本约束、未匹配输出 `unknown`、工具
implementation identity/version 不进入 attribution、缺失 producer 只能降低 coverage，
不能改写 v0.2 verdict。只有成熟工具无法提供一个已命名的关闭状态转换时，才允许设计
lab-local 最小 C++ probe；任何 upstream-facing probe 继续受 #197232 明确结果 gate 约束。
