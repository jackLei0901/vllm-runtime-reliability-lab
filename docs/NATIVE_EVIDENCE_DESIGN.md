# Native-state evidence design

Status: design baseline for Block 5 capability experiments. The experimental
contract and bounded acquisition boundary are implemented; no public schema
migration or target-runtime attribution rule is authorized by this document.

## 1. Design goal

Add native-state attribution without turning the lab into a stack collector or
allowing tool-specific facts to bypass the frozen v0.2 verdict contract.

The design separates acquisition, normalization, interpretation, and claims:

```text
existing producer (PyStack / Flight Recorder / NCCL RAS)
        |
        | bounded private output + typed producer outcome
        v
producer-specific acquisition boundary
        |
        | common observation shape; no verdict
        v
normalizer
        |
        | version-constrained closed facts
        v
attribution evaluator -----------------------+
        |                                     |
        | attribution + coverage              | v0.2 observations
        v                                     v
experimental sidecar                    existing verifier
        |                                     |
        +---------- displayed together -------+
                       never merged
```

The existing verifier remains the only owner of
`process_missing | health_lost | progress_observed |
alive_health_ok_no_progress | undetermined`.

## 2. Architectural decisions

### D-01 — sidecar before schema extension

Block 5 writes an experimental native sidecar rather than modifying
`observations.json` or `summary.json`. This prevents capability exploration
from silently changing the v0.2 contract.

A future integration requires a separately versioned schema, migration rules,
field-role audit, and mutation tests.

### D-02 — raw acquisition remains private

Producer output is stored only in an operator-controlled private directory.
The experimental public sidecar contains hashes and closed normalized values,
never raw frames or stderr.

### D-03 — producer kind and implementation are separate

Producer kind declares the observation family, such as `stack_snapshot`,
`flight_recorder`, or `nccl_ras`. Implementation name/version is provenance.
Attribution rules may constrain producer kind but shall never read or constrain
implementation name or implementation version. The producer-specific
normalizer owns format/version support; the attribution evaluator consumes only
normalized facts, producer kind, target-runtime constraints, and rule-set
identity.

### D-04 — classification is exact and version constrained

Rules are data. A rule applies only when every declared target-runtime
version/topology constraint and ordered normalized-frame predicate matches.
Implementation-version checks stop at normalization. No substring-only
root-cause classification, fuzzy score, or nearest match is allowed.

### D-05 — attribution and verdict remain separate

Native state may explain an existing no-progress or producer-loss claim. It
cannot establish demand, service progress, process identity, health history, or
collective membership.

### D-06 — degraded operation is normal

Coordinator rejection, preflight failure, and attempted execution are separate
stages. Missing binaries, unsupported platforms, occupied capture slots, rate
limiting, ptrace denial, timeout, output-budget exhaustion, and empty output
retain their stage-specific meaning. They reduce attribution coverage; they do
not fail the primary collector or become target-state evidence.

## 3. Experimental data model

The examples below are design shapes, not committed schemas.

### 3.1 Capture envelope

```json
{
  "schema_version": "native-evidence-experiment-v0",
  "subject": {
    "process_identity_kind": "linux_proc_start_ticks",
    "process_identity_value": "123456",
    "post_capture_identity_value": "123456",
    "declared_role": "engine_core",
    "declared_rank": null
  },
  "window": {
    "start_monotonic_ns": 100,
    "end_monotonic_ns": 200,
    "producer_timeout_ns": 5000000000,
    "coordinator_timeout_ns": 7000000000,
    "max_output_bytes": 1048576
  },
  "producer": {
    "kind": "stack_snapshot",
    "implementation_name": "pystack",
    "implementation_version": "reviewed-version",
    "binary_sha256": "64 lowercase hex",
    "platform": "linux"
  },
  "outcome": {
    "attempt_stage": "execution",
    "outcome_code": "produced",
    "raw_output_sha256": "64 lowercase hex"
  }
}
```

`implementation_name` and `implementation_version` are shown to make the role
boundary explicit. If this shape becomes public, those values must be bounded
and classified as non-decisional provenance. Evaluators must not branch on
them. The current v0.2 optional stack adapter invokes `py-spy`; evaluating
PyStack in Block 5 does not silently replace that released adapter.

For `not_requested`, coordinator, and preflight stages,
`raw_output_sha256` is required to be `null` because the producer was not
invoked. For execution, it is non-null when output exists; `produced` requires
a non-null digest. `not_requested × disabled` is the explicit opt-out record.
Yama scope and similar permission hints are provenance only; only an actual
execution attempt may return `permission_denied`.

`timeout`, `output_budget_exceeded`, and `execution_failed` may carry a digest
of bounded partial output. The digest vouches only for those retained bytes; it
does not upgrade the outcome or make the partial output normalizable.

### 3.2 Normalized stack facts

Private normalization may produce multiple thread facts. Frame classes retain
only the interleaved ordering needed by rule predicates:

```json
{
  "thread_ref": "capture-local opaque id",
  "execution_domain": "mixed",
  "gil_state": "waiting",
  "ordered_frame_classes": [
    "python:publisher",
    "python:queue-put",
    "native:condition-wait"
  ],
  "lifecycle_facts": []
}
```

Allowed `execution_domain` values:

```text
python | native | mixed | unknown
```

Allowed `gil_state` values:

```text
holding | waiting | dropping | unknown
```

Addresses, arguments, local variables, full paths, arbitrary function names,
and thread names do not enter the publishable shape.

A `stack_snapshot` observation must have an empty `lifecycle_facts` list.
Lifecycle transitions come from a separate `lifecycle_stage_flags` capture;
attaching them to stack output would falsely make the stack digest appear to
vouch for independently produced facts.

### 3.3 Lifecycle facts

Lifecycle facts are distinct from stack classifications:

```json
{
  "component": "process_group_nccl",
  "stage": "dump_responder_stop_requested",
  "logical_sequence": 3,
  "observed": true
}
```

Closed candidate stages for the #196968 experiment are:

```text
dump_responder_active
dump_responder_stop_requested
communicator_destroy_started
communicator_destroy_completed
peer_dump_request_observed
dump_completed
```

`dump_responder_active` records entry into the default process group's enabled
dump-signal polling loop, not merely a configured option or an inferred live
thread. `dump_responder_stop_requested` records a termination request; it is
not a thread-exit event. Only active-before-stop-requested,
destroy-started-before-destroy-completed, and request-observed-before-dump-
completed are globally ordered when both events are present. The relative
order of stop request and communicator destruction belongs in an exact
source-versioned lifecycle rule.

The evaluator rejects duplicate logical sequence positions, impossible order,
unknown stages, and identity changes. A lifecycle-stage record contains no
stack frames, execution domain, or GIL claim.

### 3.4 Public attribution projection

```json
{
  "attribution_schema_version": "native-attribution-v0",
  "subject_binding_digest": "64 lowercase hex",
  "binding_status": "valid",
  "capture_attempt_stage": "execution",
  "capture_outcome": "produced",
  "blocked_in": "queue_wait",
  "execution_domain": "mixed",
  "gil_state": "waiting",
  "rule_set_id": "vllm-zmq-queue-wait-v1",
  "rule_match": "exact",
  "coverage": ["thread_state", "gil_state"],
  "raw_output_sha256": "64 lowercase hex"
}
```

Only `queue_wait`, `communicator_destruction`, and `unknown` are currently
emittable `blocked_in` values. Reserved values do not become valid outputs
until a Block 4 row and taxonomy admission require them. `rule_match` is
`exact | unmatched`; `unmatched` requires `blocked_in = unknown`. Coverage is a
closed set, not a confidence score.

## 4. Rule representation

A Stage A single-producer rule contains only declarative constraints:

```yaml
rule_set_id: vllm-zmq-queue-wait-v1
applies_to:
  producer_kind: stack_snapshot
  platform: linux
  vllm_versions: [explicitly-reviewed-version]
  pytorch_versions: [explicitly-reviewed-version]
  pytorch_backends: [not-applicable]
  nccl_versions: [not-applicable]
  topologies: [single-process]
requires:
  ordered_frame_classes:
    - python:queue-put
    - native:condition-wait
  lifecycle_stages: []
forbids:
  lifecycle_stages: []
emits:
  blocked_in: queue_wait
```

The Stage A evaluator uses explicit reviewed-version allowlists rather than
open-ended semantic ranges. Wider ranges may be admitted only after Block 5
captures establish compatibility. `producer_kind` is allowed here;
implementation name and implementation version are forbidden. A missing target
version, topology mismatch, missing required fact, forbidden fact, or frame
mismatch produces `unknown`.

A rule with neither a required frame predicate nor a required lifecycle
predicate is invalid rather than universally matching. A Stage A rule may use
one predicate family only: stack rules cannot require lifecycle stages, and
lifecycle rules cannot require stack frames.

Existing stack rules retain this exact shape, preserving the published Stage B
rule digest. A `lifecycle_stage_flags` rule instead requires an additional
`requires.lifecycle_order` list of `[before, after]` stage pairs. Both stages
must also appear in that rule's required lifecycle stages, and the rule's
`applies_to.pytorch_versions` and
`applies_to.pytorch_source_revisions` must pin the tested build and exact
40-character source commit; the lifecycle target record must carry the same
source revision. Stack rules and their target records keep their published
shape. For the unpatched
legacy arm only, a reviewed rule may require
`dump_responder_stop_requested -> communicator_destroy_started`. The proposed
fix's successful order is different and remains valid at the observation
layer; no global validator may impose the unpatched order. No Stage C rule is
admitted by this design text alone.

### 4.1 Stage C multi-producer join

The #196968 attribution cannot be represented by pretending stage flags came
from PyStack. Stage C therefore uses separate capture records:

```text
stack_snapshot capture ----------+
                                 +--> explicit joined attribution
lifecycle_stage_flags capture ---+
```

The join must verify the same subject-binding digest, compatible incident
windows, target-runtime identity, and each source's own typed outcome. Its
public projection must retain a closed list of source producer kinds, subject
bindings, and raw-content digests. A single `raw_output_sha256` is insufficient
for a joined claim. The current Stage A evaluator deliberately has no joined
rule path; Stage C remains non-scorable until that separate join contract and
its contradiction tests are reviewed.

Producer comparison also has a closed non-claim result. `not_scorable` with
reason `no_admitted_rule` is returned when all three attributions are
`unmatched`; reason `unusable_capture` is returned when any capture did not
produce usable, identity-bound evidence. Neither result is producer
interchangeability.

## 5. Capture orchestration

### 5.1 Trigger input

Block 5 may trigger one capture from an already established condition:

- a transition to `alive_health_ok_no_progress`;
- the frozen #196968 reproducer reaching its declared teardown marker; or
- an explicit operator command during a healthy control.

Native capture does not create the trigger condition.

### 5.2 Flow control

The capture coordinator keeps:

- one active capture slot per target process;
- a fixed capture timeout;
- a maximum raw-output size;
- a cooldown after any attempt;
- a maximum count per incident window.

Additional triggers return `capture_occupied` or `rate_limited`. They do not
queue unbounded work or attach again.

The `platform` constraint currently names the observer/producer platform. The
Stage B same-PID-namespace requirement makes it identical to the target host;
cross-host collection would require separate observer and target fields.

### 5.3 Clock and identity

The coordinator records its monotonic bounds around the producer invocation and
rechecks process start identity afterwards. Producer timestamps from another
clock domain remain producer-local unless an explicit mapping exists.

The post-capture identity recheck validates provenance only. If it fails, the
native capture is unbound and cannot emit attribution. It never asserts
`process_missing` and cannot override the process state recomputed by the v0.2
verifier.

Cross-host monotonic values are never directly compared. Cross-rank order uses
logical stage or collective sequence where available.

## 6. Producer evaluation plan

### PyStack

Evaluate first because it may provide mixed Python/native frames and GIL state
without rebuilding vLLM or PyTorch. The experiment records:

- supported target/runtime combinations;
- whether all threads or only a selected thread are available;
- Python/native frame availability;
- GIL state availability;
- attach pause behavior;
- ptrace permission failure;
- timeout and partial/empty output behavior;
- raw fields that cannot cross the privacy boundary.

Success means PyStack supplies a normalized fact required by a Block 4 row. It
does not mean PyStack becomes a mandatory dependency.

The released `py-spy` path and PyStack necessarily attach sequentially. The
target is therefore held in a deterministic blocked state, and one tool
brackets the other with a repeat capture, for example
`py-spy A -> PyStack B -> py-spy A2`. A/A2 must satisfy the same rule predicates
and emit the same `blocked_in`; otherwise the pairing result is
`target_not_stable` and says nothing about interchangeability.

If the stability control passes, comparison occurs at the rule-match level:
both tools must satisfy the same required predicates and emit the same
`blocked_in`, with coverage declared separately. Their normalized frame
sequences are not required to be identical because unwinders may differ in
inlined-frame recovery and demangling. Mock producers remain useful for
evaluator tests but do not satisfy interchangeability acceptance.

### Flight Recorder

Reuse existing per-rank dump identity, collective sequence, and completion
state. It remains the authority for facts present in its artifact, while an
absent artifact remains `producer_missing` until participant state is proven
separately.

### NCCL RAS

Evaluate only on a supported NCCL version and only for communicator/rank state
that can eliminate a named competing explanation. Unsupported versions are a
valid `unsupported` result. Do not build a new NCCL telemetry collector.

### Minimal C++ lifecycle probe

Design only if the three mature sources above cannot expose LLR-009. The probe
would emit closed stage transitions at existing lifecycle boundaries. It would
not unwind stacks, copy logs, sample arbitrary state, classify root cause, or
change shutdown behavior.

That decision has two scopes. A bounded lab-local measurement patch may test the
evidence relation after mature tools fail. An upstream-facing probe proposal is
separately blocked until #197232 receives an explicit maintainer outcome. A
triage label, passing CI, an open PR, or silence does not close that gate.

## 7. Verification model

The experimental verifier performs four independent checks:

1. **Integrity:** closed file set and content digests.
2. **Binding:** process identity and declared incident relation.
3. **Normalization:** input maps to the closed common shape.
4. **Attribution:** exactly one version-constrained rule matches, or the result
   is `unknown`.

It then displays the attribution beside the separately recomputed v0.2 verdict.
It never feeds the attribution into `derive_verdict()`.

Before normalization, it also enforces the closed `attempt_stage ×
outcome_code` matrix from LLR-005. Coordinator and preflight records never enter
producer-output normalization because no producer output exists.

## 8. Required tests

| Test | Required result |
| --- | --- |
| same normalized facts from mock producer A and B | identical evaluator output; evaluator purity only |
| all three attributions unmatched because no rule is admitted | `not_scorable`, never `interchangeable` |
| rule has no required frame or lifecycle predicate | verification failure |
| same-tool captures bracketing the other producer on a held target | same rule predicates and `blocked_in`, or `target_not_stable` with no interchangeability claim |
| real `py-spy` and PyStack captures after stability passes | same applicable rule predicates and `blocked_in`; frame sequences may differ; coverage is explicit |
| producer B omits GIL state | same blocked location, reduced coverage, GIL `unknown` |
| target-runtime version outside rule allowlist | `blocked_in = unknown` |
| frame shape unmatched | `blocked_in = unknown` |
| lifecycle order invalid | verification failure |
| fix-arm destroy-started and destroy-completed before stop-requested | valid lifecycle observation; no unpatched-order rejection |
| unpatched stop-requested before destroy-started | matches only an exact source-revision-constrained lifecycle rule |
| lifecycle facts attached to a stack observation | verification failure |
| process start identity changes after capture | native binding failure; no `process_missing` claim; v0.2 verdict unchanged |
| raw path/frame/stderr injected into public sidecar | schema failure |
| native sidecar removed from a sufficient v0.2 bundle | primary verdict unchanged |
| high GPU utilization with flat progress | no-progress verdict unchanged |
| second trigger while capture active | `capture_occupied`, no second attach |

## 9. Case-specific experiment sequence

1. **#53859 healthy/fault pair:** determine whether PyStack differentiates the
   queue-wait arm while leaving the existing verdict unchanged.
2. **#196968 healthy/fault pair:** determine whether mixed/native frames and GIL
   state narrow teardown attribution; compare with retained lifecycle flags and
   Flight Recorder output.
3. **#196996:** record a design conclusion that native capture is unnecessary
   unless a future competing explanation survives the existing dtype evidence.
4. **NCCL RAS:** run only if the available environment satisfies its version and
   permission requirements and it can answer a named #196968 communicator
   question.
5. **Probe decision:** record `join_contract_required` by default. Only after
   the identity/window join contract is reviewed may the case record
   `existing_tools_sufficient`, or name the single missing LLR-009 transition
   and design its smallest lab-local measurement patch. Do not prepare an
   upstream instrumentation proposal while #197232 lacks an explicit outcome.

## 10. Security and publication boundary

- private outputs use restrictive permissions and are excluded from Git;
- public records carry no raw frames, addresses, paths, command lines, prompts,
  response content, credentials, or arbitrary errors;
- producer binaries and raw output are content-addressed;
- the design provides tamper evidence, not authenticity against a malicious
  bundle editor;
- attaching to another process requires explicit operator authorization and is
  never the default `collect` behavior;
- published rules state their exact tested version and topology boundary.

## 11. Go/no-go decisions

Proceed from capability check to an adapter only if:

- at least one Block 4 ambiguity is reduced;
- the same fact has a negative control;
- degraded operation is typed and bounded;
- public normalization preserves the privacy boundary;
- real-producer interchangeability can be tested at the normalizer boundary.

Proceed from adapter evaluation to a C++ probe only if the LLR-009 gate in
[`LOW_LEVEL_CAPABILITY_REQUIREMENTS.md`](LOW_LEVEL_CAPABILITY_REQUIREMENTS.md)
is fully satisfied.

Otherwise the correct output is a documented negative design result: the
existing evidence is sufficient, or the remaining ambiguity cannot be closed
safely with the evaluated producer.

## 中文设计摘要

Block 5 先使用实验 sidecar，不修改 v0.2 bundle。coordinator、preflight 与 execution
使用关闭的 stage/outcome pair；只有 execution 才表示 producer 已调用。normalizer 将
工具输出转换成与 implementation 无关的 observation；规则可以约束 producer kind 和
target-runtime 版本，但不能约束 implementation name/version。现有 verifier 独立重算主
verdict，两条路径只在展示层并列，不能合并。

PyStack 首先用于验证 mixed Python/native frame 与 GIL state，并与真实 `py-spy` 路径在
同一稳定受控 target 上比较重叠 observation；Flight Recorder 继续负责
已有 collective 逻辑事实；NCCL RAS 只在版本支持且能排除一个已命名解释时评估。只有
三者都不能暴露 #196968 所需的 responder/shutdown 生命周期关系，才设计最小 C++ stage
probe。该 probe 默认是 lab-local measurement；upstream-facing 工作等待 #197232 明确
结果。未匹配版本或 frame 一律输出 `unknown`，原始栈、地址、路径和参数默认私有。
