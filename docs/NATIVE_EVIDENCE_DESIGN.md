# Native-state evidence design

Status: design candidate for Block 5 capability experiments. No implementation
or public schema migration is authorized by this document.

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
The attribution evaluator consumes normalized facts and rule-set identity, not
the implementation name.

### D-04 — classification is exact and version constrained

Rules are data. A rule applies only when every declared version/topology
constraint and ordered frame predicate matches. No substring-only root-cause
classification, fuzzy score, or nearest match is allowed.

### D-05 — attribution and verdict remain separate

Native state may explain an existing no-progress or producer-loss claim. It
cannot establish demand, service progress, process identity, health history, or
collective membership.

### D-06 — degraded operation is normal

Missing binaries, unsupported platforms, ptrace denial, timeout, occupied
capture slots, rate limiting, and empty output are valid terminal producer
outcomes. They reduce attribution coverage; they do not fail the primary
collector or become target-state evidence.

## 3. Experimental data model

The examples below are design shapes, not committed schemas.

### 3.1 Capture envelope

```json
{
  "schema_version": "native-evidence-experiment-v0",
  "subject": {
    "process_identity_kind": "linux_proc_start_ticks",
    "process_identity_value": "123456",
    "declared_role": "engine_core",
    "declared_rank": null
  },
  "window": {
    "start_monotonic_ns": 100,
    "end_monotonic_ns": 200
  },
  "producer": {
    "kind": "stack_snapshot",
    "implementation_name": "pystack",
    "implementation_version": "reviewed-version",
    "binary_sha256": "64 lowercase hex"
  },
  "outcome": {
    "state": "produced",
    "error_kind": null,
    "raw_output_sha256": "64 lowercase hex"
  }
}
```

`implementation_name` and `implementation_version` are shown to make the role
boundary explicit. If this shape becomes public, those values must be bounded
and classified as non-decisional provenance. Evaluators must not branch on
them. The current v0.2 optional stack adapter invokes `py-spy`; evaluating
PyStack in Block 5 does not silently replace that released adapter.

### 3.2 Normalized stack facts

Private normalization may produce multiple thread facts:

```json
{
  "thread_ref": "capture-local opaque id",
  "execution_domain": "mixed",
  "gil_state": "waiting",
  "native_frame_shape": [
    "module:symbol-class",
    "module:symbol-class"
  ],
  "python_frame_shape": [
    "package:function-class"
  ]
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

### 3.3 Lifecycle facts

Lifecycle facts are distinct from stack classifications:

```json
{
  "component": "process_group_nccl",
  "stage": "dump_responder_stopped",
  "logical_sequence": 3,
  "observed": true
}
```

Closed candidate stages for the #196968 experiment are:

```text
dump_responder_active
dump_responder_stopped
communicator_destroy_started
communicator_destroy_completed
peer_dump_request_observed
dump_completed
```

The evaluator rejects duplicate logical sequence positions, impossible order,
unknown stages, and identity changes.

### 3.4 Public attribution projection

```json
{
  "attribution_schema_version": "native-attribution-v0",
  "subject_binding_digest": "64 lowercase hex",
  "capture_outcome": "produced",
  "blocked_in": "communicator_destruction",
  "execution_domain": "mixed",
  "gil_state": "unknown",
  "rule_set_id": "pytorch-pg-nccl-legacy-shutdown-v1",
  "rule_match": "exact",
  "coverage": ["thread_state", "lifecycle_stage"],
  "raw_output_sha256": "64 lowercase hex"
}
```

Allowed `blocked_in` values are defined by LLR-006. `rule_match` is
`exact | unmatched`; `unmatched` requires `blocked_in = unknown`. Coverage is a
closed set, not a confidence score.

## 4. Rule representation

A rule record contains only declarative constraints:

```yaml
rule_set_id: pytorch-pg-nccl-legacy-shutdown-v1
applies_to:
  platform: linux
  pytorch_backend: nccl-legacy
  pytorch_revision_range: explicitly-reviewed-range
  nccl_version_range: explicitly-reviewed-range
requires:
  ordered_frame_classes:
    - process-group-destroy
    - communicator-destroy-or-abort
  lifecycle_stages:
    - dump_responder_stopped
    - communicator_destroy_started
forbids:
  lifecycle_stages:
    - communicator_destroy_completed
emits:
  blocked_in: communicator_destruction
```

The actual version ranges shall be filled only after Block 5 captures are
reviewed. A missing version, topology mismatch, missing required fact, forbidden
fact, or frame mismatch produces `unknown`.

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

### 5.3 Clock and identity

The coordinator records its monotonic bounds around the producer invocation and
rechecks process start identity afterwards. Producer timestamps from another
clock domain remain producer-local unless an explicit mapping exists.

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

## 7. Verification model

The experimental verifier performs four independent checks:

1. **Integrity:** closed file set and content digests.
2. **Binding:** process identity and declared incident relation.
3. **Normalization:** input maps to the closed common shape.
4. **Attribution:** exactly one version-constrained rule matches, or the result
   is `unknown`.

It then displays the attribution beside the separately recomputed v0.2 verdict.
It never feeds the attribution into `derive_verdict()`.

## 8. Required tests

| Test | Required result |
| --- | --- |
| same normalized facts from mock producer A and B | identical attribution |
| producer B omits GIL state | same blocked location, reduced coverage, GIL `unknown` |
| version outside rule range | `blocked_in = unknown` |
| frame shape unmatched | `blocked_in = unknown` |
| lifecycle order invalid | verification failure |
| process start identity changes after capture | binding failure |
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
5. **Probe decision:** either record `existing_tools_sufficient`, or name the
   single missing LLR-009 transition and design its smallest source patch.

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
- producer interchangeability can be tested.

Proceed from adapter evaluation to a C++ probe only if the LLR-009 gate in
[`LOW_LEVEL_CAPABILITY_REQUIREMENTS.md`](LOW_LEVEL_CAPABILITY_REQUIREMENTS.md)
is fully satisfied.

Otherwise the correct output is a documented negative design result: the
existing evidence is sufficient, or the remaining ambiguity cannot be closed
safely with the evaluated producer.

## 中文设计摘要

Block 5 先使用实验 sidecar，不修改 v0.2 bundle。采集工具只负责生成有界私有输出和
类型化 producer outcome；normalizer 将其转换成与工具无关的 observation；带显式版本
约束的数据规则产生关闭 attribution；现有 verifier 独立重算主 verdict。两条路径只在
展示层并列，不能合并。

PyStack 首先用于验证 mixed Python/native frame 与 GIL state；Flight Recorder 继续负责
已有 collective 逻辑事实；NCCL RAS 只在版本支持且能排除一个已命名解释时评估。只有
三者都不能暴露 #196968 所需的 responder/shutdown 生命周期关系，才设计最小 C++ stage
probe。未匹配版本或 frame 一律输出 `unknown`，原始栈、地址、路径和参数默认私有。
