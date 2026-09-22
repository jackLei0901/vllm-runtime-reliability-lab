# Block 5 native capability run

Status: Stage A real-producer smoke passed on Linux CPU; Stage B retained-vLLM
execution pending a GPU environment.

This block evaluates whether existing attach tools can supply the lower-level
facts admitted by Block 4. It does not add native state to the v0.2 verdict,
does not publish raw stacks, and does not authorize a C++ probe.

## Stage A — executable contract

Implemented in:

- `src/dfxlab/native_evidence.py`: closed capture validation, exact
  version-constrained attribution, and A/B/A2 comparison;
- `src/dfxlab/native_producers.py`: Linux-only bounded `py-spy` and PyStack
  acquisition into a private file;
- `experiments/native-evidence-capability/run_stack_pair.py`: fixed
  `py-spy A -> PyStack B -> py-spy A2` capture sequence;
- `tests/test_native_evidence.py` and `tests/test_native_producers.py`: contract,
  isolation, degraded-operation, and pairing tests.

No target-runtime rule ships in Stage A. A rule can be admitted only after the
real healthy/fault captures establish an exact normalized shape and version
boundary. Until then every captured record ends with no attribution claim, and
three `unmatched` attributions evaluate to `not_scorable`, never
`interchangeable`.

The first contract review also fixed five pre-run failure modes: vacuous
interchangeability without a rule, predicate-free universal rules,
stack/lifecycle provenance conflation, case-sensitive version provenance, and
output-budget exhaustion being collapsed into generic execution failure.

## Stage B — #53859 retained reproducer

Run against the existing Stage 1 environment; do not create a new fault area.

| Cell | Target state | Sequence | Required result |
| --- | --- | --- | --- |
| C0 | unpaused healthy control | A/B/A2 | captures produced; no fault-only predicate |
| F1 | base/pause queue backpressure | A/B/A2 | A/A2 stable; B compared at rule-predicate level |

Pre-register before execution:

- exact vLLM, PyTorch, Python, producer, platform, and topology versions;
- EngineCore PID and `/proc/<pid>/stat` start ticks;
- 5-second per-producer timeout, a separate bounded cleanup allowance, and a
  1 MiB output limit unless the environment requires smaller bounds;
- the existing queue-wait hypothesis only:
  `publisher -> queue put -> condition wait`;
- raw output location under an excluded private directory.

The run is not scorable when:

- the target identity changes;
- any capture is not `execution × produced` (`not_scorable`, not an exception);
- A and A2 do not satisfy the same applicable predicates and attribution;
- the held marker is released during the three captures;
- a target or producer version was not frozen before execution.

Different frame sequences, demangling, or inlined-frame recovery do not fail
the comparison. The comparison asks whether the same admitted rule predicates
are satisfied and whether `blocked_in` agrees. Coverage is reported separately.

## Stage C — #196968 retained reproducer

Only after Stage B validates the acquisition/normalization pipeline:

1. capture healthy and teardown-held states;
2. compare mixed/native and GIL facts with the retained Flight Recorder and
   lifecycle-flag evidence;
3. decide whether mature producers expose the responder/shutdown relation in
   LLR-009;
4. record `join_contract_required` as the default exit. Only after the
   identity/window join contract is reviewed may the case conclude
   `existing_tools_sufficient`, or name exactly one missing lifecycle
   transition for a bounded lab-local probe.

An upstream-facing probe remains blocked until #197232 has an explicit
maintainer outcome. Open status, CI, labels, or silence do not satisfy that
gate.

Stack and lifecycle evidence remain separate producer records. Stage C must
define and review an explicit identity/window join whose public result retains
both source digests. A stack capture cannot carry lifecycle flags, and its raw
digest cannot vouch for them. Until that review is complete, Stage C is
non-scorable and its only permitted conclusion is `join_contract_required`.

## Current environment result

The Windows development host still uses mocks for contract regression. A
Linux CPU-only host then ran the real producer sequence against a deterministic
condition-wait target at
[`../results/native-stack-pair-stage-a-cpu-20260922/README.md`](../results/native-stack-pair-stage-a-cpu-20260922/README.md).
With target-scoped ptrace authorization, `py-spy 0.4.2`, PyStack `1.7.1`, and
the repeated `py-spy` capture all produced bounded private output while process
identity remained stable. The repeated `py-spy` raw digests were identical.

This closes only the Linux acquisition smoke gate. The public record remains
`normalization_pending` with `claim = null`: no target-runtime rule was admitted
and no producer-interchangeability or vLLM fault-attribution claim is made.
Real Stage B still requires the retained #53859 healthy/fault pair on a GPU
environment. Stage C remains gated behind Stage B and the reviewed join
contract.

## Review commands

```bash
python -m unittest \
  tests.test_native_evidence \
  tests.test_native_producers -v
python -m compileall -q src tests \
  experiments/native-evidence-capability
ruff check \
  src/dfxlab/native_evidence.py \
  src/dfxlab/native_producers.py \
  tests/test_native_evidence.py \
  tests/test_native_producers.py \
  experiments/native-evidence-capability/run_stack_pair.py
```

## 中文审阅摘要

Block 5 Stage A 已把底层证据合同变成可执行代码，但尚未声称 PyStack 或 `py-spy`
能够解释真实 vLLM fault。实验 sidecar 与 v0.2 verdict 完全隔离；stage/outcome 在
normalization 前校验；PID start ticks 在采集前后核对；原始输出只进入 private 目录；
implementation name/version 不参与 attribution；未匹配版本或 frame 必须输出
`unknown`。真实验证首先复用 #53859 Stage 1 的 healthy/fault pair，并采用
`py-spy A -> PyStack B -> py-spy A2` 稳定性控制。只有该管线通过后，才进入
#196968 lifecycle gap；C++ probe 仍受成熟工具不足证明和 #197232 upstream outcome
双重 gate 约束。
