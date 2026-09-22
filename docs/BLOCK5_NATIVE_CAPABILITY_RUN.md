# Block 5 native capability run

Status: Stage A real-producer smoke and Stage B v2 scored GPU validation passed.
The original retained-environment route remains closed at its build-identity
gate; Stage B v2 is a separately identified reviewed baseline.

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
The first Stage B entry used the retained #53859 GPU host, but the original
dependency-pool installation target had been deleted. Reconstructing the same
package versions did not reproduce the frozen RECORD identities: both arms had
144 RECORD SHA-256 mismatches, six verified-file-count mismatches, and one
extra distribution. The model and server were therefore never started. The
public preflight result is
[`../results/native-stack-pair-stage-b-preflight-20260922/README.md`](../results/native-stack-pair-stage-b-preflight-20260922/README.md).

Stage B v2 then created and restore-tested a new immutable environment archive,
froze the model revision and per-file hashes, and ran the #53859 healthy/fault
pair on one RTX 4090. The scored result is published at
[`../results/native-stack-pair-stage-b-v2-20260922/README.md`](../results/native-stack-pair-stage-b-v2-20260922/README.md):

- C0 completed 64 tokens with zero dropped batches; all three captures were
  usable but did not satisfy the fault-only rule;
- F1 kept `/health` at 2xx while progress stalled; py-spy A, PyStack B, and
  py-spy A2 all exactly satisfied the admitted `publisher -> queue put ->
  condition wait` predicates and emitted `blocked_in=queue_wait`;
- A/A2 stability passed, the cross-producer result was `interchangeable`, and
  frame-sequence equality was explicitly not compared;
- cleanup removed both EngineCore subjects and process groups, and post-run
  base/fix build identities matched the pre-run restored identities.

The scored execution is pinned to commit `456d425`, which temporarily added an
explicit Stage-B-only control-capture option to the campaign process so its
children inherited the EngineCore's scoped ptrace authorization. The main
branch restores the frozen Stage 1 campaign immediately afterward; this keeps
the published Stage 1 R3 implementation hash verifiable. Reproduction of this
specific Stage B result must therefore check out the pinned execution commit.

Raw stacks and server logs remain private. The public evidence contains typed
capture outcomes, bounded provenance, raw digests, normalized attribution, and
the scored comparison only. Stage C remains gated on a separately reviewed
multi-producer join contract; an upstream-facing C++ probe still requires an
explicit #197232 outcome.

The environment decision and next contract are recorded separately:

- [`STAGE_B_IDENTITY_POSTMORTEM.md`](STAGE_B_IDENTITY_POSTMORTEM.md) explains
  why matching distribution versions did not restore the reviewed identity;
- [`../experiments/native-evidence-capability/STAGE_B_V2_PREREGISTRATION.md`](../experiments/native-evidence-capability/STAGE_B_V2_PREREGISTRATION.md)
  freezes the permitted claims, required artifacts, C0/F1 cells, and stop rules
  for a new reviewed baseline;
- [`MATCHING_VERSIONS_ARE_NOT_RUNTIME_IDENTITY.md`](MATCHING_VERSIONS_ARE_NOT_RUNTIME_IDENTITY.md)
  and its [Chinese version](MATCHING_VERSIONS_ARE_NOT_RUNTIME_IDENTITY.zh-CN.md)
  are publication drafts, not additional experimental evidence.

## Review commands

```bash
python -m unittest \
  tests.test_native_evidence \
  tests.test_native_producers \
  tests.test_stage_b_pair_adapter \
  tests.test_stage_b_pair_scoring \
  tests.test_vllm_zmq_stage1 -v
python -m compileall -q src tests \
  experiments/native-evidence-capability
ruff check \
  src/dfxlab/native_evidence.py \
  src/dfxlab/native_producers.py \
  tests/test_native_evidence.py \
  tests/test_native_producers.py \
  tests/test_stage_b_pair_scoring.py \
  experiments/native-evidence-capability/run_stack_pair.py \
  experiments/native-evidence-capability/score_stage_b_pair.py
```

## 中文审阅摘要

Block 5 Stage B v2 已在真实 RTX 4090 环境完成 #53859 healthy/fault pair。C0 三次
采集均未命中 fault-only rule；F1 的 py-spy A、PyStack B、py-spy A2 均精确命中
`publisher -> queue put -> condition wait`，输出 `blocked_in=queue_wait`，A/A2
稳定性通过，cross-producer comparison 为 `interchangeable`。实验 sidecar 仍与
v0.2 verdict 完全隔离，原始 stack 与 server log 不公开，运行后 build identity 与
运行前一致。Stage C 仍需要单独评审 join contract；C++ probe 仍受成熟工具不足证明
和 #197232 upstream outcome 双重 gate 约束。
