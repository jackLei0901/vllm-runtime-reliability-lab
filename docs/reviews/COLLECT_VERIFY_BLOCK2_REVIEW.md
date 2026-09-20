# Collect/Verify v0.2 Block 2 review entry / Block 2 中英 Review 入口

Status: test contract ready for review; no progress or verdict implementation
has been added yet.

状态：测试合同已可审查；尚未加入 progress 或 verdict 实现。

## Review scope / 审查范围

Block 2 converts the frozen design into implementation-independent test vectors.
The matrix covers producer evaluation, demand evidence, verdict precedence,
scope conflicts, PID identity, cached samples, and stack invariance. Its pinned
`case_count` is 32: 9 producer, 7 demand, and 16 verdict cases.

Block 2 将冻结设计转成与实现无关的测试向量。矩阵覆盖 producer 判定、demand
证据、verdict 优先级、scope 冲突、PID identity、缓存样本和 stack 不变量。
`case_count` 固定为 32：9 个 producer、7 个 demand、16 个 verdict 用例。

Review these files in order:

1. [`../../docs/COLLECT_VERIFY_V0_2.md`](../COLLECT_VERIFY_V0_2.md) — normative
   contract / 规范合同；
2. [`../../tests/fixtures/collect_verify_v0_2/cases.json`](../../tests/fixtures/collect_verify_v0_2/cases.json)
   — executable input/output matrix / 可执行输入输出矩阵；
3. [`../../tests/test_collect_verify_contract_cases.py`](../../tests/test_collect_verify_contract_cases.py)
   — closed-shape and consistency checks / 封闭结构与一致性检查。

## Decisions to review / 需要确认的决策

### English

- A flat server counter without admitted work is `undetermined`, not no-progress.
- The selected `decision_source` determines scope; the other producer is only
  corroborating evidence.
- A stuck client request may be request-scoped no-progress while the global
  server counter continues to increase. The conflict remains explicit.
- Counter reset, one sample, cached-only repetition, and an early-completed
  client request are `insufficient_evidence`.
- An endpoint that was never healthy cannot yield `health_lost`, and endpoint-only
  flat progress cannot establish process liveness in v0.2.
- Cached demand values do not count as fresh evidence, and a client request must
  begin before the evaluation interval.
- PID reuse has the same top-level verdict as disappearance of the original
  process: `process_missing`.
- Process loss outranks health loss; health loss outranks observed progress.
- Stack availability never changes the verdict.
- PID-only survival is `undetermined`; PID-only disappearance is
  `process_missing`.

### 中文

- server counter 保持不变但没有 admitted work 时，结果是 `undetermined`，
  不是 no-progress。
- 只有选定的 `decision_source` 决定判定范围；另一个 producer 只作为旁证。
- 单个 client request 可以在全局 server counter 仍增长时形成 request-scope
  no-progress，但必须显式保留冲突。
- counter reset、单样本、只有缓存重复值，以及提前完成的 client request 都属于
  `insufficient_evidence`。
- 从未成功 health 的 endpoint 不能得到 `health_lost`；endpoint-only 的 flat
  progress 在 v0.2 中也不能证明 process liveness。
- 缓存的 demand 值不算新证据，client request 必须在 evaluation interval 开始前
  已经启动。
- PID 被复用与原进程消失使用同一个顶层 verdict：`process_missing`。
- process loss 优先于 health loss，health loss 优先于 observed progress。
- stack 是否可用不能改变 verdict。
- PID-only 模式下，进程持续存活得到 `undetermined`，原进程消失得到
  `process_missing`。

## Deliberately not implemented / 本阶段刻意不实现

- no HTTP, process, or stack collection;
- no `progress.py` or `verify_bundle.py` behavior;
- no CLI wiring;
- no mutation of `external-runtime-observation-v1`;
- no conversion of the published #53859 R3 summary into a native v0.2 bundle.

The matrix is green because Block 2 validates that the test contract is closed,
complete, and internally consistent. Block 3 will execute the same vectors
against `progress.py`; it must not replace them with implementation-shaped tests.

矩阵当前通过，是因为 Block 2 验证测试合同本身封闭、完整且内部一致。Block 3
会让 `progress.py` 执行同一组向量，不能用贴合实现的测试替换这些合同向量。

## Review commands / 审查命令

```bash
python -m json.tool tests/fixtures/collect_verify_v0_2/cases.json > /dev/null
python -m unittest tests.test_collect_verify_contract_cases -v
python -m unittest discover -s tests -v
python -m compileall -q src tests
ruff check tests/test_collect_verify_contract_cases.py
git diff --check
```

PowerShell JSON check:

```powershell
Get-Content tests/fixtures/collect_verify_v0_2/cases.json -Raw |
  ConvertFrom-Json | Out-Null
```

## Block 3 entry condition / Block 3 开始条件

Begin implementation only after reviewers agree on the expected outcomes in
the matrix, especially the request-scoped conflict case and the verdict
precedence cases.

只有在 reviewer 同意矩阵中的预期结果后才开始实现，尤其需要确认 request-scope
冲突和 verdict precedence 相关用例。
