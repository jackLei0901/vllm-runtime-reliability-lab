# Collect/Verify v0.2 Block 3 review entry / Block 3 中英 Review 入口

Status: the frozen Block 2 matrix now drives a pure, fail-closed decision core.
Collection, bundle verification, stacks, and CLI wiring remain out of scope.

状态：Block 2 冻结矩阵现已驱动一个纯函数、fail-closed 的判定核心。采集、
bundle 验证、stack 和 CLI 接线仍不在本阶段范围内。

## Review scope / 审查范围

Review these files in order:

1. [`../COLLECT_VERIFY_V0_2.md`](../COLLECT_VERIFY_V0_2.md) — normative
   contract / 规范合同；
2. [`../../tests/fixtures/collect_verify_v0_2/cases.json`](../../tests/fixtures/collect_verify_v0_2/cases.json)
   — the 32 frozen input/output vectors / 32 个冻结输入输出向量；
3. [`../../src/dfxlab/progress.py`](../../src/dfxlab/progress.py) — pure producer,
   demand, and verdict evaluation / 纯 producer、demand 与 verdict 判定；
4. [`../../tests/test_progress.py`](../../tests/test_progress.py) — direct execution
   of the frozen inputs without an adapter / 不经过 adapter 直接执行冻结输入。

## Implemented behavior / 已实现行为

### English

- `evaluate_producer()` keeps server-counter and client-request evidence
  separate and returns only the frozen producer states and reasons.
- `evaluate_demand()` rejects cached-only server values and client requests
  that did not span the evaluation interval.
- `derive_verdict()` applies the frozen precedence:
  `process_missing` > `health_lost` > `progress_observed` >
  `alive_health_ok_no_progress` > `undetermined`.
- The decision producer owns the progress scope. A corroborating producer can
  set `producer_conflict`, but cannot silently change the verdict scope.
- Endpoint-only no-progress remains `undetermined` in v0.2 because endpoint
  health is not treated as process-liveness evidence.
- Stack state is validated but cannot affect the verdict.
- Unknown fields, invalid vocabulary, malformed intervals, and inconsistent
  target state raise `ProgressContractError` rather than being guessed.

### 中文

- `evaluate_producer()` 始终分开处理 server counter 与 client request，只返回
  冻结合同中的 producer state 和 reason。
- `evaluate_demand()` 拒绝仅有缓存值的 server 证据，也拒绝未覆盖完整判定区间的
  client request。
- `derive_verdict()` 按冻结优先级执行：`process_missing` > `health_lost` >
  `progress_observed` > `alive_health_ok_no_progress` > `undetermined`。
- decision producer 决定 progress scope；corroborating producer 可以产生
  `producer_conflict`，但不能静默改变 verdict scope。
- v0.2 中 endpoint-only no-progress 仍为 `undetermined`，因为 endpoint health
  不被当作进程存活证据。
- stack state 会被校验，但不能影响 verdict。
- 未知字段、非法词汇、错误 interval 和不一致 target 状态都会抛出
  `ProgressContractError`，不会猜测或自动修正。

## Deliberately not implemented / 本阶段刻意不实现

- no recorder, HTTP, process, metrics, or stack I/O;
- no bundle schema or `verify_bundle.py`;
- no `collect` or `verify` CLI subcommands;
- no conversion of the published #53859 R3 evidence;
- no mutation of `external-runtime-observation-v1`.

Block 4 must assemble evidence into a bundle without weakening these pure
rules. The published #53859 Stage 1 R3 result remains the compatibility gate:
if a truthful conversion cannot preserve its existing conclusion, the new
bundle contract must be corrected instead of rewriting the historical result.

Block 4 必须在不削弱这些纯判定规则的前提下组装 evidence bundle。已发布的
#53859 Stage 1 R3 仍是兼容性 gate：如果诚实转换无法保留既有结论，应修正新
bundle 合同，而不是改写历史结果。

## Review commands / 审查命令

```bash
python -m unittest tests.test_collect_verify_contract_cases tests.test_progress -v
python -m unittest discover -s tests -v
python -m compileall -q src tests
ruff format --check src/dfxlab/progress.py tests/test_progress.py
ruff check src/dfxlab/progress.py tests/test_progress.py
git diff --check
```
