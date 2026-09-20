# Collect/Verify v0.2 Blocks 4–6 unified review / Block 4–6 统一中英 Review

Status: implementation complete for repository review; untagged and
unreleased.

状态：仓库内实现已完成，可统一审查；尚未打 tag 或发布 release。

## Outcome / 结果

The frozen Block 2 matrix still owns the claim vocabulary, and Block 3 remains
the only progress/verdict decision core. Blocks 4–6 add evidence transport and
execution around it without introducing another classifier.

Block 2 冻结矩阵继续决定 claim vocabulary，Block 3 仍是唯一的 progress/verdict
判定核心。Block 4–6 只在其外部增加 evidence transport 和执行层，没有引入第二套
classifier。

## Block 4 — native bundle and verifier / 原生 Bundle 与 Verifier

Review:

1. [`../../src/dfxlab/bundle.py`](../../src/dfxlab/bundle.py)
2. [`../../src/dfxlab/verify_bundle.py`](../../src/dfxlab/verify_bundle.py)
3. [`../../tests/test_bundle.py`](../../tests/test_bundle.py)

Key properties / 关键性质：

- `observations.json` is written first; `summary.json` is derived from its exact
  SHA-256 identity / 先写 observation，再由其准确 SHA-256 派生 summary；
- `verify` recomputes producer, demand, conflict, scope, health, PID identity,
  and verdict / verifier 会重新计算全部 claim，不信任 summary 中的结论；
- unknown shapes, duplicate JSON keys, unexpected public files, privacy-boundary
  fields, and semantic divergence fail closed / 未知结构、重复 key、额外公开文件、
  隐私越界字段和语义分歧全部 fail closed；
- native synthetic R3 facts yield `alive_health_ok_no_progress` for base and
  `progress_observed` for fix / native synthetic R3 base/fix 得到预期 verdict；
- the immutable public R3 result remains legacy evidence. Its projection sets
  `native_v0_2_verdict: null` and reports missing PID identity, repeated health
  samples, and server-counter samples / 旧 R3 不会被重标为 native v0.2。

## Block 5 — optional stack producer / 可选 Stack Producer

Review:

1. [`../../src/dfxlab/stacks.py`](../../src/dfxlab/stacks.py)
2. [`../../tests/test_stacks.py`](../../tests/test_stacks.py)

`py-spy` execution uses a fixed command shape. Raw output is written only under
`private/`; the public section retains sampler name/version, binary digest,
platform attach context, exit status, output digest, and a bounded error kind.
Missing permission, binary, timeout, execution failure, or empty output remains
visible as `unavailable`. Stack state is validated but cannot change a verdict.

`py-spy` 使用固定命令结构。原始输出只写入 `private/`；公开部分仅保留 sampler
名称/版本、binary digest、platform attach context、退出状态、输出 digest 和有界
error kind。权限、binary、timeout、执行失败或空输出都会显式成为 `unavailable`。
Stack 状态可被校验，但不能改变 verdict。

## Block 6 — bounded collection and CLI / 有界采集与 CLI

Review:

1. [`../../src/dfxlab/collect_bundle.py`](../../src/dfxlab/collect_bundle.py)
2. [`../../src/dfxlab/cli.py`](../../src/dfxlab/cli.py)
3. [`../../tests/test_collect_bundle.py`](../../tests/test_collect_bundle.py)
4. [`../COLLECT_VERIFY_V0_2.md`](../COLLECT_VERIFY_V0_2.md)

The collector supports process+endpoint, endpoint-only, explicit PID-only
observation, server-counter decisions, opt-in streaming client-request
decisions, and opt-in stack sampling. Linux start ticks and Windows process
creation time prevent PID reuse from inheriting evidence. Endpoint IDs are
per-run fingerprints; URLs and request/response content are not published.

Collector 支持 process+endpoint、endpoint-only、显式 PID-only observation、
server-counter 判定、opt-in streaming client-request 判定和 opt-in stack。
Linux start ticks 与 Windows creation time 防止 PID reuse 继承旧证据。Endpoint
ID 每次运行独立生成；URL 和 request/response 内容不会进入公开 bundle。

CPU-only end-to-end tests cover:

- alive process + all-2xx health + flat token counter + continuous demand;
- conservative endpoint-only no-progress;
- request-scoped content progress without retaining content;
- PID-only observation;
- offline CLI verification and invalid-mode rejection.

## Deliberate boundaries / 刻意保留的边界

- no target discovery, remediation, restart, or multi-node joiner;
- no claim that bundle hashes authenticate against a malicious editor;
- no arbitrary stack frames in public evidence;
- no claim that endpoint health proves process liveness;
- no conversion of legacy R3 into a native bundle without the missing evidence;
- no release/tag in this block.

## Unified review commands / 统一审查命令

Current local result: 203 tests passed, 2 platform-specific tests skipped;
Ruff, `compileall`, and `git diff --check` passed. The installed editable
`vllm-dfx --help` exposes both new commands. A new wheel was not claimed: the
task virtual environment has neither `setuptools` nor `wheel`, and network
installation was intentionally not used.

当前本地结果：203 个测试通过，2 个平台相关测试跳过；Ruff、`compileall` 和
`git diff --check` 均通过。当前 editable install 的 `vllm-dfx --help` 已显示两个
新命令。本轮不声称完成了新 wheel 构建：任务虚拟环境没有 `setuptools`/`wheel`，
且没有为此启用网络安装。

```bash
python -m unittest \
  tests.test_collect_verify_contract_cases \
  tests.test_progress \
  tests.test_bundle \
  tests.test_stacks \
  tests.test_collect_bundle -v
python -m unittest discover -s tests -v
python -m compileall -q src tests
ruff format --check \
  src/dfxlab/{progress,bundle,verify_bundle,stacks,collect_bundle,cli,collectors}.py \
  tests/test_{collect_verify_contract_cases,progress,bundle,stacks,collect_bundle,collectors}.py
ruff check src tests
git diff --check
```

The release gate after this review is a clean install test plus one external
run. Upstream PR status is independent: #197232 can still gate publication of
its case study without blocking use of collect/verify.

本轮 review 后的 release gate 是 clean install 测试和至少一次外部运行。
#197232 仍可独立 gate 对应 case study 的发布，但不阻塞 collect/verify 被试用。
