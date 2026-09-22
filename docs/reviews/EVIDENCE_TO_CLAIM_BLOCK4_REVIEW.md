# Block 4 evidence-to-claim review / Block 4 证据到结论统一 Review

Status: complete for repository review; no runtime implementation and no new
upstream claim.

状态：仓库内文档产物已完成，可开始审查；本 Block 不包含运行时代码，也没有新增
upstream claim。

## Review order / 审查顺序

1. [`../EVIDENCE_TO_CLAIM_BLOCK4.md`](../EVIDENCE_TO_CLAIM_BLOCK4.md) — the
   normative three-case claim tables / 三案规范化 claim 表；
2. [`../LOW_LEVEL_CAPABILITY_REQUIREMENTS.md`](../LOW_LEVEL_CAPABILITY_REQUIREMENTS.md)
   — requirements derived from unresolved distinctions / 从剩余判定缺口反推的底层能力需求；
3. [`../NATIVE_EVIDENCE_DESIGN.md`](../NATIVE_EVIDENCE_DESIGN.md) — proposed
   Block 5 sidecar and producer-evaluation design / Block 5 sidecar 与 producer
   评估设计。

Read the claim tables first. If a capability cannot point to a row it makes
more discriminating, reject the capability before reviewing its mechanics.

先审 claim 表。如果某项能力不能指出它让哪一行判断更可区分，应在讨论实现方式之前
直接拒绝该能力。

## Block 4 outcome / Block 4 结果

The three existing cases now use one review shape:

三个现有案例现在统一使用以下审查结构：

```text
observation
  -> allowed inference
  -> forbidden inference
  -> required corroboration
  -> contradiction
  -> verdict or insufficient_evidence
```

| Case / 案例 | Bounded claim / 有界结论 | Lower-level decision / 底层能力结论 |
| --- | --- | --- |
| PyTorch #196968 / proposed #197232 | a peer dump request existed; rank 1 remained present while its legacy Flight Recorder producer was unavailable during teardown; missing dump did not prove missing participation / peer dump request 确实存在；rank 1 仍在 teardown 中存活，但 dump producer 已不可用；缺 dump 不能证明未参与 | lifecycle relation may require one minimal stage probe, but only after PyStack, Flight Recorder and NCCL RAS are shown insufficient / 生命周期关系可能需要一个最小 stage probe，但必须先证明成熟工具不足 |
| vLLM #53859 / proposed #53883 | one real EngineCore was alive and health-responsive while admitted work stopped; proposed fix restored liveness with four dropped batches / 真实 EngineCore 存活且 health 2xx，但已进入的请求停止推进；提议修复恢复活性并丢弃四个 batch | no native fact is required for the verdict; native state is attribution only / 主 verdict 不需要 native fact；底层状态只做归因 |
| PyTorch #196996 | the mixed-gradient-dtype correctness failure reproduces on one GPU and can present as a distributed hang when ranks diverge / mixed-gradient-dtype correctness failure 可单卡复现，并可在 rank 分歧时表现为分布式 hang | existing structured dtype and assertion evidence is sufficient; no generic native probe / 现有 dtype 与 assertion 结构化证据已足够，不需要通用 native probe |

## Post-review corrections / Review 后修正

The review of commit `02bb428` produced seven accepted findings. This revision
resolves them as follows:

对 `02bb428` 的 review 提出了七项有效问题，本次逐项处理如下：

1. **Outcome categories:** replaced one flat enum with a closed
   `attempt_stage × outcome` matrix for coordinator, preflight, and execution /
   将扁平枚举改为 coordinator、preflight、execution 三类互斥 pair；
2. **Rule boundary:** rules may constrain producer kind but never implementation
   name/version; implementation compatibility stops at the normalizer /
   attribution rule 只约束 producer kind，implementation name/version 止于 normalizer；
3. **Interchangeability:** mock fixtures now test evaluator purity only; Block 5
   requires a real `py-spy`/PyStack pair on one stable controlled target /
   mock 只验证 evaluator purity，正式验收要求真实双 producer 对照；
4. **Identity recheck:** a failed post-capture recheck invalidates native
   provenance and never emits `process_missing` / capture 后 identity 失败只使 native
   provenance 无效，不产生主 verdict；
5. **Vocabulary admission:** only `queue_wait`,
   `communicator_destruction`, and `unknown` are emittable; four broader names
   are reserved until a case row admits them / 当前只允许两个已命名分类与 `unknown`，其余
   名称保留但不可输出；
6. **Probe sequencing:** lab-local measurement and upstream instrumentation are
   separate; upstream work waits for an explicit #197232 outcome / lab-local
   measurement 与 upstream instrumentation 分离，后者等待 #197232 明确结果；
7. **Verification environment:** the test count now names OS, Python,
   `jsonschema`, and the absent `py-spy` executable, and the review command
   installs the development extra / 测试数量补充完整环境与 dev-extra 前提。

A follow-up review added four contract corrections and two test/CI notes:

后续 review 又补充了四项 contract 修正和两项 test/CI 说明：

1. `permission_denied` is execution-only; Yama and other preflight permission
   hints are provenance and cannot gate attach / `permission_denied` 只能来自实际
   execution，Yama 等预检信息不阻止 attach；
2. opt-out is the explicit pair `not_requested × disabled`, with no raw-output
   digest / opt-out 使用显式 pair，且不存在 raw-output digest；
3. sequential producer comparison is bracketed by a same-tool stability
   control; instability terminates the comparison without a claim / 顺序双工具对照先做
   同工具稳定性控制，不稳定时不产生 interchangeability 结论；
4. cross-tool acceptance compares applicable rule predicates and `blocked_in`,
   not normalized frame-sequence equality / 跨工具验收比较 rule predicate 与
   `blocked_in`，不要求 frame sequence 完全相同；
5. CPython 3.14.2 is explicitly a local regression run outside the release CI
   matrix / CPython 3.14.2 明确属于 release CI matrix 之外的本地回归；
6. `test_external_schema` now skips its jsonschema-dependent class when the dev
   extra is absent, so `test_external_writer` no longer fails transitively while
   importing its shared fixture / 缺少 dev extra 时 external-schema 测试按类 skip，
   external-writer 不再因共享 fixture import 连带报错。

## Decisions to approve / 需要确认的设计决策

### 1. Verdict evidence and attribution remain separate

Native stacks, GIL state, NCCL RAS, lifecycle stages, and hardware context do
not change the v0.2 verdict. They may produce a separately versioned
attribution or reduce attribution coverage.

native stack、GIL state、NCCL RAS、lifecycle stage 和硬件上下文不改变 v0.2
verdict。它们只能产生独立版本的 attribution，或降低 attribution coverage。

### 2. Block 5 starts with a sidecar

Capability experiments do not modify `observations.json`, `summary.json`, the
field-role registry, or `derive_verdict()`. Any future integration requires a
new schema review and mutation tests.

能力实验不修改 `observations.json`、`summary.json`、field-role registry 或
`derive_verdict()`。未来如需集成，必须单独完成新 schema 审查和 mutation tests。

### 3. Attempt stage and outcome are separate

Explicit opt-out is `not_requested × disabled`. Coordinator decisions
(`capture_occupied`, `rate_limited`), preflight results
(`unsupported`, `binary_missing`, `feature_disabled`), and execution outcomes
(`produced`, `timeout`, `permission_denied`, `empty_output`,
`execution_failed`) use disjoint stage/outcome pairs. Only the execution stage
means the producer was invoked. None of the failure pairs means that the target
thread, rank, or communicator was absent. Preflight permission hints are
provenance only; only execution may return `permission_denied`. Raw-output
digests are null whenever execution did not occur.

显式 opt-out 使用 `not_requested × disabled`。coordinator decision、preflight result
与 execution outcome 使用互斥 pair。只有 execution 表示 producer 已实际调用并可返回
`permission_denied`；preflight 权限信息只属于 provenance。未执行时 raw-output digest
必须为 null，任何失败 pair 都不能被解释成目标不存在。

### 4. Native interpretation is exact and version constrained

Rules are data with explicit platform, producer kind, PyTorch/vLLM/NCCL,
topology, and normalized ordered-frame constraints. They may never constrain
producer implementation name or implementation version; those stop at the
normalizer as provenance. An unmatched input emits `unknown`; there is no
nearest-match or confidence-based fallback.

规则可以约束 producer kind、platform、PyTorch/vLLM/NCCL、topology 与标准化有序
frame，但不能读取或约束 producer implementation name/version；后两者只属于
normalizer provenance。未匹配输入输出 `unknown`，不做 nearest-match。

### 5. Mature producers are evaluated before source probes

The order is PyStack, retained Flight Recorder evidence, and NCCL RAS where the
installed version supports it. A C++ probe is admitted only for a closed
lifecycle transition none of them can expose.

顺序是 PyStack、已有 Flight Recorder evidence，以及版本支持时的 NCCL RAS。只有它们
都无法暴露一个关闭生命周期转换时，才允许 C++ probe。

### 6. Only #196968 currently has a probe candidate

The candidate fact is the order among dump-responder stop, communicator
destruction start/completion, peer request observation, and dump completion.
#53859 and #196996 do not pass the probe admission gate.

Block 5 may decide whether one lab-local measurement patch is needed after
mature producers fail. An upstream-facing instrumentation proposal remains
blocked until #197232 has an explicit maintainer outcome.

当前唯一候选是 dump responder stop、communicator destruction start/complete、
peer request observation 与 dump completion 的顺序。#53859 与 #196996 不满足
probe gate。成熟工具确认不足后，可以决定是否需要 lab-local measurement patch；但在
#197232 获得明确 maintainer 结果前，不提出 upstream instrumentation。

### 7. Healthy/fault pairs are mandatory

A frame or stage seen in both healthy and fault windows cannot be the sole
discriminator. Block 5 must retain paired observations using the same producer
and applicable version family.

healthy 与 fault window 都出现的 frame/stage 不能单独作为判据。Block 5 必须用同一
producer 和适用版本族保留成对观察。

### 8. Capture flow control is part of correctness

One active capture per target, a fixed timeout/output budget, cooldown, and
per-incident quota prevent a repeated no-progress trigger from repeatedly
attaching to a wedged process.

每个 target 只允许一个 active capture，并设置固定 timeout/output budget、cooldown 和
per-incident quota，避免 no-progress 每轮判断都重复 attach 已卡住的进程。

## Claim-by-claim review questions / 逐项审查问题

### #196968

- Does process identity plus a bounded external stack justify participant
  presence without claiming collective participation? / 稳定进程 identity 与有界外部栈是否
  足以证明 participant 仍在，同时不推断 collective participation？
- Are the stage flags sufficient to claim producer lifecycle loss, or is an
  additional source-native transition required? / 当前 stage flags 是否足以证明 producer
  生命周期丢失，还是需要额外 source-native transition？
- Is the missing artifact set demonstrably closed? / 缺失 artifact 的预期集合是否真正封闭？
- Does any sentence accidentally present #197232 as accepted or merged? / 是否有任何表述
  把 #197232 提前写成已接受或已合入？

### #53859 / #53883

- Can the verdict be recomputed without reading the stack? It must be yes. /
  不读取 stack 是否仍能重算 verdict？答案必须是 yes。
- Is request/service scope explicit wherever progress is discussed? / 所有 progress
  表述是否明确 request 或 service scope？
- Are four dropped batches described only as the deterministic cell result, not
  a production rate? / 四个 dropped batch 是否只被描述为确定性 cell 结果，而非生产率？
- Is the case consistently described as independent validation? / 是否始终明确这是独立验证？

### #196996

- Does the single-GPU evidence prove the local mechanism without claiming the
  source of fp32 in the organic run? / 单卡证据是否只证明本地机制，而未声称已解释 organic
  run 中 fp32 的来源？
- Does the two-rank result distinguish exact local assertion from peer
  non-completion? / 双卡结果是否区分本地精确 assertion 与 peer non-completion？
- Is native capture correctly rejected as unnecessary for the current claim? /
  是否正确拒绝把 native capture 作为当前 claim 的必要条件？

## Requirements review / 需求说明审查

The requirements document defines fifteen requirements. The load-bearing ones
for Block 5 are:

需求说明包含十五项要求，其中 Block 5 的关键项是：

- LLR-001/002: stable subject binding and bounded capture / 稳定主体绑定与有界采集；
- LLR-003/004: mixed Python/native frames and explicit GIL `unknown` / 混合栈与显式
  GIL `unknown`；
- LLR-005: disjoint attempt-stage/outcome pairs / 互斥的 attempt-stage/outcome pair；
- LLR-006/007/008: closed, versioned, producer-independent interpretation / 关闭、
  带版本约束、与 producer 实现无关的解释；
- LLR-009/010: lifecycle facts separated from communicator authority / 生命周期事实与
  communicator 权威来源分离；
- LLR-011/012: verdict isolation and privacy / verdict 隔离与隐私边界；
- LLR-013/014: flow control and paired controls / 采集流控与成对控制。

Reject the requirements if they imply a general instrumentation framework or
if any item cannot be traced to a Block 4 distinction.

如果这些需求暗示建设通用 instrumentation framework，或者某项无法追溯到 Block 4 的
判定缺口，应拒绝该需求。

## Design review / 设计说明审查

The proposed design has four layers: bounded producer acquisition, common
normalization, version-constrained attribution, and the unchanged v0.2
verifier. Block 5 writes an experimental sidecar and displays attribution beside
the verdict; the two are never merged.

设计分为四层：有界 producer acquisition、通用 normalization、带版本约束的
attribution，以及保持不变的 v0.2 verifier。Block 5 写 experimental sidecar，并把
attribution 与 verdict 并列展示，两者不合并。

The most important falsification tests are:

最重要的反证测试是：

1. sequential real-producer captures are first bracketed by a same-tool
   stability control; after it passes, `py-spy` and PyStack must satisfy the
   same rule predicates and emit the same `blocked_in`, while frame sequences
   and coverage may differ / 顺序采集先通过同工具前后夹持的稳定性控制；通过后真实
   `py-spy` 与 PyStack 必须满足相同 rule predicate 并输出相同 `blocked_in`，但 frame
   sequence 与 coverage 可以不同；
2. version or frame mismatch yields `unknown` / 版本或 frame 不匹配输出 `unknown`；
3. removing native evidence cannot change a sufficient v0.2 verdict / 删除 native
   evidence 不能改变已充分成立的 v0.2 verdict；
4. invalid lifecycle order fails verification / 非法 lifecycle 顺序验证失败；
5. public output rejects raw frames, paths, stderr, arguments, and addresses /
   public output 拒绝 raw frame、路径、stderr、参数与地址；
6. repeated triggers respect occupancy and cooldown / 重复 trigger 遵守 occupancy 与
   cooldown。

## Deliberate non-deliverables / 本 Block 明确不交付

- no PyStack adapter implementation / 不实现 PyStack adapter；
- no C++ source probe / 不增加 C++ probe；
- no v0.2 schema or verdict change / 不修改 v0.2 schema 或 verdict；
- no native root-cause classifier / 不建设 native 根因分类器；
- no DCGM/NVML collector / 不增加 DCGM/NVML collector；
- no new issue or experiment area / 不新增 issue 或实验领域；
- no publication of the #196968 full case study / 不发布 #196968 完整 case study。

## Acceptance checklist / 验收清单

- [ ] All three cases have subject, interval, producer, contradiction, and
      insufficient-evidence paths / 三案均包含主体、窗口、producer、反证和证据不足路径；
- [ ] #53859/#53883 remains independent validation / 保持独立验证定位；
- [ ] #197232 remains an open proposed fix pending explicit outcome / 在明确结果前保持
      open proposed fix 表述；
- [ ] only #196968 retains a possible probe candidate / 只有 #196968 保留 probe 候选；
- [ ] native evidence cannot mutate the v0.2 verdict / native evidence 不能改变 v0.2
      verdict；
- [ ] `unknown` and every valid staged outcome are terminal valid results /
      `unknown` 与每个合法 staged outcome 都是合法终态；
- [ ] coordinator, preflight, and execution outcomes cannot be confused / coordinator、
      preflight 与 execution 结果不可混淆；
- [ ] rules constrain producer kind but never implementation identity/version /
      规则只可约束 producer kind，不得约束 implementation identity/version；
- [ ] post-capture identity failure invalidates native binding without asserting
      `process_missing` / capture 后 identity 失败只使 native binding 无效，不产生
      `process_missing`；
- [ ] every future capability points to a named Block 4 row / 未来每项能力均能指向
      Block 4 的具体行；
- [ ] review approval authorizes Block 5 capability checks only, not integration
      / review 通过只授权 Block 5 capability check，不授权产品集成。

## Review commands / 审查命令

```bash
python -m pip install -e ".[dev]"
git diff --check
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

Local verification on 2026-09-21 used Windows 11
`10.0.26200`, CPython `3.14.2`, and `jsonschema 4.26.0`; no `py-spy` executable
was installed. In that environment, 210 tests passed and 2 platform-specific
tests skipped. `compileall`, `git diff --check`, and relative-link resolution
for the Block 4 review set passed. A clean checkout without the development
dependencies may skip jsonschema-dependent tests, so its count must not be
compared with this run as if the environments were equivalent. CPython 3.14.2
is outside the release CI matrix, which currently covers 3.10, 3.12, and 3.13;
this is a local regression result, not an added support claim.

A second run used CPython 3.14.2 with `-S` and `PYTHONPATH=src;tests` to simulate
the absence of site packages, including `jsonschema`. It ran all 210 discovered
tests with 14 explicit skips and no import errors; the external-writer tests
continued to run rather than being skipped transitively.

2026-09-21 本地验证环境为 Windows 11 `10.0.26200`、CPython `3.14.2`、
`jsonschema 4.26.0`，未安装 `py-spy` executable。该环境中 210 个测试通过，2 个
平台相关测试跳过；`compileall`、`git diff --check` 和相对链接检查通过。未安装开发
依赖的 clean checkout 会跳过依赖 jsonschema 的测试，不能把其测试数量与本结果视为
同环境比较。CPython 3.14.2 不在 release CI 的 3.10/3.12/3.13 matrix 内，因此这里只是
本地回归结果，不新增支持声明。

另一次使用 CPython 3.14.2、`-S` 和 `PYTHONPATH=src;tests` 模拟无 site-packages
环境：仍发现并运行 210 个测试，其中 14 个显式 skip，没有 import error；
external-writer 测试继续执行，没有被传递性跳过。

This block is documentation-only. Runtime tests are regression checks, not
evidence that the new native design has been implemented.

本 Block 只有文档变化。运行测试只用于回归检查，不代表 native design 已经实现。
