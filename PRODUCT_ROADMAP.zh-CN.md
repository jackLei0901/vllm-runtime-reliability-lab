# Lab 发展计划

> 2026-09-21 修订：从“扩大采集和部署能力”转向“故障分类、领域解释和可证伪判断”。
> 2026-09-23 目标更新：成为推理运行时可靠性领域可信赖的贡献者；
> `v0.2.0` 已发布，以下历史发布门槛保留为审计记录，不再是待办项。

长期目标的可观察证据是：Lab 的可复核结论影响 upstream 决策；别人能复算、
质疑或沿用证据规则；作者在同一 vLLM runtime 领域持续合入修复、审阅他人 PR
并维护质量。Stars、功能数量及 Collaborator/committer 头衔都不是选题依据。
后者若发生，是持续贡献的结果，不是 Lab 单独授予的身份。

## 1. 定位

vLLM Runtime Reliability Lab 不是监控平台、通用 collector framework 或 issue
收集仓库。它是面向推理运行时故障的证据实验室：将不完整、缺失或相互冲突的
运行时观察，转化为可复现、可证伪并且能明确拒绝回答的结论。

Lab 的独立价值按重要性排序：

1. 故障分类：定义需要区分的 failure modes；
2. 领域解释：说明某个 frame、counter 或 stage 能支持和不能支持什么推断；
3. verdict contract：定义身份、时间、freshness、冲突和 precedence；
4. evidence standard：冻结 schema、负对照、隐私边界和 fail-closed verifier；
5. collector：只提供上述判断所需的事实，且尽量复用成熟实现。

外部工具负责回答“观察到了什么”，Lab 负责回答“这些观察足以证明什么，以及何时
证据不足”。

## 2. 不可违反的工程规则

- 每个公开 bundle 字段必须声明为 `decisional` 或进入关闭的 `non_decisional` 集合。
  `decisional` 字段由 verdict 或 attribution rule 消费；`non_decisional` 只允许 schema
  version、完整性摘要、producer provenance 及复现所需但不参与判断的元数据。字段的角色
  按具体 claim 声明，不能把所有 timestamp 或 identity 一概视为非决策字段。
- verifier 必须通过变异测试证明：在 bundle 仍然通过 schema 和完整性校验的前提下，改变
  `non_decisional` provenance 不会改变 semantic verdict。单独篡改摘要只能导致 validation
  failure，不能产生另一个 verdict。无法归入上述两类的字段不发布。
- 每个正向 verdict 都必须有反证或不足证据测试。
- `producer_missing != member_missing`，工具失败、权限不足和目标状态不得混为一谈。
- verdict 与 attribution 分离；stack、NCCL RAS 和硬件状态只能补充解释，不能绕过
  progress、process identity 和 demand 的既有判定条件。
- 原始 native stack、地址、参数和本地路径默认私有；公开结果只保留关闭枚举、producer
  identity 和原始内容摘要。
- 成熟外部能力优先通过有界 CLI、socket 或 metrics 接口调用，不重写 stack unwinder、
  NCCL rank discovery 或 GPU telemetry。
- verifier 只能读取标准化 observation shape 和 typed producer outcome，不能 import、
  分支判断或匹配具体工具名称。两个 producer 产生相同标准化事实时必须得到相同 verdict；
  事实更少时只能得到更弱的 attribution 或 `insufficient_evidence`。
- 只有当两个已经命名的 failure modes 因缺少内部状态而无法区分时，才增加最小 C++
  probe。
- 新 producer 必须让至少一个现有 contract case 从 `insufficient_evidence` 变为可验证或
  可反证，否则不进入产品代码。
- 不为制造内容继续扩大问题数量；优先完成和解释已有案例。
- degraded operation 是正常路径：`unsupported`、`permission_denied`、`timed_out`、
  `feature_disabled` 和 `producer_missing` 必须保留原义，不能静默补默认值。

## 3. Fault taxonomy admission gate

新的 failure category 只有同时满足以下条件才能进入 taxonomy：

- 有一个已命名、可观察的 discriminating evidence requirement；
- 至少有一次可重放 reproduction；
- 至少有一个 negative control 或明确反例；
- 声明适用的软件版本、拓扑和已知不支持边界；
- 无法满足条件时保留 `unknown`，不得选择“最接近”的已有类别。

`unknown` 是合法的终态，不是等待随意补分类的临时错误。

## 4. 当前基线：维护已发布的 v0.2，而不是扩展产品面

已发布 `v0.2.0` 的范围冻结为：

- CPU-only published-result replay；
- bounded `collect` 与 offline `verify`；
- server/client progress producers 及显式冲突规则；
- process identity、demand、health 和 progress precedence；
- closed-shape bundle、内容摘要和 fail-closed verdict；
- #53859/#53883 的归档 base/fix 重放。

`v0.2.0` 发布本身已完成；旧 release plan 中未回填的 checklist 不自动视为通过，
验证细节仍以实际 release 记录为准。后续仅修复可复现的缺陷，不以新功能维持发布
节奏。PyStack、NCCL RAS、DCGM、通用 trigger bus、部署模板及新的 collector
均不进入 v0.2。

#196968 case study 继续遵守 #197232 的明确 upstream outcome gate；等待期间不扩张该
case 的公开结论。

## 5. 阶段 E1：现有案例的 capability-gap audit

不编写新 adapter，先用已有案例回答“成熟工具能看到什么，Lab 还必须判断什么”。

| 案例 | 现成观察 | 尚需 Lab 证明的关系 |
| --- | --- | --- |
| PyTorch #196968/#197232 | Flight Recorder dump、外部 native stack | participant 是否仍存在、producer 为何消失、absence 能否解释 |
| vLLM #53859/#53883 | health、token counter、EngineCore stack、drop counter | demand 存在时是否停止服务、修复是否恢复活性并引入数据丢失 |
| PyTorch #196996 | distributed hang symptom、本地 assertion | distributed symptom 是否可缩减为单卡 correctness mechanism |

每个案例产出一份固定结构的 evidence-to-claim 表：

```text
observation
  -> allowed inference
  -> forbidden inference
  -> required corroboration
  -> contradiction
  -> verdict or insufficient_evidence
```

Block 4 已将三案固定为
[`docs/EVIDENCE_TO_CLAIM_BLOCK4.md`](docs/EVIDENCE_TO_CLAIM_BLOCK4.md)，并从剩余
判定缺口反推
[`docs/LOW_LEVEL_CAPABILITY_REQUIREMENTS.md`](docs/LOW_LEVEL_CAPABILITY_REQUIREMENTS.md)
和 [`docs/NATIVE_EVIDENCE_DESIGN.md`](docs/NATIVE_EVIDENCE_DESIGN.md)。该结果只完成
claim 与 capability contract；PyStack/NCCL RAS capability check 和任何最小 probe
仍属于后续阶段。

现有案例的有限 fault taxonomy v0 见
[`docs/FAULT_TAXONOMY_V0_2026-09-24.md`](docs/FAULT_TAXONOMY_V0_2026-09-24.md)；
#53859 的同案 DFX 消融见
[`docs/reviews/VLLM_53859_DFX_BASELINE_ABLATION_2026-09-24.md`](docs/reviews/VLLM_53859_DFX_BASELINE_ABLATION_2026-09-24.md)。
这两份是可反驳的分析基线，不是 issue tracker 的完整 corpus：R3 未保留指标
时间序列，因此 Prometheus-only 对照仍为 `not scored`。
允许另行预注册一个固定数量的已关闭 vLLM 可靠性 issue 随机样本，给现有类别
测覆盖率和 `insufficient_information` 比例；这不等于从样本中不断增设类别，
也不能仅凭 issue 描述推断某种 DFX 工具在真实部署中未发现故障。具体抽样框、
随机种子、排除规则和证据限制见上述 taxonomy v0，执行前必须冻结。

### E1 通过条件

- 三个案例均列出主体、时间窗口、producer 和反证；
- schema 中每个公开字段均声明为 decisional 或关闭集合内的 non-decisional；
- schema-valid 的 non-decisional provenance 变异不能改变 semantic verdict；摘要篡改只能
  导致 validation failure；
- 至少一组等价 producer vectors 证明 verifier 不依赖工具身份；
- 删除所有只改善展示、不改变 discrimination 的候选字段；
- 明确哪些结论由 Lab 发现、独立验证或仅提供系统级证据；
- 不把 #49869 描述为 Lab 发现。

## 6. 阶段 E2：native-state interpretation

只在已有 reproducer 上评估成熟 producer：

1. PyStack：验证 Python/native 混合栈、GIL 状态、权限失败和超时边界；
2. NCCL RAS：在支持的 NCCL 版本上比较 healthy/fault 两个窗口中的 rank、communicator
   和 collective 状态；
3. DCGM/Prometheus：仅当实验环境已经部署时，用作硬件时间线上下文，不建设新的默认
   NVML polling 产品面。

这些工具首先是实验对照，不自动成为 Lab 依赖。评估结果必须形成带显式版本约束、以
数据表示的关闭 native-state 解释表，而不是散落在 classifier 代码中的工具特判。例如：

| 观察 | 允许推断 | 禁止单独推断 |
| --- | --- | --- |
| thread parked in communicator destruction | 该线程当前处于 teardown | rank 未参与 collective |
| collective count 跨窗口不变 | 该 communicator 未观察到新 collective | 请求没有生成 token |
| 高 SM utilization | GPU 正在执行 kernel | kernel 正在产生有效进展 |
| publisher blocked in queue operation | 发布路径正在等待队列 | 整个服务没有进展 |

### 最小 C++ probe gate

只有同时满足以下条件才增加 probe：

- 需要区分的两个 failure modes 已在 taxonomy 中命名；
- PyStack、NCCL RAS、Flight Recorder 和已有公开 counters 无法区分它们；
- 所需事实是明确的状态转换，而不是自由文本日志；
- probe 可以输出关闭枚举或 stage flag；
- base/fix 或正/负对照能够验证该字段的解释。

优先候选仍是 #196968 已证明必要的 shutdown-stage 边界，而不是通用 csrc 埋点系统。

### E2 通过条件

- healthy 与 fault 观察成对保存；
- 工具缺失、unsupported、权限不足、timeout 和 empty output 明确区分；
- native classification 不直接改变 v0.2 主 verdict；
- 分类规则声明 vLLM、PyTorch、NCCL 和 producer 的适用版本；未匹配版本或 frame shape
  一律输出 `unknown`，禁止 nearest-match；
- 至少一个分类通过已有案例证明能排除一个竞争解释；
- 若现成工具已经足够，则明确记录“不需要 probe”的负面设计结论。

### 并行主线 L：在状态所有者处建设 C++ DFX 能力

底层能力不等 taxonomy、推广或 upstream review 结束后才开始。每周为同一条
已知 runtime seam 保留两个独立的源码/实验工作时段；以证据合同选问题，以真实
状态所有者选代码层。多进程身份绑定、manifest 和 join 首先是证据关系问题，
不能为了练 C++ 而将其搬进 vLLM csrc。已有 Flight Recorder、NCCL RAS 或外部
工具能够给出所需事实时，也不另造 probe。

**L0 — 区分已有的两种 C++ 能力。** #196968/#197232 的
`ProcessGroupNCCL` 源码状态表、可达性论证和带变异负对照的 CPU litmus 是
*阅读与推理*能力；shutdown dump responder 的同步协议、锁顺序、失败边界、
gtest 和真实回归是*实现与验证*能力。将两类证据整理为同一 seam 的审阅包，
分别写明源码可达、模型断言和真实运行各自证明到哪一步。PR 的合入另记为
upstream 结果，不阻塞能力训练，也不由现有测试推断补丁已被接受。

**L1 — 审查一个 c10d 可观测性缺口，不预设补丁。** 候选包括 communicator
failure reason 的结构化/持久化，以及销毁中状态是否真的不可由现有输出表达。
这些目前只是审查假设，不是已验证的缺陷或授权的新 PR。先固定 PyTorch 源
版本、查重现有机制及提案、列出两个会被混淆的解释，再用 Flight Recorder、
NCCL RAS 与外部栈对照。只有存在一个可命名且可验证的状态关系，才设计最小
关闭字段或回归断言。#197232 在同一 seam 尚未获得实质 reviewer 反馈前，不
并行提交第二个 c10d 可观测性 PR；等待期间可完成只读审查与本地测试。
第一项候选的源码审查见
[`docs/reviews/C10D_COMM_FAILURE_REASON_SOURCE_AUDIT_2026-09-24.md`](docs/reviews/C10D_COMM_FAILURE_REASON_SOURCE_AUDIT_2026-09-24.md)：
blocking-wait 测试路径把 timeout 保留在 Work exception/log，却没有给后续
communicator abort 传 reason；只在存储旁增加 enum 无法解决该路径。
第二项候选的源码审查见
[`docs/reviews/C10D_COMM_LIFECYCLE_STATE_SOURCE_AUDIT_2026-09-24.md`](docs/reviews/C10D_COMM_LIFECYCLE_STATE_SOURCE_AUDIT_2026-09-24.md)：
正常 destroy 与失败 abort 在 `isAborted()` 上合并，但销毁中调用该 getter
会等待 mutex；不能把内存中尚未更新的 flag 写成可并发观察到的“健康”。两项
审查都尚未证明新 probe 的必要性。

**L2 — 给产出设可证伪门槛。** 至少一条源码状态/顺序不变量、一个正/负对照、
一个能使断言失败的变异，以及现有 DFX 是否已足够的结论。若事实已可获得，
提交“无需 probe”的负面设计结果；若事实仍不可获得，先提出可审阅的最小
变更和 base/fix 方案，再决定是否 upstream。源码阅读不能升级为运行时发生
过的断言，CPU litmus 不能冒充真实跨 rank 验证。

**独立主线 K — vLLM CUDA/kernel 能力，不记作 Lab DFX 成果。** #55537 的
dispatch 层约束工作可以作为已有代码入口，但它本身不是 kernel 实现。真正的
kernel 能力要由一个版本固定的 CUDA/C++ 算子合同、正确性边界、负对照及性能
或资源取舍证明；等当前 upstream 队列收敛后只选一个相邻任务。该主线服务
职业目标，不能为了称作 Lab 的 C++ 能力而强行增加 vLLM DFX 埋点。

每个 L 阶段的审阅包固定回答：源提交与适用版本、状态所有者、两个竞争解释、
关键状态机、不变量及反例、测试/变异结果、现有 DFX 能否回答、仍未知的事实。

## 7. 阶段 E3：加强可证伪 verifier

不先建设通用规则引擎。每次只为一个已验证案例加入一条显式 claim，并要求：

- subject identity 稳定；
- 时间关系可复算；
- required evidence 和 corroborating evidence 分离；
- contradiction 优先于归因；
- 缺失关键 producer 时输出 `insufficient_evidence`；
- required producer 缺失时主 claim 变弱或拒绝；optional corroborating producer 缺失时
  不改写已经充分成立的主 verdict，但 attribution coverage 必须显式降低；
- 相同标准化 observation 来自不同 producer 时 verdict 相同；
- 同一 bundle 可离线复算并通过摘要验证；
- 修改关键字节或身份后验证失败。

首个候选 claim 是：

```text
participant present + diagnostic producer stopped + artifact absent
  -> producer_missing
  != participant missing
```

该 claim 只有在 #197232 gate 关闭、公开证据足够且隐私边界明确后才进入已发布
case study。

## 8. 阶段 E4：发布和 upstream 影响

按完整证据回路发布，而不是按功能数量发布：

```text
ambiguous symptom
  -> preregistered alternatives
  -> bounded evidence
  -> minimal mechanism
  -> base/fix comparison
  -> falsifiable verdict
  -> upstream outcome
```

发布顺序：

1. 已发布 v0.2 和 #53859 CPU-only replay；
2. 已写出“Failures that never reach the supervisor”综合文章；后续传播仍以
   相关问题中的直接证据为前提；
3. 只在能提供直接证据的现有 upstream thread 中链接不可变 artifact；
4. #197232 获得明确结果后发布 Flight Recorder case study；
5. 若方法形成稳定共识，再考虑将 progress-vs-health 诊断步骤贡献到 upstream 文档。

## 9. 明确延后或不做

- 通用 plugin/adapter framework；
- 长期 telemetry 数据库、dashboard 和查询系统；
- 默认 DCGM/NVML collector；
- 自动根因分类、自动重启和自动修复；
- systemd、Kubernetes sidecar 和大规模部署模板；
- 为展示覆盖面而新增实验目录或 upstream 问题；
- 在没有 verifier consumer 的情况下增加指标；
- 将 stars、安装量或 collector 数量作为主要技术指标。

如果真实 upstream 场景要求其中某项，再以具体 claim 和 acceptance test 重新开启，
而不是提前建设平台。

## 10. 成功指标

### 技术与 upstream

- 至少一个由 Lab 发现的问题获得 upstream 明确结论，优先争取修复合入；
- 至少两个 upstream issue/PR 引用不可变 Lab 证据；
- 至少一个第三方修复通过 Lab 的 base/fix 或 claim verifier；
- 至少一个 native-state mapping 经实际案例验证，而非只停留在设计文档。
- 至少一个状态所有者明确的 C++ DFX 边界形成版本固定的状态机、可证伪不变量
  和正/负对照；即使结论是现成工具足够，也保留可审阅的负面结果。vLLM
  kernel 能力单独评价，不充作 Lab 技术指标。
- 对他人的相邻 vLLM runtime PR 至少提供一次可核查的实质审阅；被回应的边界、
  反例或测试建议，比评论数量更重要。
- 长期观察 maintainer 是否主动在作者未发起的相关问题中征询证据或判断；
  不通过索取头衔或制造新问题来追求该信号。

### 方法复用

- CPU-only replay 可在五分钟内运行；
- 外部使用者能够复算一个已发布 verdict；
- 至少一个外部 issue、PR 或讨论采用 evidence format、failure taxonomy 或判断规则；
- 一个第三方能够指出某条 claim 的反例、缺失条件或 schema 改进。

### 传播

- 首批十个真实外部 stars、一个 fork；
- 一篇 case study 获得社区成员引用或转发；
- maintainer 在新的相关问题中主动引用、邀请或 cc 作者。

Stars 只表示传播；可证伪结论、upstream 结果和方法复用才表示技术价值。

## 11. 近期执行顺序

具体日期和本季度的唯一主目标见 [2026 Q4 执行计划](docs/PLAN_2026Q4.zh-CN.md)；
持续发现的登记与抽样边界见 [Q4 痛点发现协议](docs/PAIN_POINT_DISCOVERY_2026Q4.zh-CN.md)。
本节仍作为工程规则和优先级，不是第二套日程。

1. 维护已发布 v0.2 的 replay、verifier 和证据边界；外部反馈优先修复真实复用障碍。
2. 以现有案例形成 fault taxonomy v0 和同案 DFX baseline 对照，先验证 Lab 在哪里
   真正改变诊断结论；允许固定样本测覆盖分母，不以批量挖 issue 扩大类别。
3. 同步执行底层主线 L：整理 #197232 的源码级状态机与回归证据，评审上述
   两项 c10d 只读审查的反例及现成 DFX 覆盖；不等待 review 才开始训练，也不在
   同一 seam 并行开第二个未审阅 PR。vLLM CUDA/kernel 主线 K 独立排期。
4. 将 V1 的 recorder 无进展触发缺口作为独立设计 gate：先证明 demand、freshness、
   身份和冷却规则可与 frozen verdict 一致，再决定是否在未来版本实现；不修改 v0.2。
5. 在现有 vLLM runtime/lifecycle 主线中完成已打开的 upstream 工作，记录合入、
   明确拒绝或设计结论；不把未审阅 PR 记作 upstream 成果。
6. 选择与已读代码相邻的他人 PR 做实质审阅：先复现或核对源码，再提出可证伪的
   边界或测试；不为增加审阅数量写泛泛评论。
7. 只在已有案例的判断缺口确实需要时完成 PyStack/NCCL RAS capability 对照，
   并用版本约束、负对照和 `unknown` 限制 native attribution。
8. #197232 得到明确 upstream 结果后发布 case study；此前只引用已公开且有边界的证据。

这一路线不以“拥有更多 collector”为进展。每一阶段都必须让一个已命名 failure mode
更可区分、一个错误推断更难发生，或一个结论更容易被第三方反证。
