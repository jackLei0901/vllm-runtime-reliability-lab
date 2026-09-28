# Lab 技术说明：运行时边界上的证据规则

状态：2026-09-28 本地草案，未批准、未发布；本中文版为准，[English](LAB_TECHNICAL_SPEC.md) 为对应译文。本文回应[Lab 需求说明](LAB_REQUIREMENTS.zh-CN.md)，不改变 v0.2 verdict、历史评分或 Q4 预注册规则。

## 1. 本文回答的问题

目前的 Lab 技术能力，能否支撑其目标：让非作者面对真实 vLLM/PyTorch 运行时故障时，借助可核验的证据规则，比只看常规日志或普通复现更可靠地判断观察证明了什么、没有证明什么，以及下一步该查、测或修什么？

**回答：**现有组件能支持有界的单目标进展案例或针对具体问题的验证，不能覆盖所有运行时边界。第一项已命名的外部候选才能检验其覆盖是否足够。交付前必须固定与普通复现的对照，并核对所引用源码的当前版本。只有终止传播候选确实需要时，才考虑可执行的终止检查；本文不授权新框架层、分类器、collector 或该检查的实现。具体差距见 §5。

## 2. 自顶向下：故障信号在哪里改变含义

运行于 PyTorch 上的 vLLM 服务跨多个进程。每层为了自己的用途产生信号，其他层或人再消费它们。以下是审计入口，不是穷尽的拓扑：

| 层 | 所有者与主要信号生产者 | 信号原本服务的目的 |
| --- | --- | --- |
| L5 前端 | API server、`AsyncLLM` 输出处理、`/health`、指标、launcher watchdog | 提供请求服务；报告 engine client 是否处于错误状态 |
| L4 EngineCore | busy loop、scheduler、KV-cache manager、`ENGINE_CORE_DEAD` | 执行模型步骤；报告自身死亡 |
| L3 Executor | `MultiprocExecutor` worker monitor、RPC deadline、共享内存环 | 派发步骤；发现 worker 退出或 RPC 超时 |
| L2 Worker | model runner、CUDA graphs、custom ops、`torch.compile`、profile/warm-up | 对真实和合成输入执行 kernel |
| L1 PyTorch runtime | c10d `ProcessGroupNCCL` watchdog/heartbeat、Flight Recorder、allocator | 发现 collective 超时；转储历史 |
| L0 CUDA/NCCL/driver | CUDA sticky error、NCCL async error、NCCL RAS | 报告设备和 communicator 故障 |

L3–L5 的源码事实见[活性契约盘点](ENGINE_LIVENESS_CONTRACT_INVENTORY_2026-09-27.md)：S1–S11、T1–T14，vLLM 固定版本 `c8602c79`。L1 见[c10d 生命周期走读](reviews/C10D_LIFECYCLE_DEEP_DIVE_2026-09-23.md)：提议修复源码固定在 `a339711`。这两个 pin 是不同项目、不同用途；具体交付还需按目标版本重核。

以下是**选取的**生命周期和诊断案例：消费者把跨边界信号解释成了生产者并未保证的事实；不声称所有 Lab 发现都属于这种类型。

| 容易作出的推断 | 信号实际能证明的内容 | Lab 证据 |
| --- | --- | --- |
| `/health` 2xx 表示请求仍有进展 | Engine client 尚未标记错误；没有给出进展事实 | [#53859 R3 公开结果](../results/vllm-zmq-backpressure-stage1-r3-20260916/)；本地隔离环境离线重放 |
| 顶层退出码 0 表示主动、正常停止 | #52178 修复前，EngineCore 被 SIGKILL 后也可能退出 0 | [fault-recovery](../experiments/fault-recovery/README.md) base/fix 表 |
| 一个 shutdown timeout 覆盖整个清理过程 | 外层宽限可能短于嵌套的内层时序 | [K1/K2](reviews/ENGINE_LIVENESS_K1_K2_CPU_RUN_2026-09-27.md)；真实 vLLM 函数、假进程树，只覆盖测过的有限数值 |
| Flight Recorder dump 缺失表示 rank 不存在 | communicator 销毁时 dump responder 可能无法响应 | [c10d 验证](../experiments/pytorch-c10d-shutdown-dump/UPSTREAM_FIX_VALIDATION_2026-09-16.md)，PyTorch #197232 |
| 包版本相同意味着运行时身份相同 | 构建与进程身份仍须分别绑定 | [运行时身份边界](MATCHING_VERSIONS_ARE_NOT_RUNTIME_IDENTITY.md) |

Q4 关键词限定样本的 AI 首轮中，40 条合格报告有 30 条标为 `outside_model`，另有 4 条 `insufficient_information`。按 v2 指南，前 30 条的 `fault_domain` 均**被标为** `leaf`，但这些 AI 标签尚未经人工逐条审核；它们不能证明故障事实上都属于 leaf 层，或错误信息已自行说明原因。人工一致性未评分，也不能从该样本估计生产故障比例。以下设计选择来自 Lab 的有界目标，而不是这次抽样：检查跨边界信号允许消费者作出什么推断，不重复成熟工具的底层错误采集。

## 3. 可靠性层级：方向而非承诺

| 层级 | 内容 | Lab 范围 |
| --- | --- | --- |
| 0 | 原始日志、指标、退出码、dump | 现有工具提供 |
| 1 | 绑定身份、时间和新鲜度的观察；有条件地另行固定构建身份 | 已支持特定目标，不支持自动发现 |
| 2 | 单一契约、负对照、关闭的结果集合与明确边界 | Lab 当前的主要工作层 |
| 3 | 合并多项事实的诊断规则 | 需要在受控案例中测量误报与漏报；目前不作此声明 |
| 4 | 基于诊断自动行动 | 范围外 |

上层只能消费下层确有依据的事实并保留其限制。任何自动告警或诊断消费者都先需要第 3 层的错误率证据；当前 Lab 产物没有提供这项证据。

## 4. 已有工具与证据做法

下表列的是被不同检查复用的工具和做法，不是一条已经实现的统一流水线，也不是新增层的设计。

| 编号 | 职责 | 当前形态 |
| --- | --- | --- |
| C1 信号语义源码图 | 记录生产者、消费者、实际含义、禁止推断及源码 pin | 活性盘点与 c10d 走读文档；不是软件 registry |
| C2 采集与身份 | 有界观察；提供 PID 时记录进程启动身份，构建 pin 另按案例固定 | `collectors`、`recorder`、`collect_bundle`、`native_producers`、`stacks`；endpoint 与 PID 的关系仍由操作者声明 |
| C3 封装与脱敏 | 关闭形状、摘要核验、排除私有材料 | `bundle`、`collect_bundle`、`verify_bundle`、`schema`、`external_schema` |
| C4 分离的契约检查 | 各自检查一条契约与其对照 | 进展：`progress.derive_verdict`；退出传播：`experiments/fault-recovery`；生命周期预算：K1/K2 |
| C5 离线检查 | 不重跑 GPU 实验也可重算支持的结论 | `verify_bundle` 用于 v0.2 collect bundle；`replay` 固定于 #53859 R3。干净 venv 试验在直接 `pip install .` 遭包代理失败后，使用本地构建 wheel。 |
| C6 交付清单 | 说明非作者在线程中会收到什么 | 已有 review entry 可作例子；§5 的 G1 是逐候选清单，不是新组件 |

各检查**按适用范围**复用身份、新鲜度、对照、产物固定、脱敏和 replay 做法；它们并不共享同一套实现或决策逻辑，也没有合并 verdict 的组件。

## 5. 对需求的适配与缺口

| 需求 | 现状 | 收口条件 |
| --- | --- | --- |
| 非作者无需追问就能检查产物 | 部分成立：#53859 能离线 replay；K1/K2 需要 Linux 和固定的 vLLM 函数；终止案例的原始证据仍是私有材料 | **G1 逐候选交付清单**；清单本身不证明独立采用。交付前完成。 |
| 信号事实适用于目标版本（R3、R6） | 盘点固定在 `c8602c79`，upstream 已变化 | **G2 使用时重新 pin**：在目标线程所用提交核对被引用的行；否则标 `unverified`。这是步骤，不是新工具。 |
| 关闭结果并控制装置失效（R4） | 进展可得 `undetermined`；K1/K2 有各自装置结果；终止案例目前是 base/fix 退出码表，不是共用 verifier | **G3 有条件的终止检查**：只有明确的外部终止案例无法由现有表与测试服务时才考虑。先写失败测试，再冻结互斥的 `propagated`、`not_propagated`、`unscored`，其中 `apparatus_failed` 是后者的原因，并保留 SIGTERM 对照。 |
| 每检查一条契约（R7） | C4 保持分离 | 无新增组件 |
| PyTorch 边界 | c10d 有一个案例，无通用 PyTorch 采集 | 目前无缺口建设；新边界只通过有时间上限的走读进入 |

**G1 交付清单。**在决定性运行或公开草稿前，冻结第 1–3 项以及会否定增量主张的结果。交付物最终说明：

1. 线程及其要做的决定。
2. 普通复现原本能证明什么——需求说明要求的事前对照。
3. 主张、证据等级和所检查的单一契约。
4. 构建与运行时身份；重新核对的信号源码 pin。
5. 非作者可运行的命令，或可直接检查的精确文件。
6. 正反对照、关闭的结果集合，以及 `unknown`/`unscored`。
7. 证据边界、可能反例，以及事前声明的增量对照结果。
8. 目标项目当前贡献/AI 披露规则、隐私与链接检查、发出后的线上回读。这些是人工步骤，不是声称 `verify_bundle` 能检查发布安全。

## 6. 验收与非目标

**技术适配检查：**在建设前，将最迟 2026-10-12 命名的首个候选依次对照 C1–C5 和 G1/G2。记录现有工具能否提供事前声明的增量事实，以及可运行或可检查的产物。若不能，记录 `unsupported` 和准确缺少的事实；不默认增设组件。G3 只能在满足条件时使用。这项检查即使通过，也只说明技术架构能交付**该案例**，不证明外部有人采用，更不证明整个 Lab 目标完成。

**非目标：**合并不同检查的框架层；新分类器或 taxonomy 扩张；自动扩充 `src/dfxlab`；第 3–4 层声明；用功能、文档或检查数量验收。个人影响力可能来自长期有用贡献，但不是技术验收指标。

**停损：**遵循[需求说明](LAB_REQUIREMENTS.zh-CN.md)。若向活跃线程反复交付而未观察到有用采用，应区分对方尚未来得及审阅与明确认为产物无用，再决定是否进入维护模式；不能以更大的架构回应。
