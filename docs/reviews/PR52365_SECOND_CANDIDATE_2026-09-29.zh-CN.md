# 第二个外部候选：vLLM #52365

状态：仅完成源码预检，2026-09-29。本文为准。已核对 PR head `d996d76ec68e9f6a348b7b085ae1da61cb6095be`；交付前须重新核对。本记录不授权实验、开卡或上游评论。

## 消费者与决策

消费者是 [#52365](https://github.com/vllm-project/vllm/pull/52365) 的作者和 reviewer；该 PR 针对另一位用户报告的[生产环境 GPU kernel 卡死](https://github.com/vllm-project/vllm/issues/52247)。未决问题是：超时应覆盖异步 GPU 输出事件等待、整个 EngineCore iteration，还是由进程级 watchdog 负责。[维护者反对轮询](https://github.com/vllm-project/vllm/pull/52365#issuecomment-5303805293)；[作者询问 watchdog 应覆盖哪一层](https://github.com/vllm-project/vllm/pull/52365#issuecomment-5305822295)。PR 仍开放，但自述尚未准备合入。截至本次核查，讨论中没有找到维护者对该范围问题的答复。

## 增量比较：在后续工作前固定

| 问题 | 预检记录 |
| --- | --- |
| 普通复现 | 原 issue 已证明报告环境中 `copy_event.synchronize()` 无限等待且 `/health` 仍为绿色。PR 已用假事件测试轮询 helper。重复这两项不算 Lab 交付。 |
| Lab 增量 | 用固定源码版本梳理**谁在何处等待、哪个截止时间从何时起算**，对照单进程事件等待、多进程 RPC 等待，以及 [#55700](https://github.com/vllm-project/vllm/pull/55700) 仅用于诊断的 watchdog。能改变决策的问题是：过去无人读取的 60 秒设置，是否应在每次输出事件等待时成为**默认开启的致命上界**；缺失的负控是事件在 60 秒后才完成、但属于合法慢步骤的情况。单纯代码路径列表不算 Lab 交付。 |
| 初步源码事实 | [`AsyncOutputFuture.result(timeout)` 不支持超时](https://github.com/vllm-project/vllm/blob/d996d76ec68e9f6a348b7b085ae1da61cb6095be/vllm/v1/executor/uniproc_executor.py#L32-L44)。拟议的[事件等待在进入该等待时开始默认 60 秒计时](https://github.com/vllm-project/vllm/blob/d996d76ec68e9f6a348b7b085ae1da61cb6095be/vllm/v1/worker/gpu/event_utils.py#L20-L38)。多进程 [`execute_model` 和 `sample_tokens` 使用 RPC 预算](https://github.com/vllm-project/vllm/blob/d996d76ec68e9f6a348b7b085ae1da61cb6095be/vllm/v1/executor/multiproc_executor.py#L340-L363)，[默认 300 秒](https://github.com/vllm-project/vllm/blob/d996d76ec68e9f6a348b7b085ae1da61cb6095be/vllm/envs.py#L1794-L1795)，起点是 RPC 发送，而不是进入事件等待。因此在多进程场景中，拟议等待有可能先于 RPC 截止时间报错，但实际先后取决于 worker 何时进入等待。这只是源码推断，并非实测。 |
| 推翻该区分的结果 | 源码图发现此版本的单进程事件等待已经有上界、多进程输出路径不受 RPC 预算约束，或者现有 watchdog 已执行同样的终止动作。出现任一情况，交付前须撤回或修正上述区分。 |

## 有界下一步

最多用两小时做只读源码图和现有测试的反例控制审查。覆盖 V1/V2 的生成与 pooling、`UniProcExecutor` 和 `MultiprocExecutor` 调用路径、异常传播，以及 #55700 超时后实际采取的动作。未核实的状态转移应明确标为未知，不能推断超时必然重启服务。本预检不触发新探针或 GPU 预约。若结果只是普通源码 review 也能得到的表格，则归为 Q5 review 材料，**不计为 Lab 增量价值**。上游发帖前须刷新 PR head 和讨论、遵守项目贡献规则，并由用户审核措辞。

## 源码图检查点（非运行评分）

补丁覆盖四处 `get_output()`：[V1 生成](https://github.com/vllm-project/vllm/blob/d996d76ec68e9f6a348b7b085ae1da61cb6095be/vllm/v1/worker/gpu_model_runner.py#L331-L338)、[V1 pooling](https://github.com/vllm-project/vllm/blob/d996d76ec68e9f6a348b7b085ae1da61cb6095be/vllm/v1/worker/gpu_model_runner.py#L451-L460)、[V2 生成](https://github.com/vllm-project/vllm/blob/d996d76ec68e9f6a348b7b085ae1da61cb6095be/vllm/v1/worker/gpu/async_utils.py#L169-L174)和[V2 pooling](https://github.com/vllm-project/vllm/blob/d996d76ec68e9f6a348b7b085ae1da61cb6095be/vllm/v1/worker/gpu/async_utils.py#L248-L255)。`UniProcExecutor` 通过 [`AsyncOutputFuture.result()`](https://github.com/vllm-project/vllm/blob/d996d76ec68e9f6a348b7b085ae1da61cb6095be/vllm/v1/executor/uniproc_executor.py#L32-L46) 调用 `get_output()`；异常到达 [EngineCore 的结果等待](https://github.com/vllm-project/vllm/blob/d996d76ec68e9f6a348b7b085ae1da61cb6095be/vllm/v1/engine/core.py#L642-L650)。`MultiprocExecutor` 的 worker [把 `get_output()` 异常转换为 FAILURE 回复](https://github.com/vllm-project/vllm/blob/d996d76ec68e9f6a348b7b085ae1da61cb6095be/vllm/v1/executor/multiproc_executor.py#L985-L997)，接收方[再抛出 `RuntimeError`](https://github.com/vllm-project/vllm/blob/d996d76ec68e9f6a348b7b085ae1da61cb6095be/vllm/v1/executor/multiproc_executor.py#L426-L441)。EngineCore 的外层处理器[把未捕获异常作为致命错误](https://github.com/vllm-project/vllm/blob/d996d76ec68e9f6a348b7b085ae1da61cb6095be/vllm/v1/engine/core.py#L1431-L1438)。以上只是代码路径，未在服务中实测；前端状态和重启策略仍未评分。

设计差别已经明确：#52365 拟设置**终止性的事件局部**截止时间；#55700 的[自述目标](https://github.com/vllm-project/vllm/pull/55700)是漏喂心跳时作**诊断性栈转储**，其 PR 描述没有声称会终止进程。两者的契约互不替代。余下的决定是事件轮询成本能否接受，或是否存在无需轮询而又能给出终止上界的机制。现有四个假事件测试只证明 helper 行为，不能回答性能或恢复问题。本次源码 review 不触发硬件预约。

## 默认开启的问题及证据边界

在 PR 的精确基线运行 `git grep VLLM_ENGINE_ITERATION_TIMEOUT_S 157bcb7c489689dd34cf28d9c9970a465d326a03 -- vllm`，结果只有 `envs.py` 中的声明、环境变量表和变量名列表，没有运行时读取点。这不只是 Lab 旧版 `c8602c79` 清单的结论。补丁让输出路径读取该变量，[默认值为 60 秒](https://github.com/vllm-project/vllm/blob/d996d76ec68e9f6a348b7b085ae1da61cb6095be/vllm/envs.py#L802-L803)。V1 的复制流[先等待主计算流，再记录完成事件](https://github.com/vllm-project/vllm/blob/d996d76ec68e9f6a348b7b085ae1da61cb6095be/vllm/v1/worker/gpu_model_runner.py#L310-L329)；[V2 也是如此](https://github.com/vllm-project/vllm/blob/d996d76ec68e9f6a348b7b085ae1da61cb6095be/vllm/v1/worker/gpu/async_utils.py#L137-L168)。因此，事件就绪可能包含尚未完成的前向计算，而非仅等待复制。60 秒从 `get_output()` 进入拟议等待时起算，**不是**从步骤开始起算：若此时剩余 GPU 工作超过 60 秒，即使最终能完成，也可能报错。这是源码支持的可能性，不是已观察到的合法负载或已证实的回归。PR 的四个假事件测试没有这一慢但可完成的负控。60–300 秒的负控还能检验多进程超时顺序变化；未匹配真实负载和配置前，不应声称其有代表性。在已核查的 PR 讨论里，尚无人解决默认开启策略；njhill 留下的异议针对轮询。

一种建设性的策略选择是让超时默认值为 `0`（显式启用），或新增一个单独命名的 opt-in 设置。helper 的[非正数分支使用原生 `event.synchronize()`](https://github.com/vllm-project/vllm/blob/d996d76ec68e9f6a348b7b085ae1da61cb6095be/vllm/v1/worker/gpu/event_utils.py#L20-L26)，所以这样可在**默认情况下**同时避免轮询和新的致命上界。但这既不能解决显式开启后轮询是否可接受，也不能保护未开启设置的运营者免受原有 [#52247 卡死](https://github.com/vllm-project/vllm/issues/52247)。这是交给作者和维护者权衡的策略，不是已证实的修复。

另有一个**不放进拟议上游评论**的运维依赖：EngineCore 的致命异常路径本身不能证明 supervisor 会重启。Lab 的 [#52178 故障恢复表](../../experiments/fault-recovery/README.md)观察的是基线下 **SIGKILL** EngineCore 后顶层退出码为 0，而不是本 PR 抛异常的退出路径。此处异常的退出码和重启行为仍未实测。不要借他人的 PR 推广 #52178。

## 决策分支与时间

- 从 **2026-10-06** 起，即使维护者尚未回答范围问题，也可向作者简短提出 opt-in 默认值这一设计选择。先核对最新 head 和讨论；不自动发帖、开卡或 @ 任何人。若维护者随后回答，只提供与其回答有关的源码事实。
- 若作者或 reviewer 明确需要行为测量，先另行冻结测试：含慢但会完成的负控、卡住的事件组，以及硬件/时间上限和装置失败时的 `unscored` 结论。不能把假事件测试当成服务实测。
- 若到 **2026-10-26 至 2026-11-01 的复盘**仍无实质答复，则以 `not_observed` 关闭外部采用情况；若 PR 更早合入或关闭，则提前结束此候选。opt-in 问题属普通 Q5 review，**不算**十一月目标所需的 Lab 实证材料。

约在 10 月 1 日的 #52178 跟进已涉及 njhill。#52365 不安排在同一周，也不 @ 任何人。考虑发帖前须刷新讨论和 PR head。

## 真实慢步骤检索（2026-09-29）

纳入标准是：**未经修改、最终成功的服务负载**中，实测某一次 `get_output()` 事件等待进入后仍超过 60 秒未完成。TTFT、整个 prefill 阶段、启动、排队、编译或无限故障都不满足。已核查的公开报告仅是线索，尚不能纳入：

| 报告 | 不能证明该等待的原因 |
| --- | --- |
| [#51454](https://github.com/vllm-project/vllm/issues/51454) | 1M token 的首次提问约 66 秒，但报告使用 8,192 token 的批次上限。TTFT 跨越多个 prefill 步骤，未测量单次事件等待；其 B200／8 卡配置也无法在 Lab 的 4090 上复现。 |
| [#54919](https://github.com/vllm-project/vllm/issues/54919) | 长 prefill 期间的数分钟 decode 饥饿，后续分析指向重复的阻塞 D2H 同步及大量步骤，而非实测一次输出复制事件等待超过 60 秒。 |
| [#40707](https://github.com/vllm-project/vllm/issues/40707) | 调度器修复后，双视频请求在 130.7 秒完成；这是整次请求时长，没有单次事件等待数据。 |

检索结论：**没有找到可纳入的真实慢步骤组；本次公开报告检索到此结束**。公开报告很少有单次事件时间；继续寻找会成为缺少消费者决策的装置工作。不能把注入延迟当成生产发生率证据，也不应仅凭这些报告租 GPU。只有运营者已有自然变慢但最终成功的追踪，或作者／reviewer 明确要求，才重启测量。单次追踪须有 `get_output()` 进入、事件就绪、步骤身份和请求完成情况；只有实测等待超过 60 秒，才进入误报验证。

## E1 修订：Lab 负责人要求的有界实测

状态：尚未运行即由[实测补充 A1](PR52365_SECOND_CANDIDATE_ADDENDUM_A1_2026-09-29.zh-CN.md)取代。下文仅保留日期明确的设计历史；E1 的 1 秒单元不属于 A1 的开卡或评分。

2026-09-29，Lab 负责人希望在提出 review 问题前先有真实验证。本修订**增加单独评分的可选路径**，不重启公开报告检索，也不改变默认 60 秒的证据等级。`VLLM_ENGINE_ITERATION_TIMEOUT_S` [按整数解析](https://github.com/vllm-project/vllm/blob/d996d76ec68e9f6a348b7b085ae1da61cb6095be/vllm/envs.py#L802-L803)，因此 1 秒是最小的正数测试值。

1. 在一张 CUDA 卡上，以有限的有效 GPU 运算和真实 `torch.cuda.Event` 测试**未修改的 PR helper**。`timeout=0` 应等待并完成；对已校准、超过 1 秒才就绪的事件，`timeout=1` 应报超时，随后设备工作仍须完成且设备未重置。此项只评分 CUDA helper 行为；PR 现有假事件测试没有覆盖它。
2. 第一项通过后，才用真实模型、无模型路径人工 hold 的单卡服务验证 PR 精确 head。记录实际执行器／runner；以开卡前固定源码和摘要、只记录进入／返回／异常的 wrapper 测量导入的 `wait_for_gpu_event` 每次调用。模型和有界的 prompt／batch 梯度也须在开卡前选定。先在 `timeout=0` 下找到自然发生、超过 1 秒且最终完成的等待，再用全新服务在 `timeout=1` 下重放相同请求和配置；两组都加短请求控制。请求、EngineCore、health、顶层退出状态分别记录。这是**主动设低超时**的实验，不能声称默认 60 秒不安全。
3. `supported`：第一项通过，实测服务等待超过 1 秒，在 `0` 下成功而在 `1` 下失败，短请求控制成功。`refuted`：同一类实测等待与有效控制下，在 `1` 仍完成，与拟议截止时间相反。`unscored`：设置失败、自然等待未超过 1 秒、缺失计时或身份见证，或两组不可比。不得在运行后挑选有利子集。

准备工作最多两小时，单次 GPU 会话含设置最多三小时；若该会话内服务候选无法评分即停。不会自动扩为 TP=2 或 60 秒压力实验。固定源码的 GPU 结果即使不能解决默认策略问题，也可能对作者有用；但须经用户审核并选择分享，才算 Lab 交付。本修订仅就这一轮有界检查替代先前“仅在作者／reviewer 要求时测量”的门槛，不自行授权开卡。
