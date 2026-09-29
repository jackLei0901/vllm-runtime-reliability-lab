# 第二个外部候选：vLLM #52365

状态：仅完成源码预检，2026-09-29。本文为准。已核对 PR head `d996d76ec68e9f6a348b7b085ae1da61cb6095be`；交付前须重新核对。本记录不授权实验、开卡或上游评论。

## 消费者与决策

消费者是 [#52365](https://github.com/vllm-project/vllm/pull/52365) 的作者和 reviewer；该 PR 针对另一位用户报告的[生产环境 GPU kernel 卡死](https://github.com/vllm-project/vllm/issues/52247)。未决问题是：超时应覆盖异步 GPU 输出事件等待、整个 EngineCore iteration，还是由进程级 watchdog 负责。[维护者反对轮询](https://github.com/vllm-project/vllm/pull/52365#issuecomment-5303805293)；[作者询问 watchdog 应覆盖哪一层](https://github.com/vllm-project/vllm/pull/52365#issuecomment-5305822295)。PR 仍开放，但自述尚未准备合入。截至本次核查，讨论中没有找到维护者对该范围问题的答复。

## 增量比较：在后续工作前固定

| 问题 | 预检记录 |
| --- | --- |
| 普通复现 | 原 issue 已证明报告环境中 `copy_event.synchronize()` 无限等待且 `/health` 仍为绿色。PR 已用假事件测试轮询 helper。重复这两项不算 Lab 交付。 |
| Lab 增量 | 用固定源码版本梳理**谁在何处等待、哪个截止时间从何时起算**，对照单进程事件等待、多进程 RPC 等待，以及 [#55700](https://github.com/vllm-project/vllm/pull/55700) 仅用于诊断的 watchdog。产物必须指出一个具体设计选择或缺失的反例控制，不能只是代码路径列表。 |
| 初步源码事实 | [`AsyncOutputFuture.result(timeout)` 不支持超时](https://github.com/vllm-project/vllm/blob/d996d76ec68e9f6a348b7b085ae1da61cb6095be/vllm/v1/executor/uniproc_executor.py#L32-L44)。拟议的[事件等待在进入该等待时开始默认 60 秒计时](https://github.com/vllm-project/vllm/blob/d996d76ec68e9f6a348b7b085ae1da61cb6095be/vllm/v1/worker/gpu/event_utils.py#L20-L38)。多进程 [`execute_model` 和 `sample_tokens` 使用 RPC 预算](https://github.com/vllm-project/vllm/blob/d996d76ec68e9f6a348b7b085ae1da61cb6095be/vllm/v1/executor/multiproc_executor.py#L340-L363)，[默认 300 秒](https://github.com/vllm-project/vllm/blob/d996d76ec68e9f6a348b7b085ae1da61cb6095be/vllm/envs.py#L1794-L1795)，起点是 RPC 发送，而不是进入事件等待。因此在多进程场景中，拟议等待有可能先于 RPC 截止时间报错，但实际先后取决于 worker 何时进入等待。这只是源码推断，并非实测。 |
| 推翻该区分的结果 | 源码图发现此版本的单进程事件等待已经有上界、多进程输出路径不受 RPC 预算约束，或者现有 watchdog 已执行同样的终止动作。出现任一情况，交付前须撤回或修正上述区分。 |

## 有界下一步

最多用两小时做只读源码图和现有测试的反例控制审查。覆盖 V1/V2 的生成与 pooling、`UniProcExecutor` 和 `MultiprocExecutor` 调用路径、异常传播，以及 #55700 超时后实际采取的动作。未核实的状态转移应明确标为未知，不能推断超时必然重启服务。本预检不触发新探针或 GPU 预约。若结果只是普通源码 review 也能得到的表格，则归为 Q5 review 材料，**不计为 Lab 增量价值**。上游发帖前须刷新 PR head 和讨论、遵守项目贡献规则，并由用户审核措辞。

## 源码图检查点（非运行评分）

补丁覆盖四处 `get_output()`：[V1 生成](https://github.com/vllm-project/vllm/blob/d996d76ec68e9f6a348b7b085ae1da61cb6095be/vllm/v1/worker/gpu_model_runner.py#L331-L338)、[V1 pooling](https://github.com/vllm-project/vllm/blob/d996d76ec68e9f6a348b7b085ae1da61cb6095be/vllm/v1/worker/gpu_model_runner.py#L451-L460)、[V2 生成](https://github.com/vllm-project/vllm/blob/d996d76ec68e9f6a348b7b085ae1da61cb6095be/vllm/v1/worker/gpu/async_utils.py#L169-L174)和[V2 pooling](https://github.com/vllm-project/vllm/blob/d996d76ec68e9f6a348b7b085ae1da61cb6095be/vllm/v1/worker/gpu/async_utils.py#L248-L255)。`UniProcExecutor` 通过 [`AsyncOutputFuture.result()`](https://github.com/vllm-project/vllm/blob/d996d76ec68e9f6a348b7b085ae1da61cb6095be/vllm/v1/executor/uniproc_executor.py#L32-L46) 调用 `get_output()`；异常到达 [EngineCore 的结果等待](https://github.com/vllm-project/vllm/blob/d996d76ec68e9f6a348b7b085ae1da61cb6095be/vllm/v1/engine/core.py#L642-L650)。`MultiprocExecutor` 的 worker [把 `get_output()` 异常转换为 FAILURE 回复](https://github.com/vllm-project/vllm/blob/d996d76ec68e9f6a348b7b085ae1da61cb6095be/vllm/v1/executor/multiproc_executor.py#L985-L997)，接收方[再抛出 `RuntimeError`](https://github.com/vllm-project/vllm/blob/d996d76ec68e9f6a348b7b085ae1da61cb6095be/vllm/v1/executor/multiproc_executor.py#L426-L441)。EngineCore 的外层处理器[把未捕获异常作为致命错误](https://github.com/vllm-project/vllm/blob/d996d76ec68e9f6a348b7b085ae1da61cb6095be/vllm/v1/engine/core.py#L1431-L1438)。以上只是代码路径，未在服务中实测；前端状态和重启策略仍未评分。

设计差别已经明确：#52365 拟设置**终止性的事件局部**截止时间；#55700 的[自述目标](https://github.com/vllm-project/vllm/pull/55700)是漏喂心跳时作**诊断性栈转储**，其 PR 描述没有声称会终止进程。两者的契约互不替代。余下的决定是事件轮询成本能否接受，或是否存在无需轮询而又能给出终止上界的机制。现有四个假事件测试只证明 helper 行为，不能回答性能或恢复问题。只有作者或 reviewer 需要这项比较，或源码支持的反例控制能改变设计选择时，才考虑开卡。
