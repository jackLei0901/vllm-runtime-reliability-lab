# 首个外部候选：vLLM #55700

状态：2026-09-28 本地选题记录；尚无测试结果或上游评论。本文为准。源码固定到 PR head `b274bf04dd4c6d54807a136babce5b5d17dd74be`；测试或发帖前重新核对。

## 接收方与决策

接收方是他人发起且仍活跃的 [vllm-project/vllm#55700](https://github.com/vllm-project/vllm/pull/55700) 的作者和评审。评审[要求提供可供运维告警的超时计数](https://github.com/vllm-project/vllm/pull/55700#pullrequestreview-5263026590)，作者已加入计数指标。要回答的窄问题是：**一次尚未恢复、当前步骤仍阻塞的超时**，能否及时反映到指标；这不同于恢复之后补报超时。

## 预先固定的增量比较

| 问题 | 测试前的记录 |
| --- | --- |
| 普通复现 | 心跳未喂入会产生栈转储；不能说明 `/metrics` 何时变化。 |
| Lab 增量 | 比较阻塞期间、解除阻塞后和 RPC 超时终止后的计数，并查明事件跨越 worker、EngineCore、前端边界的条件。 |
| 源码推论 | watchdog 后台线程记录超时。EngineCore 路径由 [`update_from_output()` 调用 `Scheduler.make_stats()`](https://github.com/wenjinhust/vllm/blob/b274bf04d/vllm/v1/core/sched/scheduler.py#L2343-L2355)，后者[读取计数](https://github.com/wenjinhust/vllm/blob/b274bf04d/vllm/v1/core/sched/scheduler.py#L2874-L2876)；多进程 worker 路径由 [`enqueue_output()` 把计数附在模型输出上](https://github.com/wenjinhust/vllm/blob/b274bf04d/vllm/v1/executor/multiproc_executor.py#L993-L1013)，此时 `get_output()` 已返回。[Prometheus 从 `SchedulerStats.watchdog_stats` 增量](https://github.com/wenjinhust/vllm/blob/b274bf04d/vllm/v1/metrics/loggers.py#L1110-L1115)。若步骤始终不返回，两条路径可能都无法发布计数。这尚未经过运行验证，不能称为已证实缺陷。 |
| 推翻该推论的结果 | 步骤持续被阻塞、没有完成的步骤输出时，`/metrics` 的计数仍然增长。 |

EngineCore 和 worker 的可见性分别评分：单进程 TP=1 主要涉及 EngineCore 路径；多进程 TP>1 的 worker 阻塞还涉及 worker 路径。另设独立评分的问题：[多进程 `execute_model` 和 `sample_tokens` 只要求 `output_rank` 回复](https://github.com/wenjinhust/vllm/blob/b274bf04d/vllm/v1/executor/multiproc_executor.py#L341-L362)。验证其他 rank 的 watchdog 统计量是否抵达前端；不能仅凭源码路径就声称这些计数已丢失。

## 有界下一步

最多投入约四小时 CPU 工作，针对准确的 PR head 先走读完整指标路径，再编写能区分三种情况的测试：（a）阻塞且未恢复，（b）阻塞后恢复，（c）短于 watchdog 超时的正常工作。假阻塞必须释放 GIL（例如使用 `threading.Event.wait()`），让同进程的 Python watchdog 线程能够运行。显式设置非空 `dump_dir` 和较短的 `timeout`、`check_interval`，且保持阻塞时间短于 RPC 截止时间：[空 `dump_dir` 会禁用 watchdog](https://github.com/wenjinhust/vllm/blob/b274bf04d/vllm/utils/watch_dog.py#L223-L249)，默认检查间隔为 10 秒。记录阻塞前的计数基线。阻塞**期间**必须观察到转储文件或成功写入的 `feed timeout` 转储日志，才允许评分。对 EngineCore 和 worker 分别评分：有触发见证且 `/metrics` 保持基线为 `supported`；有触发见证且持续阻塞期间计数增长为 `refuted`；无触发见证、环境失败或 CPU 假对象不能忠实代表传输路径均为 `unscored`。不得用复制逻辑的测试冒充 PR-head 验证。不自动租 GPU 或发上游评论。拟公开前重新固定 PR head，检查贡献与 AI 辅助披露规则，交由用户审核结论和措辞。

本工作不声称 watchdog 等于进展检测，不否认恢复后计数的价值，也不预设评审要求所有卡住场景都实时上报。
