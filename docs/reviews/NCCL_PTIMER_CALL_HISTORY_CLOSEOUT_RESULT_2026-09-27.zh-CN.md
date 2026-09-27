# NCCL profiler 时钟调用历史收尾：装置未进入评分阶段

英文版：[NCCL_PTIMER_CALL_HISTORY_CLOSEOUT_RESULT_2026-09-27.md](NCCL_PTIMER_CALL_HISTORY_CLOSEOUT_RESULT_2026-09-27.md)。本次按 Lab 提交 `38cee45` 的[冻结协议](NCCL_PTIMER_CALL_HISTORY_CLOSEOUT_GATE_2026-09-27.zh-CN.md)执行，但**没有得到可评分的 A/B/C 对照**。

## 结论

A 组 `no_eager` 在 replay 前的第一次 callback 快照停止。两份 rank 日志各只有 **2 条**已完成 collective occurrence，属于**两个不同 communicator** 的构造 warm-up。运行脚本错误地在这个时点要求 5 条，把 graph capture 的三次调用也当成已经出现在 callback 日志中。限定等待结束后，rank 0 抛出 `bounded profiler snapshot unavailable`，rank 1 随后失去 Gloo 对端；驱动非零退出。插入调用和 graph replay 均未执行，B、C 组也没有启动。

这个“replay 前必须有五条”的要求是新增协议的**实验装置错误**，不是时钟归属证据。本次只证明在该 replay 前窗口没有出现预期的三条 capture callback；不能据此概括它们的一般投递时机，更不能判断内部 planner、计数器、`sub->base`、原版 Inspector 指标，或同 communicator eager 调用是否影响 replay 时钟。[此前 serving／独立对比](NCCL_PTIMER_GRAPH_COMPARISON_2026-09-27.zh-CN.md)提出的调用历史假设仍未证实，也未被推翻。

| 组别 | 结果 | 范围 |
| --- | --- | --- |
| A `no_eager` | `unscored / pre_replay_callback_assumption_failed` | 每 rank 两条 warm-up；没有 replay 或时钟评分。 |
| B `same_comm` | `not_run / gate_stopped` | 无观测。 |
| C `other_comm` | `not_run / gate_stopped` | 无观测。 |

## 证据与界限

双卡实例及 SSH 指纹与此前独立确认值一致；传输后的运行脚本、评分器和 Inspector 二进制哈希均符合冻结协议。所选环境报告 Torch `2.13.0+cu130`、vLLM `0.1.dev586+gc8602c790.precompiled`。程序完成 communicator 构造及 graph capture，随后在快照处停止；实际加载 NCCL 的身份检查没有拒绝，但因中途停止，没有产生最终运行清单。

原始 rank 日志、程序输出、冻结脚本及依赖仅在私有归档中。归档 SHA-256 在远端及下载后均为 `853d3f276a3d6a11ebe68b3276a4b27ca8a72617ed4a1e309ceda845866872ce`；共 22 个成员，没有绝对路径、路径穿越或链接。不得公开原始日志、communicator 身份、时钟、PID 或主机路径。上述事件计数来自对留存日志的只读解析。输入未通过评分器门槛，因此不报告等值类或假设结论。

归档传回并复核后已执行 `shutdown -h now`，SSH 随即断开。断连**不能证明**云平台已停止计费，仍需用户在控制台确认。

## 收尾决定

遵守预注册的一次开卡规则：本机未临时改脚本，也未重跑。NCCL 时钟归属候选问题以**未解决**状态暂存；不据此提出 NVIDIA issue、Lab 探针或原版指标错误主张。未来若要重试，必须另行审阅协议，移除 replay 前必须有 graph callback 的错误要求，并重新决定是否值得投入。这次失败不重启真实 TP 飞行中定位主线。
