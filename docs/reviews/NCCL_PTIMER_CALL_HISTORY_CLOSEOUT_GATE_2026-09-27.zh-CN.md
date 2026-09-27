# NCCL profiler 时钟归属：一次开卡的收尾门槛

英文版：[NCCL_PTIMER_CALL_HISTORY_CLOSEOUT_GATE_2026-09-27.md](NCCL_PTIMER_CALL_HISTORY_CLOSEOUT_GATE_2026-09-27.md)。本协议在下一次 GPU 运行**之前**冻结。它只收尾 callback 时钟归属候选问题，不重启 Lab 的 TP 飞行中故障定位主线。

## 要回答的问题

[既有独立复现](NCCL_PTIMER_STANDALONE_RESULT_2026-09-27.zh-CN.md)在 capture 后直接 replay，未见时钟复用。[离线对比](NCCL_PTIMER_GRAPH_COMPARISON_2026-09-27.zh-CN.md)显示真实 serving 的 AllReduce 最大等值类为 73/78，而 AllGather 和独立复现均为 1。Serving 每步在同一 communicator 的 AllReduce 之后执行 eager AllGather。[修正后的 NCCL 源码走读](NCCL_GRAPH_PROFILER_COUNTER_LIFECYCLE_SOURCE_REVIEW_2026-09-27.zh-CN.md)提出一个尚未测得内部状态的假设：replay 时 profiler 计数是否增长，可能取决于同一 communicator 上一次调用留下的 planner 状态。

本次仅检验：在 capture 与 replay 之间插入一次同 communicator 的 eager AllReduce，是否使 callback 时钟出现复用；换成不同 communicator 是否不会。它不验证 TP hang、设备进度、原版 Inspector 的时延字段或修复补丁。

## 固定三组

每组使用全新双 rank 进程和仅所有者可访问的目录；两只 communicator 在每组都创建，构造时的 warm-up 位于 capture 前。

| 组别 | 调用顺序 | 预期 |
| --- | --- | --- |
| A `no_eager` | primary 上 capture 三次独立 AllReduce → 两次 replay | 复现此前“不复用”的对照。 |
| B `same_comm` | 同样 capture → primary 上一次 eager AllReduce → 两次 replay | 若调用历史假设成立，两 rank 的 callback 时钟应复用。 |
| C `other_comm` | 同样 capture → 独立 secondary 上一次 eager AllReduce → primary 上两次 replay | 若问题局限于 communicator，则不复用。 |

评分器要求两 communicator 身份确实不同、每 rank 窗口前恰有五次 AllReduce（两次 warm-up、三次 capture）、插入调用落在指定 communicator、replay 的六次调用落在 primary、channel 标记齐全且双 rank 一致。前／中／后快照将 eager 插入与 replay 分开。所有 eager 与 replay 数值结果均须正确。不预设时钟必须为零或非零；只评分不同 occurrence 间的时钟等值。看过 A 的结果后不得改脚本或阈值。

目标环境：两张 RTX 4090、双 `torchrun` rank，已验证的 vLLM `c8602c79062440074a018c1d5f875a5571eb6881`、Torch `2.13.0+cu130`、加载的 NCCL `2.29.7`（两 rank SHA-256 `aa957cdfb91b516eae0d54a28e9ee5db52730d02e0ab45580efc3c19a68327a4`）、Inspector 二进制 SHA-256 `ef734dc33acd60af9edc43b8ec84204c70e3d1fde53294478dc5b08e0d352c3b`。身份变动先停为 apparatus 问题，不得默默替换。[驱动脚本](../../experiments/vllm-tp-dfx/run_ptimer_call_history.sh)会在运行前检查二进制和源码哈希，运行脚本会核对 vLLM、Torch 及实际加载的 NCCL 身份。新增[运行脚本](../../experiments/vllm-tp-dfx/ptimer_call_history.py) SHA-256 `61d939062274819f8f41ad330563ac45931f343524695190333196987d6d44d3`；[评分器](../../experiments/vllm-tp-dfx/ptimer_call_history_score.py) SHA-256 `f869729baf414b58545a1d7db435a525922498897addb1cce2674d08ad3bb59f`。

## 执行与停止规则

1. 最多 30 分钟只读预检：双卡、venv、加载的 NCCL、插件哈希、磁盘空间、`torchrun`、`ninja` 和用户独立确认的 SSH 指纹。不得为过关而删除数据或改动现有环境。
2. 按 A、B、C 顺序一次运行 `bash run_ptimer_call_history.sh VENV_DIR PLUGIN_SO PRIVATE_BASE LAB_ROOT`，每组 180 秒上限。构建、超时、数值、rank 绑定、occurrence、channel 或哈希失败即记 `unscored`，保留现场，不调参、不重跑、不挑有利的 rank 或 collective。
3. 将原始日志、快照、命令、环境与评分 JSON 保存为仅所有者可访问的私有归档；远端和下载后分别哈希，解包前检查成员名与链接。公开材料只保留封闭计数、摘要与版本身份，不公开原始时钟、communicator hash、PID、主机名、IP、trace 和私有路径。
4. 证据转移并复核后，按用户授权关机；SSH 断开不代表云平台停止计费，另请用户确认平台电源状态。

本机准备：新增三个 CPU 评分器测试及 `PYTHONPATH=src` 全套测试通过（**318 项，跳过 15 项**）；新运行脚本通过 CPython 3.14 编译检查，shell 驱动通过 Git Bash `bash -n`。这些检查不等于 GPU 路径或 profiler callback 语义已经验证。

## 预先冻结的解释

| A／B／C | 结论 |
| --- | --- |
| A 不复用，B 双 rank 复用，C 不复用 | `conditional_callback_reuse_reproduced`。准备可独立审阅的 NVIDIA 问题材料；仍非 `base=0` 或原版指标错误的证明。TP 定位主线保持暂停。 |
| A、B、C 均不复用 | `candidate_not_reproduced`。真实 serving 与独立实验的差异仍未解释，停止此方向，不再预约第二次。 |
| A 或 C 复用、单 rank 复用、结构校验失败或其他不一致组合 | `unscored_or_confounded`。留存证据并停止，不选择有利子集。 |

本结果不准入 Lab 新探针。若未来重新开启真实 TP 飞行中定位，需另有真实故障、新的可信 producer 能力，或通过 Lab §5 探针准入门槛。
