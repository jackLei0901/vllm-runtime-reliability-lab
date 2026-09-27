# NCCL graph-profiler 时钟：离线对比与修正后的调用历史假说

英文正文：[NCCL_PTIMER_GRAPH_COMPARISON_2026-09-27.md](NCCL_PTIMER_GRAPH_COMPARISON_2026-09-27.md)。本文是**事后分析**，不是新 GPU 运行、旧评分重算或 NCCL 缺陷证明。对比[健康 V2 服务 cell](NCCL_PTIMER_DUAL_GPU_RESULT_2026-09-27.zh-CN.md)与[独立 graph cell](NCCL_PTIMER_STANDALONE_RESULT_2026-09-27.zh-CN.md)；服务 hold 仍为 `unscored / target_collective_ambiguous`。

## 输入与方法

两份私有归档下载后二次哈希均与远端一致：服务 `17d3c0a687e2d7f555591fbdc54dcc7512421fb1bb76305895420ba0b7849d3e`，独立用例 `fb393ef17415ff3d9c663b3d838c98c9760a3c10ac4f91aabeab0a16767dd757`。服务摘要 SHA-256 为 `3669c55f0ac1691c03dfd240cacfbedef35200b2d90453eb904791051f506d37`，独立 graph 评分为 `3ec7305da4ef6ca422df3ed9f0ae079b317bf619b15a497f583b0f5b6e458070`。

[离线脚本](../../experiments/vllm-tp-dfx/ptimer_graph_compare_offline.py) SHA-256 为 `ba7bf879c0dbf6b86c46ef0dfeb75bb3b092fbcbc2535bc5f773c455ca52b07e`。它检查 before/after 快照哈希、rank 日志与留存摘要的哈希一致性、两 rank 的事件历史及 descriptor 完整性、pTimer marker 覆盖；只输出闭合的函数/channel 计数及等值类大小，不发布 communicator ID、原始时钟、PID 或主机路径。已完成语法检查并在两份留存 cell 上运行。该脚本是新增的描述性分析，不是新判定规则。

## 留存证据中的差异

| 事实 | 健康 V2 服务 | 独立 graph |
| --- | --- | --- |
| 图与运行路径 | V2 Model Runner，配置为 `FULL_AND_PIECEWISE`；日志报告 cached FULL replay，强制 `PYNCCL`。回调日志**没有**独立标记 1,184 次请求 collective 各自的 graph mode。 | 直接用 `torch.cuda.CUDAGraph` 捕获三次 `PyNcclCommunicator.all_reduce`，中间插入有界 `_sleep`；两次 replay 数值检查通过。 |
| 请求窗口函数序列 | **每个 rank** 都是连续 16 组：73 次 `AllReduce` 后接一次 `AllGather`；共 1,168 + 16 = 1,184 个 occurrence。 | 每 rank 两组，每组三次 `AllReduce`；共 6 次，无 `AllGather`。 |
| channel 覆盖 | 1,095 次单 channel 和 73 次双 channel `AllReduce`；16 次 `AllGather` 均为双 channel。每 rank 的 START/STOP 各 1,273。 | 六次 `AllReduce` 均为单 channel；每 rank START/STOP 各 6。 |
| 评分窗口前最后一次 collective | 两 rank 均为 `AllGather`，与窗口中 AllReduce 使用同一通信器。观测到的每个请求组末尾 AllGather 也使用该通信器。 | 两 rank 均为 `AllReduce`（捕获图以它结束）；runner 在 replay 前没有插入 eager collective。 |
| 按函数分层的 START 等值类 | `AllReduce`：rank 0/1 的不同 occurrence 最大等值类为 73/78；`AllGather`：两 rank 均为 1。STOP 对应为 `AllReduce` 73/77、`AllGather` 1/1。 | `AllReduce` 的 START/STOP 在两 rank 上最大等值类均为 1。 |

服务侧等值类按 communicator、channel、函数和事件种类分组，不是把无关通信器或 channel 混在一起。首个服务组有较多不同时钟；其后每组 `AllReduce` 的 channel-0 START 仅有一到两个不同值。这与旧槽位复用吻合，但快照没有将每条回调绑定 CUDA graph 节点，也没有 `sub->base`。AllGather 与 AllReduce 的差异是观察事实，不能单凭它证明因果。

## 源码更正与可证伪解释

旧版[计数器生命周期走读](NCCL_GRAPH_PROFILER_COUNTER_LIFECYCLE_SOURCE_REVIEW_2026-09-27.zh-CN.md)错误地把 [`groupCleanup` 清零 planner](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/group.cc#L406-L424)当成成功结束 group 的常规行为；该函数在[失败路径](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/group.cc#L716-L729)调用。成功情况下，planner 在[同一通信器下一次加入 collective group](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/include/group.h#L125-L131)时才重置。[任务准备按 capture 状态设置 `comm->planner.persistent`](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/enqueue.cc#L360-L365)，而 [replay 侧 profiler 追加分支仅在该字段为 true 时推进计数](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/proxy.cc#L553-L564)。

由此得到的**假说**依赖调用历史：capture 后直接 replay 可保留 persistent planner，与独立用例无重复相符；若中间在**同一通信器**执行 eager collective，新 planner 可留下非 persistent 状态，使后续 replay 即便使用 persistent plan 也跳过 profiler 计数递增。服务窗口中同一通信器的 AllGather 排在每组 73 次 AllReduce 后；runner 先进行[模型执行](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/v1/worker/gpu/model_runner.py#L1390-L1448)，随后[采样并计算 logits](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/v1/worker/gpu/model_runner.py#L1143-L1150)。这同时解释了独立阴性结果，以及服务重复类集中于 AllReduce 而非 AllGather。

这**还不是内部机制实测**。两轮均未记录回调时的 `comm->planner.persistent`、保留的 `plan->persistent`、`op->workCounter`、`sub->base`、槽位计数或有效 stream。服务回调也未逐 occurrence 标明 graph mode。因此不能宣称 73 次 reduction 都处于同一个 graph，不能宣称已经测得 `base=0`，也不能宣称 stock Inspector 执行时间/带宽错误。

## 若再次开卡，先冻结的下一实验

已运行的 capture → 直接 replay 保留为 A 对照。新矩阵应在开卡前固定：B 在 capture 和 replay 之间，在**同一通信器**上插入一次 eager AllReduce；C 在**另一个经过身份验证的通信器**上插入等价 eager collective。如果只有 B 出现回调时钟复用，调用历史机制才得到干净的独立复现。可再用有界预热变体检验槽位历史是否把零时钟变为重复的非零时钟。精确的零/非零结果**不能**由现有归档先验断言，须实测。数值完成、双 rank 身份、库身份、回调完整性和隐私检查仍沿用上次独立门槛。此区分验证前，不启动 NCCL-core 改造、不提上游 issue、不新增 Lab 探针。
