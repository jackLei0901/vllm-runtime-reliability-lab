# NCCL CUDA Graph profiler 计数器生命周期——源码走读

英文版：[NCCL_GRAPH_PROFILER_COUNTER_LIFECYCLE_SOURCE_REVIEW_2026-09-27.md](NCCL_GRAPH_PROFILER_COUNTER_LIFECYCLE_SOURCE_REVIEW_2026-09-27.md)。

## 问题与边界

在 [TP=2 限时 hold 实验](VLLM_TP_V2_OCCURRENCE_ID_RESULT_2026-09-27.md#post-run-offline-callback-audit)中，rank 1 尚未执行 graph replay，rank 0 却对 74 个当次 replay 的 occurrence ID 收到了 `KernelChStart` 和 `KernelChStop`。本走读解释固定版本源码中**如何可能发生**这种现象，不是新 GPU 运行、已确认的 NCCL 缺陷报告，也不改变冻结的 `unscored / target_collective_ambiguous` 结果。源码固定为 NVIDIA NCCL `v2.29.7-1`，提交 `b91894bd5b190c874d98a017f93f5daa515b65d0`。本地 checkout 仅 Inspector 文件有修改，下述 NCCL 核心文件与该提交一致。

## 两种 `persistent`，两种生命周期

| 状态 | 源码事实 |
| --- | --- |
| `plan->persistent` | 在 [`ncclLaunchPrepare`](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/enqueue.cc#L1507-L1529) 中根据 graph capture 设置。graph 保留该 plan 供 replay；[group 清理](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/group.cc#L391-L404)不会回收 persistent plan。 |
| `comm->planner.persistent` | 属于**当前提交任务使用的 planner**，不是被 graph 保留的 plan。在[准备任务](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/enqueue.cc#L360-L365)和[准备 launch](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/enqueue.cc#L1507-L1511)时赋值，group 结束后随整个 planner [清零](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/group.cc#L406-L424)。graph replay 不会重新运行任务准备过程。 |
| host 侧 profiler 计数器 | `comm->profiler.workCounter[channel]` 与上述两个字段独立。[`incWorkCounter`](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/proxy.cc#L549-L550)将递增或未递增的值复制到 `op->workCounter`。所检查的 [collective proxy op](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/include/proxy.h#L103-L115)要求递增。 |
| device 侧计数槽 | [通信器初始化](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/init.cc#L611-L617)通过[映射且清零的 host 内存分配](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/include/alloc.h#L125-L133)保存 `workStarted` 和 `workCompleted`。槽位按 [64 取模](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/include/device.h#L425-L430)复用；[设备代码](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/device/common.h#L318-L335)在 START 与 STOP/FINI 写入计数和时间戳。槽位并非每次 replay 独有。 |

## 从 capture 到回调的路径

1. 在检查过的 collective 调度路径中，[`calcCollChunking` 先将 `proxyOp` 清零](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/enqueue.cc#L2238-L2247)，故 `workCounter` 初值为零。[`addProfilerProxyOpIfNeeded`](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/enqueue.cc#L563-L569)查询是否需要 profiler proxy，并将操作复制进保留的 plan 队列（[入队](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/enqueue.cc#L106-L114)）。查询分支中的 [`SaveProxyProfiler`](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/proxy.cc#L553-L564)仅在 `comm->planner.persistent` 为 **false** 时递增；capture 时它为 true，入队操作因此可以保留 `workCounter=0`。
2. capture 的 persistent plan 安装 [`cudaLaunchHostFunc`](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/enqueue.cc#L1631-L1650)；[CUDA graph capture 记录而不执行 stream 工作](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)。replay 时，[`hostStreamPlanTask`](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/enqueue.cc#L1390-L1413)发出 collective 回调、上传保留的 proxy 操作并启动 proxy。此时 capture 对应的 collective group 已结束，当前 planner 已被清零。因此可以同时出现 `plan->persistent=true` 和 `comm->planner.persistent=false`。
3. replay 上传时，[`uploadProxyOps`](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/enqueue.cc#L1348-L1378)调用 `ncclProxySaveOp(..., nullptr)`。实际追加分支的 [`SaveProxyProfiler`](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/proxy.cc#L553-L564)仅在 `comm->planner.persistent` 为 **true** 时递增。若 planner 仍为清零状态，它不会递增保留操作的零计数。[`ncclProxyOpToArgs`](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/proxy.cc#L363-L392)再把该值复制到 proxy 的 sub-argument。
4. [`profilerProxyProgress`](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/transport/profiler.cc#L18-L48)设 `sub->base=sub->workCounter`，并以 `base <= slot[base % 64].counter` 判定 START 和 STOP。计数器是无符号数：当 `base=0` 时，**无论槽位保存什么值**，两个比较都会成立，不依赖槽位初值为零。START 在一次 proxy 轮询发出；`continue` 让同样已满足的 STOP 在下一次轮询发出。插件把回调归到当前 host 创建的 collective occurrence，但判定并未确认本次 occurrence 的 device work。

这是固定源码中一条具体的假阳性路径，不是只凭事件名称猜测。[replay 侧递增旁的注释](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/proxy.cc#L559-L563)明确表达了让 proxy 计数跟上 graph kernel 计数的意图；但在通常的 group 结束后 replay 路径中，保留的 `plan->persistent` 为 true，当前的 `comm->planner.persistent` 为 false，受后者控制的递增因而跳过。由此得到可证伪预测：**正常 graph replay 也可能提前发出 START 和 STOP**，hold 不是原因。它还解释了为什么插件的精确 occurrence ID 不等于精确 device-work ID：ID 绑定当前 host descriptor，而 proxy 判定读取的是通信器级计数槽。即使 `base` 非零，旧槽值（包括模 64 复用后的值）也可能满足 `<=`。候选修正是基于保留 plan 的 capture 属性，并在每次 replay 上传时推进计数，而非只在 capture 时推进；这仍是设计假说，不是已审阅补丁。host callback 运行时其他线程能否改变当前 planner，需要另做同步性审计。

## 实验支持什么、尚未证明什么

已校验摘要的 before/during 快照有 74 个不对称 occurrence ID；rank 0 在这些 ID 上新增 75 个 `kernel_ch_stop` 通道事件，覆盖全部 74 个 occurrence，rank 1 为零。rank 1 尚未 replay，因此这些回调不能证明 rank 0 对应的当次 collective 已完成。stock Inspector 的 `AllReduce` 完成记录总数同期增加 +21/+12，但这只是聚合值，不能与那 74 个 ID 逐个关联。观察与上述源码路径吻合。

本次实验没有在回调瞬间记录 `plan->persistent`、`comm->planner.persistent`、`op->workCounter`、`sub->base` 或被比较的槽位值。因此，它**没有证明** 74 个回调各自实际走了哪条分支、取到什么值，也不能外推到其他 NCCL 版本、路由或设备。走读证明的是计数器生命周期不匹配的源码可达路径，尚非独立复现的 NVIDIA 缺陷。

## Inspector 指标的潜在后果与现存数据检查

Inspector 把接口传来的 `pTimer` 分别存为[通道起始 GPU 时钟](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/plugins/profiler/inspector/inspector_plugin.cc#L259-L284)和[结束 GPU 时钟](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/plugins/profiler/inspector/inspector_plugin.cc#L454-L480)。它根据两者之差计算执行时间，GPU 数据无效时[退回 CPU 时间](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/plugins/profiler/inspector/inspector.cc#L1617-L1682)，再计算[带宽](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/plugins/profiler/inspector/inspector.cc#L1558-L1593)，写出 `coll_exec_time_us`、`coll_timing_source`、`coll_algobw_gbs` 和 `coll_busbw_gbs`（[JSON 字段](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/plugins/profiler/inspector/inspector.cc#L487-L504)）。若当前 occurrence 继承了其他工作的 `pTimer`，**现有 stock 指标**也可能错误。这是待检验后果，**不是**本次已确认的用户可见症状。

对[原结果固定的私有归档](VLLM_TP_V2_OCCURRENCE_ID_RESULT_2026-09-27.md#frozen-inputs-and-provenance)做了一次探索性离线检查：健康对照的 rank 0/1 分别有 388/390 条完成记录，其中 366/368 条是 `AllReduce`。全部 778 条标为 `kernel_gpu`，`coll_exec_time_us` 为零的记录数为零。详细模式中的 `event_trace_ts` 来自 [CPU 回调时钟](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/plugins/profiler/inspector/inspector_plugin.cc#L52-L56)，**不是**执行时间使用的 `pTimer` GPU 时钟。stock JSON 没有绑定单个 73 次 reduction 的 decode step 或插件 occurrence ID；相较每 rank 请求窗口内 1,184 次回调，保留的完成记录也很稀疏。因此，“同一步内起始时间聚集”无法打分。不过，每条记录内从 `coll_start_ts` 到 `kernel_start_ts` 的延迟不需要 step 绑定，可作为下文预注册的较弱回调时序检查。微秒取整后重复的执行时间、`kernel_gpu` 标签和非零带宽，单独都不足以支持或反驳计数器假说。

### 离线延迟检查：计算前预注册

此前已打开归档统计完成记录并检查字段名，但尚未计算逐记录延迟分布。先固定以下规则，再计算。按 rank 选取 `coll_start_ts` 和 `kernel_start_ts` 均为正的完成态 `AllReduce` 记录；每个有效 channel 计算一次 `kernel_start_ts - coll_start_ts`，单位微秒，负值单独计入完整性异常。按 rank 仅发布样本数、中位数、95 分位和低于 200 微秒的比例。另报告 `kernel_stop_ts - kernel_start_ts` 的描述统计，但不用于判定。不得输出原始时间戳、通信器 ID、PID、文件名或主机名。

同一 rank 的 `coll_sn` 若在保留的 `AllReduce` 中重复至少 10 次，将其列为 **graph 候选**分层，而不是证明其来自 graph；剩余记录另行描述，除非有独立路由证据，否则不称为 eager。只有两 rank 各有至少 30 条有效 graph 候选 channel 记录，且各自至少 95% 的延迟低于 200 微秒，才记为**支持提前回调**，不能据此判定 GPU 执行时间错误。若任一具有足够样本的 rank 其中位数大于 1 毫秒，则与广义的提前回调模式相矛盾。其余情形（含保留记录不足）均为不确定。这些阈值是探索性的，未以实测 step 时长校准。即使结果支持提前回调，也不能单凭它证明 `base=0`、确定 `pTimer` 对应的设备工作，或确认 stock 执行时间/带宽错误；这些仍需关卡 2 或独立的计数轨迹。

### 离线结果及预注册规则的覆盖缺陷

计算时未修改上述阈值。每个 rank 实际只有**一个**重复 `coll_sn` 的 `AllReduce` 组，分别包含同一通信器的 140/141 条完成记录、三类消息大小，产生 165/167 条有效 channel 延迟。rank 0/1 的中位数是 55/54 微秒，按 nearest-rank 计算的 95 分位是 115/99 微秒，低于 200 微秒的比例是 100%/99.4%；负延迟数为零。次要统计 `kernel_stop_ts - kernel_start_ts` 的中位数为 15/12 微秒。候选组之外，328/329 条 channel 延迟的中位数为 1,389.5/4,187 微秒，低于 200 微秒的比例为 29.9%/38.6%；这些记录**没有**独立证据证明是 eager。

预注册的数值门槛确实达到，但规则漏了必要的**graph 内位置覆盖检查**：graph replay 中，一个重复序号可能对应多个不同的 collective occurrence。stock 记录不能把该组绑定到 graph 内的位置，故短延迟不能证明靠后的 collective 也提前发出了 `KernelChStart`。Inspector 仅保留完成记录，还存在存活样本选择。因此关卡 1 证实了重复序号分层内的短延迟模式，却**不能判别** 74 个回调的假说，也没有证明 `pTimer`、执行时间或带宽错误。这一覆盖缺陷是在计算分布后才识别的，不是追溯修改已注册的阈值。关卡 2 仍是下一项决定性检查。

### 为什么不能把逐 occurrence 串行时钟当作不变量

NCCL 的[设备侧 profiler](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/device/common.h#L318-L335)会先遍历一个 work batch 的**全部** `nWorks`，写入各 work 的 START 时间戳；执行整批工作后，再写 STOP 时间戳。[kernel 循环](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/device/common.h#L393-L409)确认 START → 整批执行 → STOP 的顺序。因此，同一批中两个真实且正确计时的 work 也可能满足 `start_(i+1) < stop_i`。即使 sm_89 不支持 programmatic dependent launch，这也**不是**旧槽值证据。`PyNcclCommunicator.all_reduce` 还允许显式传入 stream，否则取当前 stream；通信器身份本身不能证明整个 serving run 只有一条 stream。跨 occurrence 比较需先绑定 batch/stream 边界。单个配对的 `stop_i <= start_i` 值得作为独立异常计数，但零异常也不能反驳计数器生命周期假说。

### 事后限定：强制 PyNccl TP 路由

上述 batch 反例是 NCCL 的**一般可能性**，不是本轮普通 Qwen3 TP collective 的已证实解释。在固定 vLLM `c8602c7` 中，[PyNccl `all_reduce` 与 `all_gather` 直接调用 NCCL API](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/distributed/device_communicators/pynccl.py#L166-L213)，[NCCL 为每个未分组调用建立并结束隐式 group](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/enqueue.cc#L2786-L2823)，随后按 [plan 的 stream 发起 kernel](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/enqueue.cc#L1558-L1567)。这里的显式 group 用于[列表输入](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/distributed/device_communicators/cuda_communicator.py#L594-L603)或[对称内存](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/distributed/device_communicators/cuda_communicator.py#L620-L647)的 all-gather，而不是普通强制 PyNccl TP all-reduce。

健康对照摘要中的 73 个相等时钟，对应同一通信器/通道的 73 个不同回调 occurrence ID；普通未分组 TP occurrence 应属于各自独立的 kernel 执行。这 73 个时钟不可能**全部**是各自执行的真实 START（或 STOP）时间。这是事后将源码与观察相连的推断，不是新预注册评分、`base=0` 的直接证明或 stock Inspector 指标结论。公开摘要没有保存该相等类的函数构成；私有日志仍值得按函数分层核查，并做未显式分组的独立 graph 复现。相邻逆序仍须先绑定实际 stream 和 occurrence 顺序才评分，虽然分别提交的执行之间出现相等时钟本身已异常。

## 审阅结论与下一步验证

对于 graph replay 中的 `KernelCh` 事件，撤回“只缺 Inspector 导出策略”的结论：直接导出这些回调可能将错误的当前设备进度对外发布。当前不新增 Lab probe、不改变 v0.2 verdict，也不凭单次 serving hold 提上游 issue。

按成本顺序设置后续关卡：

1. **现存 stock JSON：已完成。** 数值门槛虽达到，但每个 rank 只有一个重复序号组，不能证明 graph 内后续位置也被覆盖。保留闭合分布及这一判别失败，不升级为设备计时正确性结论。
2. **只改插件的能力检查：已完成一次双卡预约。** [固定结果](NCCL_PTIMER_DUAL_GPU_RESULT_2026-09-27.zh-CN.md)记录了 eager 中可区分且递增的时钟，以及健康强制 PyNccl TP serving 中 73 个 occurrence 的相等类。一般 work-batch 例外不足以解释分别提交的普通 TP 调用。这是实验与源码结合的事后推断，不追溯修改协议评分。私有归档尚未下载后复核摘要；取回后按通信器/通道/函数检查相等时钟，并统计 hold 窗口内低于历史最大值的时钟。尚未计算的数值不得写成观察。
3. **先独立复现症状，再考虑 core 插桩。** 双 rank 捕获由有界设备工作隔开的未分组 all-reduce，对比 eager 与 graph replay，保留独立计数轨迹。这样可以排除 vLLM，直接检验回调时钟是否属于它所命名的执行。只有机制仍有实质歧义时，才在另一个 NCCL-core 构建中有界记录 `plan.persistent`、`planner.persistent`、`op.workCounter`、`sub.base` 及所比较的槽值；新构建必须有独立库身份。stock 指标错误仍需把完成记录绑定 occurrence，并有独立计时参照；NVIDIA issue 还须经过独立复现及 09-25 上游决策门槛。
