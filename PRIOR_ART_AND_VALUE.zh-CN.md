# 公开先例与关联价值

[English](PRIOR_ART_AND_VALUE.md)

本文记录“多份局部数据经过关联后产生新事实”的公开案例。它不主张训练故障的
分布或收益数字可以直接迁移到推理服务。

## PyTorch Flight Recorder

### 使用前的痛点

PyTorch 将 NCCL watchdog timeout 描述为一种信息不足的汇总症状：最先报告
超时的 rank 往往不是故障起点，超时时看到的 collective 也可能只是前一个错误
的后果；watchdog 线程栈还缺少主线程调度 collective 时的上下文。公开文章提到，
没有 Flight Recorder 时，定位可能耗费数小时，并且需要带着更多调试开关重跑。

来源：[Flight Recorder: A New Lens for Understanding NCCL Watchdog
Timeouts](https://pytorch.org/blog/flight-recorder-a-new-lens-for-understanding-nccl-watchdog-timeouts/)。

### 关联机制

每个 rank 在 CPU 侧保留有界记录，包括 collective 类型、生命周期状态、dtype、
size、调用栈和 process-group 内的 sequence。事故后再跨 rank、跨 process group
对齐这些记录。

关联后的 mismatch 是一种关系事实：缺失 peer、不同生命周期状态、不兼容的
操作元数据或参数不一致，都不能由单个 rank 独立定义。

PyTorch 使用独立的 TCPStore 控制路径请求 rank monitor 尽力写出 dump，之后由
外部编排收集本地文件。文章报告其在 Meta 环境中取得接近完整的 full-dump
覆盖率，并解释了为什么在超时状态已经分散的情况下选择离线分析。

### 使用后解决了什么

公开案例最初看起来像不同 `all_to_all` 变体之间的问题。对齐多个 rank 的调用栈
和调度顺序后，发现部分 rank 已进入后续 collective，而另一些仍停留在前一个
collective。关联证据排除了最初的判断，并把问题收敛到执行顺序分歧。

公开文章没有给出受控的 MTTR 降幅或节省的 GPU 小时。因此，本项目只把它作为
“关联能够修正假设”的定性证据，不把它当作量化收益基线。

### Join key 的教训

PyTorch 后续记录过一个限制：当 process group 同时包含 collective 和点对点
操作，且 ring buffer 头部已经被覆盖时，单一 sequence ID 不足以恢复对齐基准。
这说明身份与序列设计必须在 schema 固化前验证，不能把一个本地递增序号当成
完整的跨进程 join key。

来源：[pytorch/pytorch
#125173](https://github.com/pytorch/pytorch/issues/125173)。

### 一个可以直接验证的公开缺口

同一篇 PyTorch 文章明确指出，要区分 CPU 普通操作、barrier、CPU-GPU 同步点和
异常处理，Flight Recorder 还需要配合分布式 CPU 主线程栈视图；PyTorch 当时
没有提供这类诊断工具，并指出可使用 `py-spy` 等开源工具采集底层数据。

这给本项目留下了一个边界清晰的问题：从进程外采集各 rank 的 stack snapshot，
绑定已声明的 process/rank 身份，再在事故后与已有 Flight Recorder artifact
关联。第一轮首先验证真正卡在 NCCL collective 的 rank 能否被采样；只有该 gate
通过，才使用受控 execution divergence 比较相同 stack/FR 输入在未关联和已关联
时能否产生新事实。它不承诺检测任意 hang，也不会让 `py-spy` 成为核心运行时
的强制依赖。

FR 已经能够在 collective 的逻辑位置上指出缺失或不一致的 ranks。新增 stack
只回答一个更窄的问题：该 rank 的 CPU 线程当时在做什么。归一化 frame 匹配可以
支持部分关联，但如果 rank 在调度缺失 collective 之前就卡住，它并不适用。设计
不能用未同步 wall time 推断“哪个 rank 最先出错”。

原始 stack 可能暴露源码路径或业务相关符号，因此默认按私有材料处理，并保持在
当前公开 schema 之外。

## 可以迁移到本项目的设计

- 每个 producer 在故障前持续维护有界记录；
- 本地尽力持久化，不要求故障时发起新的 collective；
- 单独量化 capture coverage；
- 事故后再做身份、序列和时间对齐；
- 先完成语法关联，再比较不同 producer 的语义；
- 将排除错误假设视为有效结果。
- 可选地从进程外采集 CPU 主线程栈，并按 rank 进行关联。

## 不能直接迁移的结论

- Meta 的训练故障比例不能代表 vLLM 推理服务；
- 推理副本和角色并不总是对称的 SPMD 拓扑；
- 外部进程/GPU 观察无法复制 c10d collective 元数据；
- missing producer 或状态分歧通常只能缩小范围，不一定给出唯一根因；
- Meta 环境中的 dump 覆盖率不能证明本实现也能达到同样结果。

## 对本项目的直接要求

v0.2 的第一项价值实验使用完全相同的 producer artifacts，比较两组结果：

1. 只有彼此独立的文件，没有 topology、clock alignment 和统一视图；
2. 加入关闭的 manifest，以及有限的 vLLM process/progress semantic join。

关联必须产生可校验的新事实，不能只把文件放进同一个压缩包。结构性指标包括
capture coverage、join coverage、missing producer 检测、hash 校验，以及在
共享 logical position 上定位 mismatch。人的诊断收益单独衡量：排除了哪些假设，
以及多久能够确定下一步动作。
