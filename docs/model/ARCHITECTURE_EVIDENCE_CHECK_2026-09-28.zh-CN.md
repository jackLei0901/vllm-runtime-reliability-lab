# 运行时模型证据核验 — 2026-09-28

状态：本地可审阅结果；不是新的 v0.2 verdict，也不是 40 项抽样结果。[English](ARCHITECTURE_EVIDENCE_CHECK_2026-09-28.md)。本记录检查[运行时模型](RUNTIME_MODEL.zh-CN.md)能否把读者从事实带到有界测试和结果；模块归属仍见[架构映射](ARCHITECTURE_MAPPING_2026-09-27.md)。

## 四项实际交付

1. **冻结顺序的抽样记录校验器：**[代码](../../scripts/validate_pain_point_labels.py)及[17 个合成/变异测试](../../tests/test_pain_point_label_validator.py)。真实的 823 编号快照配空的 `in_progress` 台账可通过，不读取任何 issue 正文。校验精确审读前缀、快照字节和协议提交身份、从快照分片得出的命中词、标题和正文分别计算的摘要、登记的排除代码、第 40 个合格项停止点、M-part 与证据形状，以及从第 10 个首轮标签完成后计算的七天复核间隔。标注时间须晚于快照完成且不能沿审读顺序倒退；固定的 issue 与评论版本不能晚于标注。重标须单独固定标题/正文摘要、更新时间和证据指针；报告发生变化时一致率不评分。只打印计数，不打印 issue 编号或报告正文。它不能判断人工标签是否正确，也不能证明重标时真的看不到首轮结果。
2. **G0 CPU 身份矩阵：**现有[绑定实现](../../experiments/vllm-engine-binding-gate/binding_evidence.py)与[控制](../../experiments/vllm-engine-binding-gate/test_binding_evidence.py)在本机 Windows Python 3.14 环境复跑 **21/21** 通过；独立的[规则组合控制](../../experiments/vllm-engine-binding-gate/test_cpu_contract.py) **7/7** 通过。合成矩阵覆盖精确重复前缀、冲突身份、直接子进程/起始时间/PID namespace、重启和日志保管缺失。这是已有控制的复跑，不是真实 DP 的 engine→PID 观测。
3. **#53859 干净环境重放：**从公开 `v0.2.0` 标签克隆新 checkout，提交为 `e0a5975e750bd0cabaf0b88437c42b18323b4816`，新建 Python 3.14.2 venv。README 的直接 `pip install .` 在构建前因本机包索引代理下载 `setuptools>=68` 返回 403 而失败。随后用宿主机已有的构建工具从**同一 checkout**制成 SHA-256 为 `c0632469077371ec1dca25ea24cc7f459a109e17ade54363665b84cd50c8da8f` 的 wheel，以 `--no-index --no-deps` 安装进新 venv，再运行文档中的 `vllm-dfx replay`。退出码为 0，完整重放[公开四 cell 结果](../../results/vllm-zmq-backpressure-stage1-r3-20260916/)：EngineCore 存活、无进展期间 `/health` 为 2xx、publisher/queue 栈、修复臂完成，以及四个事件批次被丢弃。私有输出 SHA-256 为 `0fa42a0fb15bb06a42a05ede248274e5b2a556e2a2d8d3a5ec8aa5746a9daf51`。这证明本地构建 wheel 的隔离安装与离线重放，不证明依赖网络的 `pip install .`、新 GPU 运行或非作者采用。
4. **证据追溯：**下表给每个模型部分列出真实控制及其边界。链接到一个结果不等于给模型新增分数。

## 模型 → 证据 → 剩余边界

| 模型部分 | 保留证据与控制 | 当前可以说什么 | 尚未证明 |
| --- | --- | --- | --- |
| M1 拓扑/身份 | [G0 CPU 控制](../../experiments/vllm-engine-binding-gate/LOCAL_CPU_GATE_2026-09-25.md)与上述 21 项复跑 | 精确前缀加进程树的候选 join 在身份冲突或缺失时拒绝绑定。 | 各启动方式的真实 engine label→PID 绑定；Ray 和 DP>1 行为。 |
| M2 生命周期 | 暂无直接的 Lab 状态转移测试。[K1/K2 CPU 结果](../reviews/ENGINE_LIVENESS_K1_K2_CPU_RUN_2026-09-27.md)只通过 M5 预算触及 shutdown。 | M2 已纳入契约映射，但尚无独立评分的状态转移结果。 | shutdown 完成、启动达到 ready、pause/resume、sleep/wake 的转移；生命周期状态进入冻结 verdict。 |
| M3 健康/进展 | 全新 venv 重放的[#53859 R3 bundle](../../results/vllm-zmq-backpressure-stage1-r3-20260916/)；[DFX 对照](../reviews/VLLM_53859_DFX_BASELINE_ABLATION_2026-09-24.md) | 一项已接纳请求停滞时，engine 仍存活且一次 health 探测答 2xx；存档主张可离线重放。 | 服务整体或逐 engine 的生产故障频率、进展触发器、零误报。 |
| M4 信号/生产者 | 同一 R3 重放与独立的[跨栈案例图](../FAILURES_THAT_NEVER_REACH_THE_SUPERVISOR.md) | R3 中栈与健康观察含义不同。c10d 案例另证 `producer_missing` 不等于 `member_missing`；这是方法经验，不是另一条 vLLM R3 观察。 | 连续 in-flight 信号覆盖或新 native 探针。 |
| M5 预算 | [K1/K2 CPU 结果](../reviews/ENGINE_LIVENESS_K1_K2_CPU_RUN_2026-09-27.md)；[K5](../reviews/ENGINE_LIVENESS_K5_DUAL_GPU_RESULT_2026-09-27.md) | K1/K2 测得指定 helper 的时间表；K5 只给 TP=1 有界 hold 下 health 保持 200 的正式分数。 | 已评分的 TP=2 超时归因、生产环境截止时间是否合适、CUDA 宽限默认值。 |
| M6 传播 | [#48966/#52178 成对结果](../../experiments/fault-recovery/README.md)及[DP Gate 0 边界](../../experiments/vllm-dp-supervisor-exit/README.md) | 固定配置的成对实验用顶层退出码区分致命 EngineCore 丢失与主动关闭。 | 上游已合并的修复，或把 DP Gate 0 升格为完整包结果。 |

## 私有台账校验器使用方式

校验器只读取冻结的[候选快照](../../data/pain-point-sample/candidate_snapshot_2026-09-28.json)及另行保管的私有 JSON 台账：

```text
python scripts/validate_pain_point_labels.py \
  --snapshot data/pain-point-sample/candidate_snapshot_2026-09-28.json \
  --ledger path/to/private-labels.json
```

台账顶层需有 `schema_version: "q4-label-ledger-v1"`、精确的 `snapshot_sha256` 与 `protocol_commit`、`in_progress`／`complete`／`no_sample` 状态及按冻结顺序排列的 `entries`。每个审读项记 issue 编号、按冻结词顺序的 `matched_terms`、标题与正文各自的 UTF-8 摘要、`updated_at`、首轮标注 UTC 时间、可选主动耗时，以及登记的排除代码，或纳入项的标签和证据指针。纳入项须为 V 标签和每个 M-part 记录证据等级（报告内日志/输出或报告者叙述）、ping 判断及依据、关闭方式、根因已知/未知。可选评论引用只存 ID、更新时间、摘要，不存正文；可选 `relabels` 保留前 10 个合格项的顺序、时间、各自的标题/正文摘要与更新时间及证据指针。原始标题、正文、评论、宿主路径、凭据和栈作为**额外字段**被拒绝；自由文本值不会自动脱敏。台账留在公开仓库之外，转移前另作内容审查。

这个工具只做结构校验，不能检查排除**优先级**、报告真伪、M3 三项事实是否充分、真正盲法或人工证据指针的来源。这些仍由冻结的[抽样协议](../PAIN_POINT_DISCOVERY_2026Q4.zh-CN.md)和[标注指南](../PAIN_POINT_M_PART_LABELING_GUIDE_2026Q4.zh-CN.md)裁决。本轮没有读候选正文，因此还没有 M-part 计数或覆盖结论。

交付后补记（2026-09-28）：原 v1 校验路径保持不变；另有[候选 v2 指南](../PAIN_POINT_M_PART_LABELING_GUIDE_V2_2026Q4.zh-CN.md)，增加 `fault_domain`、`downstream`、指南/标注者来源及可选人工复核。同一校验器仅在同时提供精确指南字节和原始私有 v1 账本时接受 v2，并绑定 v1 已读前缀；定向测试现为 25 项。本补记不改变冻结的 v1 指南，也不改写上文对这份证据检查最初写成时尚未读取候选正文的历史描述。
