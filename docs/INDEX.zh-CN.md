# Lab 运行时模型索引

[English index](INDEX.md) · [运行时模型](model/RUNTIME_MODEL.zh-CN.md)

本索引按 M1–M6 组织已有材料，不替代[故障分类](FAULT_TAXONOMY_V0_2026-09-24.md)、
已冻结的 v0.2 verdict 或上游 vLLM 契约。每个实验 README 顶部记录状态；
`last_scored: not-indexed` 表示**此索引尚未核定最新评分材料**，不是说实验没有结果。
`model_cells: []` 表示模型外工作或 Lab 的证据层工作。

| 模型部分 | 源码和契约材料 | 实验入口 |
| --- | --- | --- |
| **M1 进程拓扑** | [存活契约清单](ENGINE_LIVENESS_CONTRACT_INVENTORY_2026-09-27.md)；[完整迁移表](model/ARCHITECTURE_MAPPING_2026-09-27.md) | [引擎身份绑定](../experiments/vllm-engine-binding-gate/README.md)、[DP supervisor](../experiments/vllm-dp-supervisor-exit/README.md)、[K5](../experiments/engine-liveness-contract/README.md) |
| **M2 生命周期** | [存活契约清单](ENGINE_LIVENESS_CONTRACT_INVENTORY_2026-09-27.md)；[提议中的契约](ENGINE_LIVENESS_RFC_OUTLINE_2026-09-27.zh-CN.md)，不等于当前实现 | [K1/K2/K5](../experiments/engine-liveness-contract/README.md)；暂停／休眠仍是已冻结 verdict 输入的映射缺口 |
| **M3 健康与进展事实** | [故障分类](FAULT_TAXONOMY_V0_2026-09-24.md)；[v0.2 字段角色](V0.2_FIELD_ROLES.md) | [ZMQ 反压](../experiments/vllm-zmq-event-backpressure/README.md)、[引擎身份绑定](../experiments/vllm-engine-binding-gate/README.md)、[organic hang](../experiments/organic-hang/README.md) |
| **M4 信号路径** | [清单 S1–S11](ENGINE_LIVENESS_CONTRACT_INVENTORY_2026-09-27.md)；[native evidence 设计](NATIVE_EVIDENCE_DESIGN.md) | [ZMQ 反压](../experiments/vllm-zmq-event-backpressure/README.md)、[native evidence](../experiments/native-evidence-capability/README.md)、[跨栈 c10d](../experiments/pytorch-c10d-shutdown-dump/README.md) |
| **M5 时间预算** | [清单 T1–T14](ENGINE_LIVENESS_CONTRACT_INVENTORY_2026-09-27.md)；[K1/K2/K5 统一审阅](reviews/ENGINE_LIVENESS_UNIFIED_REVIEW_2026-09-27.md) | [Engine liveness 检查](../experiments/engine-liveness-contract/README.md) |
| **M6 失败传播** | [失败案例](FAILURES_THAT_NEVER_REACH_THE_SUPERVISOR.zh-CN.md)；[存活契约清单](ENGINE_LIVENESS_CONTRACT_INVENTORY_2026-09-27.md) | [Fault recovery](../experiments/fault-recovery/README.md)、[DP supervisor](../experiments/vllm-dp-supervisor-exit/README.md) |

[完整迁移表](model/ARCHITECTURE_MAPPING_2026-09-27.md)也记录模型外的模块和文档。
[TP collective](../experiments/vllm-tp-dfx/README.md)、
[多模态 cache 正确性](../experiments/vllm-mm-uuid-encoder-cache/README.md)、
[FSDP2 dtype](../experiments/pytorch-unused-grad-dtype/README.md)和
[overhead harness](../experiments/overhead/README.md)均不强行归入 M1–M6。
早期的 [OOM](../experiments/oom-boundary/README.md)、
[preemption](../experiments/preemption/README.md)与
[soak](../experiments/soak/README.md) stub 已**原地归档**，未删除，也不算评分证据。

新问题先对应已有模型部分，或进入私有 `model_gap` 台账；这两种处理都不自动增加
故障类别。[Q4 发现协议](PAIN_POINT_DISCOVERY_2026Q4.zh-CN.md)要求读取 issue 正文前
固定抽样与标注规则。
2026-09-28 的[复核流程修订 A1](model/Q4_SAMPLE_REVIEW_AMENDMENT_2026-09-28.zh-CN.md)
将先行人工盲标改为抽样后的用户统一审核；v2.0.0 判定规则与冻结候选顺序不变。
