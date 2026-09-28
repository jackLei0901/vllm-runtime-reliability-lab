# Lab runtime-model index

[中文版索引](INDEX.zh-CN.md)

This index organizes existing evidence by the Lab's descriptive
[runtime model v0.1.0](model/RUNTIME_MODEL.md)
([中文](model/RUNTIME_MODEL.zh-CN.md)). It is not a replacement for the
[fault taxonomy](FAULT_TAXONOMY_V0_2026-09-24.md), the frozen v0.2 verdicts,
or an upstream vLLM contract. Each experiment's README has a small status
header; `last_scored: not-indexed` means this index has **not** verified a
latest scored result, not that no result exists. `model_cells: []` means the
experiment is outside M1–M6 or concerns the Lab evidence layer.

| Model part | Source map and contract question | Experiment entry points |
| --- | --- | --- |
| **M1 — process topology** | [Liveness inventory](ENGINE_LIVENESS_CONTRACT_INVENTORY_2026-09-27.md); [architecture map](model/ARCHITECTURE_MAPPING_2026-09-27.md) | [Engine binding](../experiments/vllm-engine-binding-gate/README.md), [DP supervisor](../experiments/vllm-dp-supervisor-exit/README.md), [K5](../experiments/engine-liveness-contract/README.md) |
| **M2 — lifecycle** | [Liveness inventory](ENGINE_LIVENESS_CONTRACT_INVENTORY_2026-09-27.md); [proposed contract, not current semantics](ENGINE_LIVENESS_RFC_OUTLINE_2026-09-27.md) | [K1/K2/K5](../experiments/engine-liveness-contract/README.md); paused/sleeping remain mapping gaps in frozen verdict inputs |
| **M3 — health and progress facts** | [Taxonomy](FAULT_TAXONOMY_V0_2026-09-24.md); [v0.2 field roles](V0.2_FIELD_ROLES.md) | [ZMQ backpressure](../experiments/vllm-zmq-event-backpressure/README.md), [engine binding](../experiments/vllm-engine-binding-gate/README.md), [organic hang](../experiments/organic-hang/README.md) |
| **M4 — signal path** | [Liveness inventory S1–S11](ENGINE_LIVENESS_CONTRACT_INVENTORY_2026-09-27.md); [native evidence design](NATIVE_EVIDENCE_DESIGN.md) | [ZMQ backpressure](../experiments/vllm-zmq-event-backpressure/README.md), [native evidence](../experiments/native-evidence-capability/README.md), [cross-stack c10d](../experiments/pytorch-c10d-shutdown-dump/README.md) |
| **M5 — budgets** | [Liveness inventory T1–T14](ENGINE_LIVENESS_CONTRACT_INVENTORY_2026-09-27.md); [K1/K2/K5 review](reviews/ENGINE_LIVENESS_UNIFIED_REVIEW_2026-09-27.md) | [Engine liveness checks](../experiments/engine-liveness-contract/README.md) |
| **M6 — failure propagation** | [Failure case study](FAILURES_THAT_NEVER_REACH_THE_SUPERVISOR.md); [liveness inventory](ENGINE_LIVENESS_CONTRACT_INVENTORY_2026-09-27.md) | [Fault recovery](../experiments/fault-recovery/README.md), [DP supervisor](../experiments/vllm-dp-supervisor-exit/README.md) |

The [full migration map](model/ARCHITECTURE_MAPPING_2026-09-27.md) also
accounts for modules and documents that are not runtime-model facts. The
following are intentionally **outside** M1–M6: [TP collective capability](../experiments/vllm-tp-dfx/README.md),
[multimodal cache correctness](../experiments/vllm-mm-uuid-encoder-cache/README.md),
[FSDP2 dtype](../experiments/pytorch-unused-grad-dtype/README.md), and the
[overhead harness](../experiments/overhead/README.md). The alpha-era
[OOM](../experiments/oom-boundary/README.md),
[preemption](../experiments/preemption/README.md), and
[soak](../experiments/soak/README.md) stubs are archived **in place**, not
deleted or counted as scored evidence.

Incoming issues first map to a model part or a private `model_gap` entry;
neither action changes the taxonomy by itself. The
[Q4 discovery protocol](PAIN_POINT_DISCOVERY_2026Q4.zh-CN.md) fixes the
sampling and labeling rules before any issue body is read.
