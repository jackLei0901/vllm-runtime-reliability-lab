# M3/M4 生产者契约表 — 2026-09-28

状态：基于 `141843c` 的 Lab 源码盘点，不新增信号或 verdict。[English](PRODUCER_CONTRACT_TABLE_2026-09-28.md)。这是[执行计划](ARCHITECTURE_EXECUTION_2026-09-28.zh-CN.md)要求的有界生产者表；描述现有观测及进一步主张所需的证据，不修改已发布的 v0.2 schema。

| 生产者与入口 | 所有者、观测单位、身份绑定 | 时间与新鲜度 | 缺失、冲突或误导时的边界 |
| --- | --- | --- | --- |
| [`record` 的 `/health`](../../src/dfxlab/collectors.py)：`collect_health`、`CadencedCollector.collect` | vLLM 端点作答；Lab 采样操作员指定的一个 URL。它是端点观测，不是 EngineCore 循环或单个 engine 的身份。 | `ExternalObservation.observed_at` 与 `monotonic_ns` 记录 Lab 样本时间；`sampled_sources` 区分本次 HTTP 尝试与沿用的旧值。连接失败会记录 `ok=False`、`status=None` 及 collector 错误。 | HTTP 2xx 不证明有用工作前进；探测失败也不直接定位死亡 engine。[`recorder.classify`](../../src/dfxlab/recorder.py)在此前健康且连续不健康时可触发 `HEALTH_LOST`，但没有无进展触发器。 |
| [`record` 进程快照](../../src/dfxlab/collectors.py)：`process_snapshot` | Lab 对一个指定 PID 使用 `os.kill(pid, 0)` 或 Windows 进程查询；`/proc/<pid>/status` 额外给出内存/线程值。不保存 engine 标签或进程启动身份。 | 按节奏采样的时间与 `sampled_sources` 区分实际检查和沿用值。 | 无法检测 PID 复用。`tracked=False` 表示未指定目标；缺少进程观测不等于 engine 不存在。 |
| [`record` 队列/压力指标](../../src/dfxlab/collectors.py)：`collect_metrics` 和 [`select_metrics`](../../src/dfxlab/prometheus.py) | vLLM 输出指标；Lab 保存 KV 使用率、抢占次数、running/waiting 数。`select_metrics` **汇总各 label 变体**，单位是所选服务总量，而不是某个 engine。 | 成功 scrape 是新 HTTP 观测；未到采样时间的样本携带上次值，不得称为新值；采集错误单独记录。 | `METRIC_FIELDS` 没有生成 token 计数。一个 engine 不动而其他 engine 继续前进时，总量会遮蔽它。队列指标与 `/health` 均不能授权 engine 级进展结论。 |
| [`collect/verify` 服务计数与需求](../../src/dfxlab/collect_bundle.py)：`_metrics`、`collect_bundle`，及 [`progress.py`](../../src/dfxlab/progress.py) | vLLM `/metrics` 输出生成 token 及 running/waiting 序列；Lab 选择操作员声明的一个服务端点。当前解析器汇总 label 变体，因此这是**服务范围**生产者。 | 成功采样带 `perf_counter_ns` 和 `fresh=True`。验证器要求足量新样本与跨度、连续需求、计数器不倒退。失败或缺失字段不会变成新值。 | 服务总量不变且有已接纳工作，须连同 verdict 的进程与健康门槛才能支持有界服务结论，仍不能定位单个 engine。label 集变化或某 engine 被其他 engine 的进展掩盖，现有逻辑均无法解决。`producer_missing` 不等于目标不存在。 |
| [`collect/verify` 进程与端点](../../src/dfxlab/collect_bundle.py)：`_process_identity`、`_health`、`collect_bundle` | 操作员提供 PID 与 URL。Linux 上 `/proc/<pid>/stat` 启动 ticks 把连续检查绑定到同一进程实例；PID 与端点之间仍是操作员声明。 | 样本的单调时间和启动身份覆盖有界窗口；健康尝试记录状态/错误。[`derive_verdict`](../../src/dfxlab/progress.py)让进程丢失、健康丢失优先。 | 身份缺失或变化阻断“仍存活”的结论。端点 2xx 证明端点响应，不证明 EngineCore 循环响应。指定 PID 和 URL 并不是独立验证的 engine–端点 join。 |
| [`collect/verify` 客户端内容探针](../../src/dfxlab/collect_bundle.py)：流式请求及 [`evaluate_producer`](../../src/dfxlab/progress.py) | Lab 发一个指定请求并观测内容 chunk；身份是请求摘要和操作员指定端点。单位是该请求，不是服务或 engine。 | chunk 与请求边界使用 `perf_counter_ns`。只有 role/usage/empty chunk 不算有用进展；整个窗口内未完成且无内容的请求可判 flat。 | 请求已结束、stream 不可用或请求未覆盖整个窗口，都不能得出无进展结论。请求级观测不能暗中替换服务级计数器。 |

## 尚无生产者的事实与下一准入门

- **循环响应：**默认 `/health` 不是 EngineCore ping。[#36451](https://github.com/vllm-project/vllm/pull/36451)仍是提案，不是 Lab 当前可用生产者。即便 ping 成功，它只证明循环存活，不证明 token 前进。
- **单 engine 进展：**[V1 接入门](../reviews/V1_RECORDER_VERDICT_INTEGRATION_GATE_2026-09-25.md)要求保留各指标 label 分区、同次 scrape 的计数与需求、新鲜度，以及稳定的标签到 PID/启动身份绑定。[G0 CPU 门](../../experiments/vllm-engine-binding-gate/LOCAL_CPU_GATE_2026-09-25.md)验证 fail-closed join 规则，不是现场绑定。本表不批准单 engine recorder 触发器。
- **生命周期状态：**pause/sleep、等待远端 KV 均未进入已发布 verdict 的生产者输入。它们是未来进展感知健康提案的负对照，而不是重解释历史 bundle 的理由。

本表是基于源码的能力映射，不是工具漏诊率测量；`external-runtime-observation-v1` 字段及 v0.2 verdict 优先级均未变。
