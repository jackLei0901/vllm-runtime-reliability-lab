# Lab 运行时模型：serving 进程的存活状态

版本：**0.1.0**（2026-09-28 为 Q4 抽样标签冻结）。英文原稿：
[RUNTIME_MODEL.md](RUNTIME_MODEL.md)；[变更记录](CHANGELOG.md)；
[按模型部分检索](../INDEX.zh-CN.md)。

这是 Lab 的**描述性**模型，不是 vLLM 已认可的公共契约。主要源码固定在
vLLM [`c8602c79062440074a018c1d5f875a5571eb6881`](https://github.com/vllm-project/vllm/commit/c8602c79062440074a018c1d5f875a5571eb6881)；
[源码清单](../ENGINE_LIVENESS_CONTRACT_INVENTORY_2026-09-27.md)保留 S1–S11、
T1–T14 的详细锚点与证据等级。后续在
[`2407f405`](https://github.com/vllm-project/vllm/commit/2407f405b51abd23adbf0203b98464f448c58edf)
只重查了 C1 相关 Python/Rust 文件，并确认其 Git blob 与上一轮 `55de40a2fc` 相同；
**这不等于 M2–M6 的所有主张都已在最新 main 重审**。
各实验继续保留自己的源码版本、评分与限制。

模型只覆盖在线 serving 进程树，以及存活、可响应、有用工作和关闭之间的边界。
kernel、allocator、collective 默认是 worker 阶段下的不透明叶子；没有独立生产者就不推断更深层状态。
M-ID 只是 Lab 内部索引，不是 v0.2 字段名，也不是故障分类。

## M1 — 拓扑与观察所有权

| ID | 角色与状态持有者 | 源码与界限 |
| --- | --- | --- |
| M1.API | API server、`AsyncLLM` 持有面向客户端的 engine-dead 与 output-handler 状态。 | 固定版本 [`async_llm.py`](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/v1/engine/async_llm.py#L1086-L1096)。 |
| M1.CLIENT | `MPClient` 监控 EngineCore 进程 sentinel。 | [`core_client.py`](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/v1/engine/core_client.py#L708-L733)；不等于 headless CLI 父进程。 |
| M1.CORE | `EngineCoreProc` 持有 busy loop 与关闭状态。 | [`core.py`](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/v1/engine/core.py#L1378-L1389)。 |
| M1.EXECUTOR | uni/multiproc executor 的等待及检测契约不同；multiproc monitor 观察 worker sentinel。 | [`multiproc_executor.py`](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/v1/executor/multiproc_executor.py#L286-L305)、[`uniproc_executor.py`](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/v1/executor/uniproc_executor.py#L32-L34)。 |
| M1.WORKER | worker 持有关闭请求与设备工作；host 栈或标记不能独自证明设备进展。 | [`multiproc_executor.py`](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/v1/executor/multiproc_executor.py#L795-L841)。 |
| M1.PARENT | launcher/CLI 父进程控制外层进程树关闭；headless、多 API 与 `MPClient` 入口不同，Rust 又包装 headless Python。 | 当前 [headless](https://github.com/vllm-project/vllm/blob/2407f405b51abd23adbf0203b98464f448c58edf/vllm/entrypoints/cli/serve.py#L253-L260)、[多 API](https://github.com/vllm-project/vllm/blob/2407f405b51abd23adbf0203b98464f448c58edf/vllm/entrypoints/cli/serve.py#L398-L413)、[Rust 子进程](https://github.com/vllm-project/vllm/blob/2407f405b51abd23adbf0203b98464f448c58edf/rust/src/managed-engine/src/process.rs#L57-L75)。 |

DP supervisor、coordinator、Ray 与可选 FT 是不同拓扑变体，不能当作默认路径的同义词。
本版未覆盖多节点的完整行为。

## M2 — 生命周期轴

固定版本的 EngineCore 有实际的[关闭状态](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/v1/engine/core.py#L1378-L1389)
和[处理路径](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/v1/engine/core.py#L1460-L1504)。
模型按转换类别记录源码事实，而不宣称已有统一 enum：

| 转换类别 | 持有者或观察边界 | 状态 |
| --- | --- | --- |
| 启动 → 就绪 | client/EngineCore 握手与就绪等待 | 源码已观察；见 T1/T2 |
| 就绪 → 有意关闭 | 父进程发信号/请求，再由 EngineCore 与 executor 收尾 | 源码已观察；预算随入口不同（M5） |
| 就绪 → 终止故障 | 进程退出、EngineCore 异常或 worker 故障 | 源码已观察；传播随路径不同（M6） |
| 就绪 ↔ 有意暂停/休眠 | 公共控制可停止产出 token 而非产生故障 | 有源码控制，但不是通用共享状态 enum |

`draining`、`stopping`、`suspect` 等期望中的公共词汇仍属于
[契约提纲](../ENGINE_LIVENESS_RFC_OUTLINE_2026-09-27.zh-CN.md)，
不能倒灌进当前源码事实。有意暂停期间没有 token，本身不是健康故障。

## M3 — 健康与进展事实

| ID | 含义 | 此版本的生产者与限制 |
| --- | --- | --- |
| M3.ALIVE | 具名进程此刻存活 | 进程 sentinel 或外部 `/proc`；旧日志行不能证明现在仍存活。Lab 的普通 process snapshot 本身不保存 PID start ticks。 |
| M3.IDENTITY | 有界窗口内观察始终属于同一进程/engine | 需要单独保存 PID/start 身份和绑定；缺失时是 `unobserved`，不是 `engine_missing`。 |
| M3.LOOP | EngineCore loop 回应请求 | 默认 [`/health`](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/v1/engine/async_llm.py#L920-L923) 不提供此事实；[#36451](https://github.com/vllm-project/vllm/pull/36451) 尚是提案。 |
| M3.DEMAND | 工作已接纳且应前进 | 范围须为 request 或 engine；排除 idle、暂停、远端 KV 等待和 dummy batch。 |
| M3.PROGRESS | 有用工作在有界窗口内前进 | Lab `collect/verify` 可对有限窗口评分；recorder 当前不能由 S10 触发。求和后的指标标签不能证明某个 engine 前进。 |
| M3.UNOBSERVED | 必需生产者缺失、过期或冲突 | 这是证据状态，不是目标缺失或健康的证据。 |
| M3.TERMINAL | 归属明确的错误或进程退出确认故障 | [sentinel/output 路径](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/v1/engine/core_client.py#L708-L733)；不同于无进展。 |

M3 事实不是 verdict。`progress.py` 的五种封闭结果与优先级不改；
暂停/休眠、loop 响应仍是现有输入映射的缺口，不因此重评分旧实验。

## M4 — 信号与生产者结果

[清单 §2](../ENGINE_LIVENESS_CONTRACT_INVENTORY_2026-09-27.md)逐项列出 S1–S11。
模型对每项分开记录**条件、生产者、传输、消费者、可用状态**。
在固定版本，S1–S3/S6 能走到终止或健康路径，S4 是 multiproc RPC 时限，
S5/S10 没有对应生产者，S7 虽实现却未在服务中接线，S8/S9 只在日志，
S11 将 engine death 传播到 launcher 退出。
例如 worker 退出经 [executor sentinel/回调](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/v1/executor/multiproc_executor.py#L286-L305)
与 [EngineCore 故障处理](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/v1/engine/core.py#L1534-L1535)；
未接线的 [`check_health`](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/v1/executor/multiproc_executor.py#L505-L507)
不能称为服务健康信号。生产者不可用不能被评分为目标状态的否定观察。

## M5 — 时间与关闭预算

[清单 §3](../ENGINE_LIVENESS_CONTRACT_INVENTORY_2026-09-27.md)逐项固定 T1–T14；
本模型不会把它们合并为一个超时：

| 预算组 | 当前行为及关系 |
| --- | --- |
| 启动 T1–T3 | client 握手与 coordinator 等待各有 owner，不是服务进展窗口。 |
| 每步/健康 T4–T6 | multiproc RPC 默认 T4 为 300 秒；uniproc 没有等价时限；T5 未接线，T6 在固定版本没有 reader。 |
| 请求排空/父进程 T7–T8 | 显式 `shutdown_timeout=0` 不等于 `None`：Python helper 不给 5 秒 fallback，而是 SIGTERM 后对仍存活的子进程进入强杀。headless/多 API 信号路径也会传零。 |
| worker/传递 T9–T11 | 外层 `x` 与内层首次等 worker 的 `x` 相同，未给内层额外 4 秒 SIGTERM 升级和传递留预算；K2 用假进程树测了部分 `x`。 |
| 监控/FT T12–T14 | launcher 轮询、告警与可选 FT 恢复各有独立期限，不构成 EngineCore 进展预算。 |

当前重查的 [ROCm 例外](https://github.com/vllm-project/vllm/blob/2407f405b51abd23adbf0203b98464f448c58edf/vllm/v1/engine/utils.py#L46-L69)
仅在请求和进程预算都为零时给 EngineCore 额外 15 秒。这是源码事实和 CUDA 的 C1 设计问题，
不是 Lab 已决定的 CUDA 默认值。四条入口路径及信号/非信号差异见
[预算表](../ENGINE_LIVENESS_RFC_OUTLINE_2026-09-27.zh-CN.md)。

## M6 — 故障与关闭传播

| 边界 | 固定版本的现象 | Lab 证据边界 |
| --- | --- | --- |
| worker → EngineCore | multiproc worker 退出会成为 executor 故障；存活但卡住的 uniproc worker 没有对应时限。 | C3；[K5](../reviews/ENGINE_LIVENESS_K5_DUAL_GPU_RESULT_2026-09-27.md)只有 TP=1 正式评分。 |
| EngineCore → client/API | sentinel/错误输出可设置 `engine_dead` 并影响 `/health`；无进展 S10 不行。 | V1、C3/C4；默认无 loop ping。 |
| API/父进程 → supervisor | 退出码可能丢失 fatal cause，必须与有意 SIGTERM 区分。 | V2/[#52178](https://github.com/vllm-project/vllm/pull/52178)；不宣称 PR 已合并。 |
| DP supervisor → 调用方 | 子进程失败可能丢失在 supervisor 退出边界。 | V3 只有 Gate 0，完整 Gate 1 待做。 |
| 关闭 → 客户端 | 请求中止输出与父进程清理可以有不同预算。 | C1/C2/C8；[#36964](https://github.com/vllm-project/vllm/pull/36964) 已在处理输出中止。 |

K5 报错命名的是调用方正在等的 `sample_tokens` future，注入时 worker 却停在
`execute_model`；TP=2 臂仍按原规则未评分。这只是
[#54638](https://github.com/vllm-project/vllm/pull/54638) 的诊断例子，不是另立上游缺陷。

## 上游语汇适配层（仅文档）

| Lab 事实 | 上游词汇 | 不等同之处 |
| --- | --- | --- |
| M3.TERMINAL | `engine_dead`、`EngineDeadError`、可选 FT `DEAD` | 同词不代表 owner 与发生时间相同。 |
| 由 M3 推导的非终止疑似状态 | 可选 FT `UNHEALTHY` | 默认 serving 的设想，不等于当前 FT 范围。 |
| M2 关闭与 M3 可联系性 | [#36258](https://github.com/vllm-project/vllm/pull/36258) 中的 `/health`/`/live` | Lab 不改 endpoint。 |
| M3.LOOP | [#36451](https://github.com/vllm-project/vllm/pull/36451) 的 health ping | 提案中；loop 回应不等于有用工作前进。 |
| M5 排空/宽限 | [#24885](https://github.com/vllm-project/vllm/issues/24885) 与 ROCm [#52281](https://github.com/vllm-project/vllm/pull/52281) | 前者过期关闭，后者限于平台。 |

上游改名，只更新此适配层，不重构 M1–M6 或重评分历史结果。
新 issue 可以映射到已有 M 部分，或登记 `model_gap`；是否增加故障分类仍须通过原有准入门槛。
