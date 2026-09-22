# vLLM Runtime Reliability Lab 中文指南

> 当前版本：已发布 [`v0.2.0`](https://github.com/jackLei0901/vllm-runtime-reliability-lab/releases/tag/v0.2.0)
> 当前定位：可公开安装和复现的研究型 Alpha，不是生产监控产品。

相关文档：

- [`DESIGN.zh-CN.md`](DESIGN.zh-CN.md)：详细方案设计与产品候选架构；
- [`PRODUCT_ROADMAP.zh-CN.md`](PRODUCT_ROADMAP.zh-CN.md)：evidence-first 阶段计划和验收条件；
- [`PRIOR_ART_AND_VALUE.zh-CN.md`](PRIOR_ART_AND_VALUE.zh-CN.md)：关联证据的公开先例、实际收益和适用边界；
- [`REQUIREMENTS.md`](REQUIREMENTS.md)：当前英文需求基线；
- [`TEST_PLAN.md`](TEST_PLAN.md)：当前英文测试计划。
- [`docs/V0.2_FIELD_ROLES.md`](docs/V0.2_FIELD_ROLES.md)：机器校验的公开字段角色审计。
- [`docs/V0.2_LAUNCH_POST.md`](docs/V0.2_LAUNCH_POST.md) 与
  [`docs/V0.2_LAUNCH_POST.zh-CN.md`](docs/V0.2_LAUNCH_POST.zh-CN.md)：可直接发布的
  中英文 v0.2 技术入口。
- [`docs/FAILURES_THAT_NEVER_REACH_THE_SUPERVISOR.md`](docs/FAILURES_THAT_NEVER_REACH_THE_SUPERVISOR.md)
  与[中文版](docs/FAILURES_THAT_NEVER_REACH_THE_SUPERVISOR.zh-CN.md)：四类故障信号在
  进程、rank、health 或诊断边界上丢失语义的综合文章。
- [`docs/EVIDENCE_TO_CLAIM_BLOCK4.md`](docs/EVIDENCE_TO_CLAIM_BLOCK4.md)：
  #196968、#53859/#53883 与 #196996 的 evidence-to-claim、禁止推断、印证和反证表。
- [`docs/LOW_LEVEL_CAPABILITY_REQUIREMENTS.md`](docs/LOW_LEVEL_CAPABILITY_REQUIREMENTS.md)
  与 [`docs/NATIVE_EVIDENCE_DESIGN.md`](docs/NATIVE_EVIDENCE_DESIGN.md)：不改变 v0.2
  verdict 的底层能力需求与 sidecar-first native evidence 设计。
- [`docs/reviews/EVIDENCE_TO_CLAIM_BLOCK4_REVIEW.md`](docs/reviews/EVIDENCE_TO_CLAIM_BLOCK4_REVIEW.md)：
  Block 4 统一中英 review 入口。

## 先看结果

vLLM Runtime Reliability Lab 是面向推理引擎故障的证据实验室。当服务进程仍然
存活、健康检查仍然成功，但跨进程或跨 rank 的推理已经停止时，它用外部观测、
受控故障注入和跨主体证据规则，把“卡住了”转化为可复现、可验证、能推动
upstream 修复的结论。它不是监控平台，也不是 issue 收集仓库。

| 外部症状 | Lab 得出的结论 | 外部结果与边界 |
| --- | --- | --- |
| Flight Recorder 有 rank 0 dump，却没有 rank 1 dump | `producer missing != member missing`：rank 1 仍存活并卡在 `destroy_process_group()`，只是诊断生产者已停止响应 | Lab 发现并提交 [PyTorch #196968](https://github.com/pytorch/pytorch/issues/196968)，形成 C++ 修复 [#197232](https://github.com/pytorch/pytorch/pull/197232)；截至 2026-09-21 两者仍 open |
| vLLM 进程存活、`/health` 返回 2xx，但 token 停止推进 | 确定性满队列使真实 EngineCore 阻塞在 `ZmqEventPublisher.publish()`；修复通过丢弃事件 batch 恢复活性 | 独立验证已有报告 [vLLM #53859](https://github.com/vllm-project/vllm/issues/53859) 和修复 [#53883](https://github.com/vllm-project/vllm/pull/53883)，不是 Lab 首次发现；截至 2026-09-21 两者仍 open |
| torchtitan 表现为分布式 hang | 多轮 gate 去除分布式表象后，在单卡复现 FSDP2 mixed-gradient-dtype assertion | Lab 发现并提交 [PyTorch #196996](https://github.com/pytorch/pytorch/issues/196996)；截至 2026-09-21 已 triage、仍 open |

三条最重要的证据规则是：

```text
alive            != making progress
health green     != serving healthy
producer missing != participant missing
```

`#49869` 是独立的 upstream 成果，不属于 Lab 发现；`#52178` 是 Lab 提供系统级
验证的独立 lifecycle 修复。

四个案例的共同结构见[《没有到达 supervisor 的故障》](docs/FAILURES_THAT_NEVER_REACH_THE_SUPERVISOR.zh-CN.md)。

### 五分钟无 GPU replay

下面的命令不会重新证明 GPU 结果，而是对已发布的 #53859 四 cell 证据执行
fail-closed 重放：校验封闭文件集合、SHA-256、base/fix 身份、progress、health、
stack 结论和事件丢失代价。

```bash
python -m pip install .  # 安装 distribution vllm-runtime-dfx-lab
vllm-dfx replay results/vllm-zmq-backpressure-stage1-r3-20260916
```

关键输出：

```text
PASS: EngineCore process remained alive during the injected stall
PASS: /health remained 2xx while token progress stopped
STACK: EngineCore -> ZmqEventPublisher.publish -> Queue.put
FIX ARM: token progress completed under the same trigger
TRADE-OFF: 4 event batches dropped
```

任何 evidence hash、record shape、cell identity 或跨 cell identity 改变都会让
replay 失败关闭。独立运行后可用
[replay report 模板](https://github.com/jackLei0901/vllm-runtime-reliability-lab/issues/new?template=replay-report.yml)
报告平台、命令和结果。
如需从全新环境复制执行，请使用
[五分钟外部试用入口](docs/V0.2_EXTERNAL_TRIAL.md)。

也可以把同一套 evidence rule 用在本地 incident 上，不需要 GPU：

```bash
vllm-dfx collect \
  --base-url http://127.0.0.1:8000 \
  --pid 12345 \
  --window 60 \
  --no-progress-window 10 \
  --output incident/
vllm-dfx verify incident/
```

`verify` 完全离线，并从 `observations.json` 重新计算 producer、demand、conflict
和 verdict。公开 bundle 不保留 base URL、prompt、response content、header、原始
stack 或完整 sampler command。endpoint-only、PID-only、client probe、stack 和
信任边界详见 [v0.2 collect/verify 合同](docs/COLLECT_VERIFY_V0_2.md)。

## 1. 这个项目解决什么问题

当前最值得解决的不是“缺少更多日志”，而是两类健康语义失真：

- EngineCore 已经死亡，但 serving process 以 0 退出，`Restart=on-failure`
  不会重启（[vLLM #48966](https://github.com/vllm-project/vllm/issues/48966)）；
- 请求不再产生输出，但进程和 `/health` 仍然存活，编排系统不会摘除实例
  （[vLLM #52319](https://github.com/vllm-project/vllm/issues/52319)）。

最终日志通常只能说明进程在哪里停止，却不一定保留故障前一段时间内的
运行轨迹，例如：

- KV cache 使用率是否持续升高；
- 等待请求是否开始堆积；
- 是否发生大量抢占；
- `/health` 从成功变成失败的时间；
- 被观察进程和 GPU 显存状态如何变化。

本项目提供一个运行在 vLLM 部署环境、但位于 vLLM 进程之外的轻量 recorder。
它按不同周期采集允许
公开的聚合信号，将最近 N 个样本保存在固定大小的内存队列中，并在观察到
进程退出、健康检查丢失、KV 压力或抢占突增时写出一个有大小上限的事件文件。

它的目标不是自动判断根因，而是把故障检测、证据保留和证据边界变成可执行、
可验证的契约。v0.2 已能在有界窗口内判断 `/health=200` 时是否存在有需求但无
进展；哪个 rank 首先出现分歧以及通用跨 rank join 仍属于后续范围。

项目的验收标准不是“是否采到了数据”，而是证据能否补上实际运维缺口：

| 运维缺口 | 产品输出 | 应支持的动作 |
| --- | --- | --- |
| `/health=200`，但已有请求停止推进 | 带支撑证据的、有界 `alive_health_ok_no_progress` verdict | 进入排查、摘流或重启流程，而不是让静默故障继续保持健康 |
| 单个 rank 卡住或消失，其他进程只有局部状态 | 经校验的 producer 集合、缺失 peer、状态分歧和顺序边界 | 不依赖故障时 collective，缩小第一个应排查的故障域 |
| 多份文件无法证明属于同一次事故 | 关闭的 manifest、身份、clock 和内容 hash | 在诊断前拒绝混入其他运行或被篡改的证据 |

v0.2 已交付有界本地 `collect` 与离线 `verify`：选择一个 decision producer，并可
保留一个 corroborating producer，但不合并二者的 scope。通用跨 rank、跨主机
关联仍未交付。

### 1.1 可选 stack producer 有严格部署门槛

v0.2 可按需调用一个有界、显式启用的 `py-spy` stack producer。原始 stack 只保留
在私有目录，不能改变 verdict；公开 bundle 只保留 typed availability 和有界的
producer identity。`py-spy` 需要读取另一个进程的内存：Linux attach 通常需要
root 或调整 `ptrace_scope`，Docker/Kubernetes 往往需要
`SYS_PTRACE`。默认采样可能短暂停顿目标进程；`--nonblocking` 可以避免暂停，但
由于多次内存读取不是原子的，可能得到错误或不完整 stack。具体限制见
[py-spy FAQ](https://github.com/benfred/py-spy#frequently-asked-questions)。

当前 adapter 不公开任意 frame、不分类 native state，也不跨 rank join stack。
这些能力必须另做固定耗时和样本预算的 go/no-go 实验。权限拒绝、超时或 partial
output 都是正常的明确结果，不能被当作 recorder 异常。

### 1.2 为什么只看持续监控还不够

vLLM 已经内置 Prometheus 指标（`vllm/v1/metrics/prometheus.py`）和
OpenTelemetry trace（`vllm/tracing/otel.py`）。本项目不替代其中任何一个，并且
把同一个 `/metrics` 端点作为自己的输入之一。它解决的是默认持续指标难以可靠
保留的三类故障现场，而不是时间序列数据库完全无法表达的数据。

1. **最后一个采样间隔可能缺失。** Prometheus 按固定间隔拉取。进程一旦死亡就
   不再响应，因此最后一次成功 scrape 到故障之间的活动可能没有进入时序存储。
   本 recorder 在本地维护有界历史，并在外部条件触发时冻结。它没有消除采样
   上限，但能在远端接口消失时保留观察边界上已经取得的样本。
2. **诊断所需的数据是高基数的。** 判断 rank 在哪里发生分歧，可能需要
   per-rank、per-collective 的 sequence number、input shape、dtype 和 stack
   frame。把它们作为持续指标标签输出会产生明显的基数成本。一次性 artifact
   避免的是持续基数成本，但仍受本项目 256 KiB 文件上限约束。
3. **答案是一种关系，不是一个数值。** “rank 1 在本地抛出异常，而 rank 0 仍停在
   collective 中”是关于两份独立记录在同一逻辑位置上对齐的陈述。per-target
   指标本身不包含这种关系；当采样周期和时钟不确定性与事件间隔相近时，也不能
   把 wall clock 当成可靠的全局顺序。因此关联设计优先使用 collective 的逻辑
   位置，而不是按时间戳猜测先后。

本仓库内的边界实验说明了这个缺口。Phase 2 Gate 1 的两 rank 任务触发了 60 秒
wall timeout，但旧 runner 丢弃了逐 rank 输出，因此无法判断是一个 rank 先在本地
assert、另一个随后等待，还是两个 rank 同时停滞。源码复核发现 Gate 1b 把预期
CPU 等待点放错了位置；Gate 1c 的 wall bound 过紧，且停止规则把机制门与终止门
错误地绑在一起，两者均已在执行前撤回。Gate 1d 随后因 runner marker 解析缺陷停止。
Gate 1e 修复该缺陷后，三次复现 rank-local assertion 与 peer wait，并取得两个 rank
的 stack；但每次只有 rank 0 产生 Flight Recorder dump，因此严格关联门按规则失败。
Gate 1f 随后按冻结协议执行了一次 affected trial，保留八项白名单逐 rank shutdown
阶段布尔值。rank 0 成功广播 dump 请求并写出本地 dump；rank 1 已停止 heartbeat
monitor、进入 communicator destruction，但既未完成 destroy，也未观察到请求。
诊断门通过，Gate 1e 的严格关联门仍保持失败关闭。

反向情形同样重要。在一例 health-green 停滞报告
（[vLLM #52319](https://github.com/vllm-project/vllm/issues/52319)）中，`/health`
和 `/metrics` 仍返回 HTTP 200，但生成吞吐降为零，等待请求继续累积。指标可以
发现服务不再推进，却不能单独指出进度停止在哪个内部 rank 或执行位置。

如果某个部署已经保留了同等的 per-rank、触发时数据，并能在故障时可靠打包成
可分享证据，那么本项目没有增量价值。计划中的 Prometheus 对照和
unlinked-versus-linked 消融是明确的产品验收项，不是前提。

### 1.3 v0.2 与后续目标

v0.2 已完成 bounded `collect`、离线 `verify`、server/client progress producer、
demand gate 和无 GPU replay。它尚未实现：

- 持续自治监控；
- API server、EngineCore 和 worker/rank 自动发现；
- 通用跨 rank 或跨主机 semantic join；
- native-state 分类；
- 自动摘流、重启或根因分类。

这些属于 v0.2 之后的实验方向，不能作为当前发布能力宣传。

## 2. 当前完成度

### 已经完成

| 能力 | 当前状态 |
| --- | --- |
| 公开源码、MIT License、版本化 Release | 已完成 |
| Python 3.10、3.12、3.13 CI | 已完成 |
| 无第三方运行时依赖 | 已完成 |
| 固定长度内存历史 | 已完成 |
| `/health`、精选 Prometheus 指标、PID、GPU 聚合采集 | 已完成 |
| 关闭的 JSON Schema | 已完成 |
| prompt、token、请求 ID、路径和任意配置不进入公开文件 | 已完成 |
| 256 KiB 文件上限、最多保留四份、POSIX `0600` | 已完成 |
| 写入失败不影响被观察服务 | 已完成并经过 GPU 冒烟验证 |
| bounded `collect` 与离线 fail-closed `verify` | v0.2 已完成 |
| server/client progress producer 与 demand-gated verdict | v0.2 已完成 |
| 无 GPU 的 `vllm-dfx replay` | v0.2 已完成 |
| 可选私有 `py-spy` capture 与公开 typed producer status | v0.2 已完成 |
| 正常 SIGTERM、EngineCore SIGKILL、受控 CUDA OOM | RTX 4090 已验证 |
| 四卡 FSDP2 已知答案的 collective divergence 重建 | 已完成；仍缺同版本负对照 |
| 两卡 unused-gradient dtype 机制 Gate 0 | 已完成；Gate 1e 机制 3/3 通过 |
| vLLM #53859 EngineCore 背压与 #53883 对照 | 单卡四 cell 通过；修复以事件丢失换取活性 |
| 需求到测试用例的机器检查 | 已完成 |

### 尚未完成

| 产品化缺口 | 为什么重要 |
| --- | --- |
| recorder 开关配对的性能实验 | 尚不能给出 CPU、RSS、TTFT、TPOT 和吞吐开销上界 |
| 新一轮真实 KV pressure/preemption 实验 | 目前只有触发逻辑测试和历史实验，缺少 alpha.3 的完整实测 |
| Gate 1e 严格双 rank Flight Recorder join | 每次缺 rank 1 dump，0/3，FAIL-CLOSED |
| vLLM TP=2 stall 和跨节点验证 | FSDP2/c10d 结果不能外推到 vLLM 热路径或多节点 |
| 多版本兼容矩阵 | 尚未覆盖多个 vLLM release、指标名变化和不同 GPU 架构 |
| 长时间运行与 crash-loop 测试 | 短期冒烟不能证明数天运行时的资源稳定性 |
| systemd、容器和 Kubernetes 部署模板 | 当前仍以命令行实验工具为主 |
| 配置校验、升级和 schema 迁移策略 | 正式产品需要稳定的运维契约 |
| 真实用户采用与诊断收益 | 文件能生成不等于它确实缩短了故障定位时间 |

因此，项目已经达到“公开 Alpha”的标准，但没有达到“生产可用正式产品”
的标准。

## 3. 设计边界

```text
vLLM /health + /metrics ---|
显式指定的 PID -----------|--> 分周期采集器
nvidia-smi ----------------|         |
                                     v
                              ExternalObservation
                                     |
                                有界 deque
                                     |
                              外部触发条件
                                     |
                                     v
                          IncidentWriter -> JSON
```

recorder 是独立进程，只能观察外部信号。它看不到 EngineCore 中的异常对象，
也无法知道异常究竟发生在调度、模型执行还是通信阶段。

这个限制不仅写在文档中，还写进了 schema：

```json
{
  "internal_kind": "unknown",
  "internal_stage": "unknown"
}
```

两个字段只能取 `unknown`。外部 artifact 如果声称知道内部异常类型或执行
阶段，会直接校验失败。这能防止后续代码通过时间间隔或日志字符串，把推测
包装成事实。

## 4. 五分钟本地演示

需要 Python 3.10 或更高版本。

```bash
git clone https://github.com/jackLei0901/vllm-runtime-reliability-lab.git
cd vllm-runtime-reliability-lab
python -m pip install -e .
python examples/fake_vllm_server.py --port 18000
```

在第二个终端运行：

```bash
vllm-dfx record \
  --base-url http://127.0.0.1:18000 \
  --output demo-output \
  --history 16 \
  --duration 5
```

健康运行不会凭空生成事故文件。要验证 `health_lost` 路径，可以让 recorder
保持运行，然后停止 fake server：

```bash
vllm-dfx record \
  --base-url http://127.0.0.1:18000 \
  --output demo-output \
  --history 16 \
  --unhealthy-samples 2 \
  --stop-on-incident
```

生成 Markdown 摘要：

```bash
vllm-dfx summarize \
  --input demo-output/incident-<具体文件名>.json \
  --output demo-output/incident-summary.md
```

PowerShell 和 Windows CMD 不一定按 POSIX shell 的方式展开 `*`，因此建议
传入具体文件名。

## 5. 连接真实 vLLM 服务

先等待 vLLM 完成启动，再显式传入 API server PID：

```bash
vllm-dfx snapshot-env \
  --output private-run/environment.private.json

vllm-dfx record \
  --base-url http://127.0.0.1:8000 \
  --pid "$API_SERVER_PID" \
  --target-vllm-version "<服务端准确版本或 commit>" \
  --target-torch-version "<服务端准确 Torch 版本>" \
  --output shareable-incidents \
  --sample-interval 1 \
  --gpu-interval 5 \
  --history 300
```

recorder 可能与 vLLM 运行在不同 Python 环境中，所以它不会读取自身环境中
安装的 vLLM/Torch 版本来冒充服务端版本。未提供准确值时，对应字段保持
`null`。

不要用模糊的命令行搜索自动选择 PID。应先检查进程树，再把目标 PID 明确
传给 recorder。

## 6. 主要参数

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `--sample-interval` | 1 秒 | recorder 主循环间隔 |
| `--health-interval` | 1 秒 | `/health` 采样周期 |
| `--metrics-interval` | 1 秒 | `/metrics` 采样周期 |
| `--process-interval` | 1 秒 | PID 状态采样周期 |
| `--gpu-interval` | 5 秒 | `nvidia-smi` 采样周期 |
| `--history` | 300 | 内存中最多保留的样本数 |
| `--timeout` | 1 秒 | HTTP 采集超时 |
| `--kv-threshold` | 0.95 | KV cache 压力触发阈值 |
| `--unhealthy-samples` | 3 | 连续失败多少次后触发健康丢失 |
| `--preemption-delta` | 20 | 抢占计数增量阈值 |
| `--incident-cooldown` | 60 秒 | 同类事件两次写入之间的冷却时间 |
| `--max-artifact-kib` | 256 | 单个公开 artifact 的大小上限 |
| `--max-artifacts` | 4 | 输出目录中保留的完成文件数量 |
| `--stop-on-incident` | 关闭 | 首次触发后退出 recorder |

`nvidia-smi` 的启动成本明显高于 HTTP 和 `/proc` 读取，因此 GPU 默认采样
周期更长。调短它之前应先测 recorder 自身开销。

## 7. 触发条件如何理解

支持四种外部触发：

- `process_exit`：显式跟踪的 PID 已退出；
- `health_lost`：曾经成功的健康检查连续失败；
- `kv_pressure`：公开指标中的 KV cache 使用率超过阈值；
- `preemption_storm`：两次有效采样之间的抢占计数增量超过阈值。

这些名称描述的是外部现象，不是根因。例如 `health_lost` 可能来自正常退出、
EngineCore 崩溃、CUDA OOM 或其他故障。需要结合服务日志、退出码、编排器事件
和其他内部证据继续判断。

## 8. 哪些文件可以分享

### 按公开契约生成

- `incident-*.json`

它使用关闭的字段白名单，并排除 prompt、token ID、请求 ID、模型路径、环境
变量、命令行、traceback 和任意配置。

### 默认按私有实验材料处理

- `snapshot-env` 输出；
- `run-summary.private.json`；
- signal injection 记录；
- 使用 `--private-raw-timeline` 生成的时间线；
- vLLM 服务日志和负载输出。

即使文件名标为 shareable，也应在公开前再次检查。字段白名单不是通用 DLP
系统，不能代替组织自身的数据合规流程。

## 9. 故障注入

`inject-signal` 默认建议先使用 `--dry-run`：

```bash
vllm-dfx inject-signal \
  --pid "$ENGINE_CORE_PID" \
  --signal SIGKILL \
  --event-log private-run/injections.jsonl \
  --dry-run
```

只有在隔离测试环境中、并且确认允许终止目标进程时，才能去掉
`--dry-run`。源码中的 CUDA OOM 注入 patch 也只供隔离实验使用，不能进入
生产 vLLM 构建。

## 10. 已完成的验证

alpha.2/alpha.3 共享同一运行时和 artifact 契约。RTX 4090 验证包括：

- 正常 `SIGTERM` 对照；
- 三次 EngineCore `SIGKILL`；
- 三次真实 `execute_model` 调用边界上的受控 CUDA OOM；
- writer 输出目录不可用；
- recorder 完全关闭的对照。

六次有效致命故障试次均未留下 vLLM 孤儿进程。外部 recorder 在正常退出、
EngineCore 丢失和 CUDA OOM 中都只能得到外部 `health_lost`，无法恢复内部
异常类型和阶段。这正是 schema 强制 `unknown` 的实验依据。

完整环境、结果和排除试次见：

- [`results/gpu-20260909-alpha2/VALIDATION_SUMMARY.md`](results/gpu-20260909-alpha2/VALIDATION_SUMMARY.md)
- [`TEST_PLAN.md`](TEST_PLAN.md)

alpha.4 还包含一个四卡 PyTorch FSDP2 已知答案重建：三次 `DebugLevel.DETAIL`
和三次自动 ProcessGroupNCCL Flight Recorder 试次都暴露了相同的
`_REDUCE_SCATTER_BASE` input-shape mismatch。它证明已知答案可以由两条独立
证据链重建，不代表发现了未知根因；同版本无分歧负对照仍未完成：

- [`experiments/organic-hang/REVIEW_RESPONSE_2026-09-10.md`](experiments/organic-hang/REVIEW_RESPONSE_2026-09-10.md)
- [`results/organic-hang-20260912/README.md`](results/organic-hang-20260912/README.md)

两卡 dtype campaign 的 Gate 0 也已完成：普通 unused-parameter 组进入检查点的
梯度均为 BF16，强制 mixed-gradient 对照则产生 BF16+FP32，并触发预期 PyTorch
断言。之后的 accumulated-gradient 试次触发了 wall timeout，但旧 runner 没有
保留足够的逐 rank 证据，不能据此判定机制。Gate 1b 因为把 CPU 等待点错误放在
reduce-scatter 调用而在执行前撤回；Gate 1c 又因 wall bound 过紧和停止规则错误
而撤回。Gate 1d 又因 runner 解析相邻 marker 失败而停止。Gate 1e 修复 runner 后，
机制、终止与双 rank stack 均在三次 affected trial 中通过，但 Flight Recorder 每次
只有 rank 0 dump，严格 join 0/3，因此总体按规则 FAIL-CLOSED：

- [`experiments/pytorch-unused-grad-dtype/REVIEW_PHASE2_RESULTS_CN.md`](experiments/pytorch-unused-grad-dtype/REVIEW_PHASE2_RESULTS_CN.md)
- [`experiments/pytorch-unused-grad-dtype/GATE1B_WITHDRAWAL.md`](experiments/pytorch-unused-grad-dtype/GATE1B_WITHDRAWAL.md)
- [`experiments/pytorch-unused-grad-dtype/GATE1C_WITHDRAWAL.md`](experiments/pytorch-unused-grad-dtype/GATE1C_WITHDRAWAL.md)
- [`experiments/pytorch-unused-grad-dtype/GATE1D_EXECUTION_STOP_2026-09-13.md`](experiments/pytorch-unused-grad-dtype/GATE1D_EXECUTION_STOP_2026-09-13.md)
- [`experiments/pytorch-unused-grad-dtype/GATE1E_RESULT_2026-09-13.md`](experiments/pytorch-unused-grad-dtype/GATE1E_RESULT_2026-09-13.md)
- [`experiments/pytorch-unused-grad-dtype/GATE1F_PROTOCOL.md`](experiments/pytorch-unused-grad-dtype/GATE1F_PROTOCOL.md)
- [`experiments/pytorch-unused-grad-dtype/GATE1F_RESULT_2026-09-13.md`](experiments/pytorch-unused-grad-dtype/GATE1F_RESULT_2026-09-13.md)

vLLM #53859 的 Stage 1 则验证了另一类问题：在单卡真实 EngineCore 中，
KV-event publisher 的背压会让 token 进度停止，但 `/health` 仍返回 2xx。
外部 stack 定位到了阻塞的 queue 路径；释放 consumer 后，请求恢复并完成。
应用 #53883 后，同一请求不再停滞，同时 EngineCore 内的计数器记录到 1 个
accepted event batch 和 4 个 dropped batches。这个结果支持“以事件丢失换取
服务活性”，不代表可靠投递，也不能作为生产环境丢失率：

- [`experiments/vllm-zmq-event-backpressure/STAGE1_R3_RESULT_2026-09-16.md`](experiments/vllm-zmq-event-backpressure/STAGE1_R3_RESULT_2026-09-16.md)
- [`results/vllm-zmq-backpressure-stage1-r3-20260916/`](results/vllm-zmq-backpressure-stage1-r3-20260916/)

## 11. 如何运行开发验证

从源码运行测试时必须安装开发依赖：

```bash
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
python -m compileall -q src tests
python -m ruff check .
python -m ruff format --check .
vllm-dfx --help
```

当前测试还会读取 `tests/cases.json`，检查每个 `FR-*` 和 `SR-*` 需求是否映射
到至少一个真实存在的测试方法。新增需求但没有新增测试、删除需求却留下旧
映射、或重命名测试后没有更新映射，都会让 CI 失败。

## 12. 从 v0.2 研究型 Alpha 到产品候选版本

建议按照下面的顺序推进，而不是先增加更多字段：

1. **建立开销基线**：交错执行 recorder disabled/enabled 两组相同负载，报告
   吞吐、TTFT、TPOT、端到端延迟、recorder CPU/RSS 和采集器耗时。
2. **扩展 no-progress 边界**：在现有 demand-gated verdict 之外区分长 prefill、
   正常排队和更多版本差异，再进行单卡 `SIGSTOP/SIGCONT` 实验。
3. **建立跨 rank 关联契约**：在 v0.2 单 incident identity、clock 和 content hash
   之上加入 topology、logical position 和通用 semantic join，并完成消融。
4. **补齐分布式拓扑**：至少覆盖 TP=2 worker loss/stall，输出第一个外部可观察
   divergence，同时明确不能由外部证明的 CUDA/NCCL 根因。
5. **建立版本矩阵**：覆盖多个 vLLM release、GPU 架构和指标命名差异。
6. **完成长稳**：运行 24 小时及更长的采集，检查 RSS、CPU、文件轮转、
   crash-loop 和服务非干扰。
7. **补部署能力**：提供 systemd、Docker/Kubernetes 示例、健康状态和配置校验。
8. **收集真实采用证据**：记录 linked evidence 是否改变了首个故障域判断、下一步动作、
   复现需求或 GPU 小时，而不是只统计生成文件数。
9. **再决定上游形态**：如果外部边界已经足够，就继续独立维护；如果真实案例
   反复需要 EngineCore 内部上下文，再用证据支持上游 in-process recorder。

满足前五项后，可以称为 production-preview；只有兼容、部署、长稳、安全响应
和真实采用机制都稳定后，才适合讨论正式的 `1.0` 产品承诺。

## 13. 与 vLLM RFC 的关系

本项目使用 `external-runtime-observation-v1`，而 RFC #54229 讨论的是运行在
EngineCore 内部的 `incident-snapshot-v1`。二者不能互相冒充：

- 外部 recorder 适合独立部署、故障时间线和非侵入式验证；
- 内部 recorder 才能可靠携带异常类型、执行阶段和每轮 EngineCore 状态；
- 外部实验室可以作为对照组和采用验证工具，但不是内部实现的替代品。

相关讨论：

- [vLLM issue #48966](https://github.com/vllm-project/vllm/issues/48966)
- [vLLM PR #52178](https://github.com/vllm-project/vllm/pull/52178)
- [Runtime incident snapshot RFC #54229](https://github.com/vllm-project/vllm/issues/54229)

## 14. 当前结论

这个项目已经是一个公开、可安装、可验证的 Alpha 产品原型。它最成熟的部分
是证据边界、隐私约束、health-green no-progress 的有界判定和可复现测试；最欠缺
的部分是通用跨 rank 关联、分布式验证、性能开销、长稳和真实采用。

因此现阶段最有价值的工作不是继续扩展 schema，而是让更多真实运行环境使用
它，并用可重复实验回答三个问题：它会不会影响服务、相比现有监控是否保留了
额外证据、linked evidence 是否真的改变了下一步诊断动作。
