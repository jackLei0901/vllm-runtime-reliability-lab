# vLLM Runtime Reliability Lab 中文指南

> 当前版本：`v0.1.0-alpha.3`  
> 当前定位：可公开安装和复现的研究型 Alpha，不是生产监控产品。

相关文档：

- [`DESIGN.zh-CN.md`](DESIGN.zh-CN.md)：详细方案设计与产品候选架构；
- [`PRODUCT_ROADMAP.zh-CN.md`](PRODUCT_ROADMAP.zh-CN.md)：阶段计划和验收条件；
- [`PRIOR_ART_AND_VALUE.zh-CN.md`](PRIOR_ART_AND_VALUE.zh-CN.md)：关联证据的公开先例、实际收益和适用边界；
- [`REQUIREMENTS.md`](REQUIREMENTS.md)：当前英文需求基线；
- [`TEST_PLAN.md`](TEST_PLAN.md)：当前英文测试计划。

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
可验证的契约。后续版本还要解决单份 artifact 无法回答的问题：哪个 rank
首先出现分歧、哪些 producer 缺失，以及 `/health=200` 时服务是否已经失去进展。

项目的验收标准不是“是否采到了数据”，而是证据能否补上实际运维缺口：

| 运维缺口 | 产品输出 | 应支持的动作 |
| --- | --- | --- |
| `/health=200`，但已有请求停止推进 | 带支撑证据的、有界 `suspected_no_progress` 状态变化 | 进入排查、摘流或重启流程，而不是让静默故障继续保持健康 |
| 单个 rank 卡住或消失，其他进程只有局部状态 | 经校验的 producer 集合、缺失 peer、状态分歧和顺序边界 | 不依赖故障时 collective，缩小第一个应排查的故障域 |
| 多份文件无法证明属于同一次事故 | 关闭的 manifest、身份、clock 和内容 hash | 在诊断前拒绝混入其他运行或被篡改的证据 |

当前 Alpha 提供的是有界本地证据这一基础能力。表中的 no-progress 和多 producer
输出属于 v0.2 验收目标，不是当前已经交付的功能。

### 1.1 计划中的 stack adapter 有严格部署门槛

CPU stack adapter 同样不是当前能力。`py-spy` 需要读取另一个进程的内存：Linux
attach 通常需要 root 或调整 `ptrace_scope`，Docker/Kubernetes 往往需要
`SYS_PTRACE`。默认采样可能短暂停顿目标进程；`--nonblocking` 可以避免暂停，但
由于多次内存读取不是原子的，可能得到错误或不完整 stack。具体限制见
[py-spy FAQ](https://github.com/benfred/py-spy#frequently-asked-questions)。

实现 joiner 之前必须先做 go/no-go 实验：在固定耗时和样本数预算内，能否从阻塞
进程获得有用的 Python/native 上下文。单进程、单 GPU 只能验证 attach 以及
CUDA/native wait；真实的 unmatched NCCL collective 需要多 rank GPU 环境。
权限拒绝、超时或 partial output 都是正常的明确结果，不能被当作 recorder 异常。

### 1.2 为什么不直接使用 Prometheus

Prometheus 和 OpenTelemetry 适合持续监控，本项目不替代它们。当前 recorder
交付的是一个本地、触发时冻结、有大小上限且可以离线校验和分享的故障窗口，
自动对齐少量 service、process 和 GPU 观察。

如果现有监控系统已经可靠保留了同等分辨率的数据，并能在故障时自动形成
可分享证据，那么本项目可能没有增量价值。后续必须通过 Prometheus 对照和
unlinked-versus-linked 消融证明价值，不能把它作为前提。

### 1.3 当前 Alpha 与后续目标

当前 Alpha 只完成单目标外部时间线和 bounded artifact。它尚未实现：

- health-green no-progress 检测；
- API server、EngineCore 和 worker/rank 自动发现；
- 多 producer correlation manifest；
- 跨 rank semantic join；
- 自动摘流、重启或根因分类。

这些是 v0.2 及后续版本的目标，不能作为当前发布能力宣传。

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
| 正常 SIGTERM、EngineCore SIGKILL、受控 CUDA OOM | RTX 4090 已验证 |
| 需求到测试用例的机器检查 | 已完成 |

### 尚未完成

| 产品化缺口 | 为什么重要 |
| --- | --- |
| recorder 开关配对的性能实验 | 尚不能给出 CPU、RSS、TTFT、TPOT 和吞吐开销上界 |
| 新一轮真实 KV pressure/preemption 实验 | 目前只有触发逻辑测试和历史实验，缺少 alpha.3 的完整实测 |
| alpha.3 的 TP=2/DP/NCCL 验证 | 单卡结论不能外推到分布式拓扑 |
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

## 12. 从公开 Alpha 到产品候选版本

建议按照下面的顺序推进，而不是先增加更多字段：

1. **建立开销基线**：交错执行 recorder disabled/enabled 两组相同负载，报告
   吞吐、TTFT、TPOT、端到端延迟、recorder CPU/RSS 和采集器耗时。
2. **验证 no-progress 边界**：先用 fake server 区分 idle、长 prefill、正常排队、
   持续进展和 health-green stall，再进行单卡 `SIGSTOP/SIGCONT` 实验。
3. **建立关联契约**：加入 job-scoped `run_id`、producer identity、clock declaration、
   artifact hash 和 closed manifest，并完成 unlinked-versus-linked 消融。
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
是证据边界、隐私约束、有限资源使用和可复现测试；最欠缺的部分是
health-green no-progress、跨 producer 关联、分布式验证、性能开销、长稳和真实采用。

因此现阶段最有价值的工作不是继续扩展 schema，而是让更多真实运行环境使用
它，并用可重复实验回答三个问题：它会不会影响服务、相比现有监控是否保留了
额外证据、linked evidence 是否真的改变了下一步诊断动作。
