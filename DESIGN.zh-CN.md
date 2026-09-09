# 方案设计：vLLM Runtime Reliability Lab

## 1. 设计目标

本项目希望在不修改 vLLM 的前提下，保存故障发生前的一段有限运行历史，供
人工排障、故障复现和后续自动化评估使用。

设计优先级依次是：

1. 不干扰被观察的推理服务；
2. 不把推测写成内部事实；
3. 默认不采集 prompt、token 和高基数字段；
4. 内存、磁盘和采集开销有明确上限；
5. 结果可通过 schema、测试和复现实验检查；
6. 保留升级到 EngineCore 内部 recorder 的清晰边界。

## 2. 系统边界

### 2.1 当前 Alpha 架构

```text
                         independent process
┌─────────────────────────────────────────────────────────┐
│                    vllm-dfx recorder                    │
│                                                         │
│  /health ─┐                                             │
│  /metrics ├─> CadencedCollector ─> ExternalObservation  │
│  /proc/PID ┤                              │              │
│  nvidia-smi┘                              v              │
│                                      bounded deque       │
│                                             │            │
│                                             v            │
│                                    TriggerClassifier     │
│                                             │            │
│                                             v            │
│                                      IncidentWriter      │
└─────────────────────────────────────────────┼────────────┘
                                              v
                                  incident-*.json (bounded)
```

recorder 与 vLLM 不共享地址空间，也不调用 EngineCore 内部接口。它只读取
HTTP、显式 PID 和 GPU 聚合状态。

### 2.2 能观察和不能观察的内容

| 类别 | 当前可以观察 | 当前不能可靠观察 |
| --- | --- | --- |
| 服务 | 健康状态、公开 metrics | 请求内部状态、异常对象 |
| 进程 | PID 是否存活、RSS/VMS、线程数 | 哪段 EngineCore 代码触发退出 |
| GPU | 聚合显存使用量、利用率等公开字段 | kernel 级故障原因、XID 因果链 |
| 调度 | running/waiting、KV 使用率、preemption 总量 | 每个 request 的调度历史 |
| 故障 | 外部触发种类和时间顺序 | CUDA OOM、worker loss 等内部根因 |

因此，外部 artifact 的 `internal_kind` 和 `internal_stage` 被 schema 固定为
`unknown`。这不是功能缺失的临时占位，而是当前证据边界的契约。

## 3. 组件设计

### 3.1 CLI

`vllm-dfx` 提供四个入口：

- `snapshot-env`：生成私有环境快照；
- `record`：采集有界历史并在触发时写 artifact；
- `inject-signal`：显式、dry-run-first 的隔离环境故障注入；
- `summarize`：把公开 artifact 转成 Markdown 摘要。

CLI 只负责参数解析和组装，不在入口层实现采集或推理逻辑。

### 3.2 CadencedCollector

不同数据源使用独立周期：

- health、metrics、process 默认 1 秒；
- GPU 默认 5 秒；
- 主循环默认 1 秒。

采集器缓存每个源的最新结果，并在对应周期到达时刷新。这样避免每轮都启动
`nvidia-smi`，也允许后续分别量化每类采集器的成本。

网络超时、连接失败和解析错误会归约为有限错误类型。公开 artifact 不保存
URL、命令行或原始异常文本。

### 3.3 ExternalObservation

每次成功采集生成一个显式字段对象，主要包含：

- 单调递增 `sequence`；
- 单调时钟与 UTC 观察时间；
- 本轮实际采样的数据源；
- health 状态；
- process 聚合状态；
- KV cache、running、waiting、preemption 等选定指标；
- GPU 聚合状态。

所有公开字段在代码和 JSON Schema 中双重定义。不能把任意字典透传到输出。

### 3.4 Bounded recorder

recording path 为单写者模型，不需要锁。样本写入固定 `maxlen` 的 `deque`：

- 队列未满时追加；
- 队列满时覆盖最旧记录；
- appended、overwritten、dropped 分别计数；
- artifact 保存 retained sequence 起止值，便于检查计数是否自洽。

默认不持续写原始 JSONL，因此正常运行时不会产生无限增长的时间线。
`--private-raw-timeline` 仅用于受控实验，明确不属于公开、有限的产品契约。

### 3.5 TriggerClassifier

当前触发类型：

| 触发 | 判定规则 | 防误判设计 |
| --- | --- | --- |
| `process_exit` | 显式 PID 已退出 | 不通过模糊命令匹配寻找 PID |
| `health_lost` | 曾成功后连续 N 次失败 | 启动阶段连接失败不算服务死亡 |
| `kv_pressure` | KV 使用率达到阈值 | 只使用允许公开的聚合指标 |
| `preemption_storm` | 两次有效样本间计数增量达到阈值 | 不使用单次累计值猜测时间范围 |

分类结果只描述外部事实。触发器不会根据时间接近、日志字符串或退出顺序推断
内部根因。

### 3.6 IncidentWriter

writer 的写入顺序是：

1. 把内部对象投影为公开 artifact；
2. 检查字段和值域；
3. JSON 编码；
4. 检查 256 KiB 上限；
5. 清理遗留临时文件并轮转旧 artifact；
6. 写临时文件；
7. 原子替换为最终文件。

POSIX 下临时和最终文件均使用 `0600`。每个目录最多保留四份完成文件。

任何校验、编码、轮转和 I/O 异常都会转换为有限的 writer error，而不是向
被观察服务发送信号或让 recorder 反向影响 vLLM。

### 3.7 Identity

incident ID 使用进程启动时生成的随机 HMAC key，只用于同一次 recorder 生命周期
内的文件关联。它不能跨进程、跨重启或跨主机比较。

这是隐私与关联能力之间的主动选择。未来如果需要稳定的 fleet signature，必须
设计独立、显式授权的身份机制，不能悄悄放宽 v1 的隐私约束。

## 4. 数据契约

公开契约名称为 `external-runtime-observation-v1`。核心原则：

- 所有对象均设置 `additionalProperties: false`；
- 内部故障类型和阶段只能是 `unknown`；
- target vLLM/Torch 版本只能由操作者显式提供；
- 未观察到的数据使用 `null`，不能用默认值伪造；
- prompt、token ID、request ID、路径、命令行、环境变量和 traceback 不进入
  公开契约。

任何新增公开字段都必须同时完成：

1. 需求说明；
2. allow-list 和数据类修改；
3. JSON Schema 修改；
4. 隐私 canary；
5. 需求—测试映射；
6. schema 兼容性判断；
7. 至少一个真实诊断场景说明其必要性。

## 5. 故障处理策略

| recorder 自身故障 | 当前处理 |
| --- | --- |
| 单次采集异常 | dropped 计数增加，继续下一轮 |
| private timeline 无法打开 | 关闭私有时间线，继续内存采集 |
| private timeline 写失败 | 停止写时间线，不停止 recorder |
| artifact 校验失败 | 返回有限错误，不抛到主循环 |
| artifact 超过大小上限 | 拒绝写入并计数 |
| 目录只读或磁盘 I/O 失败 | fail-open，记录一次有界提示 |
| run summary 无法写入 | 输出提示，recorder 正常结束 |

这些规则只保证 recorder 不主动伤害服务，不表示 recorder 自身永远不会崩溃。
产品候选版本仍需加入进程级 watchdog、退出码契约和长稳故障注入。

## 6. 安全与隐私

### 6.1 默认公开材料

只有 `incident-*.json` 按严格公开契约生成。

### 6.2 默认私有材料

环境快照、注入记录、原始服务日志、private timeline 和 run summary 需要人工
检查后才能分享。

### 6.3 威胁模型边界

当前防护重点是避免因字段设计无意泄露业务输入和运行路径。它不是：

- 通用数据防泄漏系统；
- 对抗已控制宿主机攻击者的安全边界；
- 远程认证或多租户隔离方案；
- artifact 自动上传服务。

## 7. 验证架构

验证分为四层：

1. **契约层**：schema、隐私、大小、权限、轮转、ID；
2. **组件层**：HTTP、metrics 解析、cadence、trigger、writer fail-open；
3. **进程层**：fake server、显式 PID、signal、退出码、孤儿进程；
4. **GPU/拓扑层**：EngineCore loss、运行时 OOM、KV pressure、TP/DP/NCCL。

`tests/cases.json` 将 FR/SR 需求映射到具体测试方法，CI 会检查映射双向完整且
测试方法确实存在。GPU 结果另附环境、基线 commit、排除试次和未覆盖边界。

## 8. 产品候选架构

如果外部 Alpha 的开销和采用验证通过，下一阶段建议保持采集进程独立，并增加
部署与自观测能力：

```text
vLLM pod/process                  recorder sidecar/service
┌─────────────────┐              ┌────────────────────────┐
│ health/metrics  │------------->│ bounded collectors     │
│ API server PID  │------------->│ recorder health metrics│
└─────────────────┘              │ local bounded writer   │
                                 └───────────┬────────────┘
                                             v
                                  operator-controlled store
```

产品候选版本需要增加：

- 配置文件及严格校验；
- recorder 自身 `/health` 与低基数 metrics；
- systemd、Docker sidecar、Kubernetes 示例；
- 明确的兼容矩阵和 schema 升级策略；
- artifact 导出由操作者控制，默认不自动上传；
- crash-loop、磁盘满、权限变化和服务重启的端到端测试。

## 9. 与内部 recorder 的演进关系

外部 recorder 的转换接缝是 `ExternalObservation`，而不是直接复用公开文件格式。
如果真实案例证明必须获取内部异常和每轮 EngineCore 状态，可以新增内部 producer：

```text
external collectors -> ExternalObservation -> external artifact v1

EngineCore producer  -> InternalObservation -> incident snapshot v1
                                |
                         shared bounded/writer ideas
```

内部实现必须使用单独版本的契约，不能把外部 v1 的 `unknown` 字段重新解释为
内部事实。

## 10. 关键设计决策总结

- 独立进程优先，先验证需求和采用，再讨论上游所有权；
- 有界历史优先于全量日志复制；
- allow-list 优先于事后 deny-list；
- 明确未知优先于不可靠的自动根因分类；
- writer fail-open 优先于“保证每次都写成功”；
- 真实实验、无效试次和局限说明必须一起发布；
- 自动修复暂不进入 v1，先证明证据和分类可靠。

