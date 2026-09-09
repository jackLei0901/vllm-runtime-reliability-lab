# 产品化路线图

## 1. 最终目标

目标不是把实验脚本包装成一个命令，而是交付一个满足以下条件的公开产品：

- 操作者可以稳定安装、升级和回滚；
- 默认配置不会泄露请求内容；
- 资源开销有可复现的上界；
- 在单卡和主要分布式拓扑中不干扰 vLLM；
- 故障时能稳定生成有限、可解释的证据；
- 不支持的版本、指标和拓扑会明确失败或降级；
- 至少有真实用户证明 artifact 改变了诊断动作；
- 项目有 issue、兼容、发布和安全响应流程。

建议将“正式公开产品”分成 `public alpha → production preview → 1.0` 三个承诺
等级，避免一次性声称生产可用。

## 2. 当前基线

当前版本 `v0.1.0-alpha.3` 已达到 public alpha：

- 源码、License、Release、wheel 和 CI 可公开访问；
- 运行时零第三方 Python 依赖；
- 关闭 schema、隐私边界和资源上限已有测试；
- fake server 和 RTX 4090 故障矩阵可复现；
- 需求与测试用例可机器追踪；
- 已公开无效试次和未覆盖边界。

它尚未达到 production preview，主要缺口是开销、多卡、长稳、版本兼容、部署
模板和采用证据。

## 3. 阶段 P1：单卡运行特性闭环

### 工作项

1. 完成 recorder disabled/enabled 交错实验；
2. 记录 request/token throughput、TTFT、TPOT、E2E；
3. 记录 recorder CPU、RSS、每类 collector latency；
4. 新建真实 KV pressure/preemption 场景；
5. 对健康、压力、SIGTERM、EngineCore loss、OOM 各运行至少三次；
6. 增加 2 小时和 24 小时两个稳定运行档位；
7. 注入只读目录、磁盘空间不足、目标 PID 重启和 metrics 格式变化。

### 环境

- 单卡 RTX 4090 或显存相近的 CUDA GPU；
- 固定 vLLM commit、模型、启动参数和请求集；
- recorder 与 server 分别记录 commit 和 Python 环境；
- 所有配对实验使用相同 seed、请求顺序和到达模型。

### 通过条件

- 事前固定开销统计方法，不在看到数据后改变阈值；
- 发布配对原始汇总及 null result；
- recorder 关闭时无进程、无轮询、无 artifact；
- writer 失败不改变请求结果和服务退出状态；
- 24 小时内 RSS、CPU 和文件数量没有无法解释的持续增长；
- KV pressure/preemption artifact 的计数和保留序列可独立复算。

### 预计投入

代码与脚本约 3–5 个工作日；GPU 实际占用约 1–2 天，可分批执行。

## 4. 阶段 P2：分布式与版本兼容

### 工作项

1. TP=2：分别终止非主 rank、EngineCore 和 API server；
2. 检查 rank/process 变化、health、退出码和孤儿进程；
3. 明确外部 recorder 对 DP supervisor 的观察边界；
4. 建立 NCCL hang/abort 的受控实验，不通过日志字符串伪造根因；
5. 覆盖至少三个 vLLM 版本；
6. 覆盖至少两类 GPU 架构；
7. 为 metrics 缺失、改名和 label 变化制定兼容策略。

### 环境

- 最低 2×同型号 GPU；
- 一个当前 main/开发版、一个近期稳定 release、一个较旧受支持 release；
- 优先覆盖 Ampere/Ada 与 Hopper 中至少两类。

### 通过条件

- 不把单卡结论外推为多卡结论；
- 每个进程故障点至少重复三次；
- 没有遗留 worker、共享内存或 GPU context；
- 不支持的 metrics/拓扑会输出有限、可操作的降级状态；
- 文档中形成明确支持矩阵。

### 预计投入

约 5–8 个工作日，双卡计费时间约 1–3 天，取决于模型准备和故障注入速度。

## 5. 阶段 P3：部署与可运维性

### 工作项

1. 增加 TOML/YAML 配置文件和严格 schema；
2. 提供 systemd unit；
3. 提供 Docker 和 Kubernetes sidecar 示例；
4. recorder 暴露自身 health 和低基数 metrics；
5. 定义退出码、启动失败和重启策略；
6. 定义 artifact 目录权限、磁盘配额和保留策略；
7. 加入 schema 兼容、配置迁移和回滚说明；
8. 建立安全漏洞报告和敏感 artifact 处理流程。

### 通过条件

- 新用户可以只按文档部署，不需要理解源码；
- 错误配置在启动时失败并给出明确字段；
- recorder crash-loop 不会无限生成文件；
- recorder 自身不可用不会改变 vLLM 的生命周期；
- 升级和回滚各有一次自动化端到端测试。

### 预计投入

约 1–2 周，不一定需要 GPU，可与 P2 并行准备。

## 6. 阶段 P4：真实用户试用

### 工作项

1. 招募 2–3 个真实 vLLM 使用者进行 opt-in 试用；
2. 不上传 prompt/token，只分享 schema-valid artifact 或人工审核摘要；
3. 每个事故记录 artifact 是否：
   - 改变首个故障域判断；
   - 缩小下一步操作；
   - 避免一次复现；
   - 减少 GPU 调试时长；
4. 同时记录“没有额外价值”的事件；
5. 根据真实引用字段删除长期无人使用的字段。

### 通过条件

- 至少获得 5 个真实事件或明确记录样本不足；
- 至少一名外部使用者能独立完成安装、采集和解释；
- 公开正、负结果和限制；
- 没有发生敏感字段泄露；
- 根据采用结果作出继续、收窄或停止的明确决策。

### 预计投入

2–6 周观察期。该阶段主要受用户和真实故障频率限制，不应通过合成事故填充
采用指标。

## 7. 阶段 P5：production preview

满足 P1–P4 后发布 `0.2.0` production preview，并承诺：

- 固定支持矩阵；
- 有限 schema 兼容周期；
- 有界默认资源预算；
- 部署模板和升级说明；
- 已验证的单卡与多卡场景；
- 明确的支持/不支持边界。

此阶段仍不承诺自动根因分类、自动重启或自动修复。

## 8. 阶段 P6：1.0 决策

只有下面条件同时满足才建议发布 1.0：

- production preview 经历至少两个版本周期；
- 兼容矩阵和长稳结果稳定；
- 有持续外部使用，而不是只有作者实验；
- 安全、升级、回滚和故障处理流程可执行；
- 字段集合经过真实事件裁剪；
- 是否进入 vLLM EngineCore 已有明确决策。

如果真实采用表明外部 recorder 没有增加诊断价值，应保留实验和负面结论，
停止扩大产品承诺，而不是为了达到 1.0 继续增加功能。

## 9. 近期执行顺序

按投入产出比，建议下一轮依次执行：

1. 固化单卡配对 overhead harness；
2. 在无需 GPU 的环境完成配置 schema 与 systemd 草案；
3. 租单卡完成 overhead、KV pressure 和 2 小时运行；
4. 分析结果，决定是否值得投入 24 小时和双卡；
5. 租双卡完成 TP=2 worker/EngineCore 故障矩阵；
6. 发布 `0.2.0-rc1` 试用包与支持矩阵；
7. 以可执行工具和结果表推进 RFC，而不是继续扩充 RFC 正文；
8. 招募真实使用者，进入 adoption gate。

## 10. 项目看板建议

GitHub milestone 可以按下面方式划分：

- `P1-single-gpu-evidence`
- `P2-distributed-compatibility`
- `P3-deployment`
- `P4-adoption`
- `0.2.0-production-preview`

每个 issue 必须包含：问题、范围、验收测试、所需环境、输出物、隐私边界和
不包含的工作。没有验收条件的“增强可观测性”类 issue 不进入当前里程碑。

