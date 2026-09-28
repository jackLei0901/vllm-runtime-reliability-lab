# Engine 存活、健康与关闭：内部 RFC 提纲

英文原稿：[Engine liveness RFC outline](ENGINE_LIVENESS_RFC_OUTLINE_2026-09-27.md)。

状态：**Lab 内部设计提纲，不是已提交上游的 RFC，也不是实施方案**。
[统一证据审阅](reviews/ENGINE_LIVENESS_UNIFIED_REVIEW_2026-09-27.md)
说明了为什么不必为这份提纲再开 GPU，以及什么条件下才值得补跑。
[定向查重](reviews/ENGINE_LIVENESS_DUPLICATE_CHECK_2026-09-27.md)
以 `upstream/main` `55de40a2fc` 核对了 C1/C2/C7/C8 和 K5 归因问题；
其中 GitHub 状态仅是 2026-09-27 的快照。
源码基线为 vLLM `c8602c79062440074a018c1d5f875a5571eb6881`，
[契约清单](ENGINE_LIVENESS_CONTRACT_INVENTORY_2026-09-27.md)保留了源码映射。
真正向上游发言前，还要重新核对目标源码与线程状态。

## 1. 要回答的问题

当 EngineCore 停止前进或服务正在关闭时，API server、EngineCore、executor、
worker 和外部健康探针分别可以得出什么结论？目前结论取决于等待由哪个进程持有、
以及采用哪个 executor。目标是形成跨进程的**状态、信号所有权和嵌套时间预算**契约，
使现有修复可以组合，而不是不断增加互不协调的超时开关。

关闭语义承接已因过期而关闭、但问题未获解决的
[#24885](https://github.com/vllm-project/vllm/issues/24885)；卡顿检测应回应
[#52365](https://github.com/vllm-project/vllm/pull/52365)中“整个 EngineCore iteration，
还是只管异步输出等待”的边界问题，而非提出第五个 watchdog。
本提纲不要求新采集器、自动重启策略或整体重写故障处理。

## 2. 支撑契约的证据与边界

| 契约张力 | 已有证据及限制 | 对上游可成立的表述 |
| --- | --- | --- |
| C1：请求排空时间与进程强杀宽限期共用 `shutdown_timeout`；C8 的排空/中止输出另有上游工作 | [K1 CPU 检查](reviews/ENGINE_LIVENESS_K1_K2_CPU_RUN_2026-09-27.md)运行真实关闭 helper：零宽限期下，100 ms 的 SIGTERM 清理 handler 尚未结束就收到 SIGKILL。[#36666](https://github.com/vllm-project/vllm/pull/36666)曾设 5 秒下限；维护者在已关闭的 [#40985](https://github.com/vllm-project/vllm/pull/40985)称其为有意设计；[#43016](https://github.com/vllm-project/vllm/pull/43016)后来移除了这一限制；[#52281](https://github.com/vllm-project/vllm/pull/52281)只在 ROCm 恢复了独立清理宽限期。当前核对中 `_shutdown_subprocesses` 仍有 5 秒下限，`shutdown` 则没有。未测量 CUDA 资源泄漏或客户端中止结果。 | 问“移除下限是否也有意适用于 CUDA”，这是**回归问题**，不是已证实的误改。C8 已由 [#36964](https://github.com/vllm-project/vllm/pull/36964)覆盖。 |
| C2：同一个 worker 超时限制内外两层关闭 | K2 用真实内层函数与假进程树测试 `x=1,2,5`：外层均先截断内层升级；较长宽限期对照完成 `x+4`。[#43154](https://github.com/vllm-project/vllm/pull/43154)引入共用值；[#55632](https://github.com/vllm-project/vllm/issues/55632)已报告 ROCm 上同类嵌套预算冲突。`x < x+4` 的一般关系来自源码，不是三个样本的外推。 | 承认已有报告，先约定嵌套预算不变量，再考虑默认值；不宣称此类问题为新发现。 |
| C3：worker 卡顿的检测随 executor 改变 | [K5](reviews/ENGINE_LIVENESS_K5_DUAL_GPU_RESULT_2026-09-27.md)中 TP=1 可评分：45 秒注入期间 `/health` 始终为 200，解除后恢复。TP=2 约在实验性缩短的 30 秒期限后由 200 变 503，请求返回 500，报 `sample_tokens` RPC 超时；默认 `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` 为 300 秒。TP=2 按预注册规则**未评分**，因为原预测是 `execute_model`；异步等待顺序解释了预测错误。 | 将观测差异与源码推断分开；不声称证明了自然故障根因、实际无限期挂起或默认五分钟行为。 |
| C4：EngineCore 下层健康检查未接线；C7：迭代超时变量有定义却无读取点 | 仅有源码清单，没有新增行为测试。开放中的 [#52365](https://github.com/vllm-project/vllm/pull/52365)拟再次用于 GPU event wait。 | C4 是契约问题；C7 是现有 PR 的背景，不单独报新 bug。 |

K5 揭示的形态接近二元：TP=1 在测量窗口内看起来健康；TP=2 到期限后进入
terminal engine error。实验设了 `VLLM_KEEP_ALIVE_ON_ENGINE_DEATH=1`，因此仍能看到
`/health` 503；默认启动方式可能直接结束 API 进程。两者都没有表达非终止的
“存活但未前进”。可选 FT 框架虽有 `UNHEALTHY`，但作用范围不同，需要对齐。
C5（#58242/#58279）与 C6（#48745/#49000）已有上游负责人，本提纲只把它们当作集成输入。

## 3. 待审阅的契约骨架

### 生命周期与健康信号

生命周期和健康/进展应是两条轴，但判断“没有进展”时必须联合考虑。
候选生命周期为 `starting → ready → draining → stopping → exited`，
另有 terminal failure，以及有意的 `ready ↔ paused`、`ready ↔ sleeping`。
这不是声称当前代码已有统一枚举。`pause_generation(mode="keep")` 会冻结排队请求；
`sleep` 接受相同的暂停模式，因此“有需求但不出 token”也可能是健康状态。

候选健康事实包括：进程存活且可联系、EngineCore 循环可响应、已接纳工作在有界窗口内前进、
因生产者缺失或过期而无法观测进展，以及由归属明确的错误或进程退出确认的终止故障。
进展样本缺失既不能推出“engine 缺失”，也不能推出“engine 健康”。
非终止的疑似卡顿状态需要需求、时效、进程/engine 身份、生命周期状态与有界观测窗口。
长 prefill、编译、graph capture、空闲、慢速多模态工作、有意暂停/休眠、
`WAITING_FOR_REMOTE_KVS` 和 DP dummy batch 都是必要负对照。
loop ping 成功不等于 token 前进。每次状态转换都需要明确负责者，并规定如何传到 API、
日志/指标和退出码。

### 关闭与超时预算

请求排空、EngineCore 收尾、worker 宽限、worker SIGTERM→SIGKILL 升级、
外层进程树强杀，应有不同预算。固定源码中的 Python 入口不同；Rust 又包装了 headless Python 入口：

| 入口 | 目前的外层预算 | 契约问题 |
| --- | --- | --- |
| Python API server 经 `MPClient.shutdown(timeout)` 关闭 | launcher 把默认 0 的 `shutdown_timeout` 交给 manager。 | 请求排空为 0 是否也应使 EngineCore/worker 收尾宽限为 0（C1）？ |
| `vllm serve` 的 headless 或多 API 父进程 | **信号触发**时，headless 直接把 `shutdown_timeout` 传给 `engine_manager.shutdown`；多 API 先关闭 API server，再给 engine manager 剩余的 `shutdown_by` 预算，默认亦为 0。其他退出路径传 `None`，helper 使用 5 秒 fallback；此入口不经过 `MPClient`。 | C1 同样存在；多 API 先行阶段耗时后，EngineCore 还剩多少时间？ |
| `MPClient` 后台资源清理 | 外层采用 `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=x`，内层第一次等 worker 也用 `x`。 | 外层如何容纳内层 `x`、再加 4 秒升级及 EngineCore 清理/传递余量（C2）？ |
| Rust 托管 Engine | Rust 启动 headless Python `serve`，对该进程组设置至少 5 秒的外层宽限；配置为 0 时，不向 Python 传正值 `--shutdown-timeout`。Python 内部仍会以 0 关闭 EngineCore manager。 | 5 秒只是外层上限，不能当作 EngineCore/worker 的清理预算。尚未核实 EngineCore 是否属于接收 Rust SIGTERM 的同一进程组。 |

只修 C2 不会修复默认 API 关闭路径；只在 `MPClient` 或 launcher 修 C1，
又会漏掉 headless、多 API 和 Rust 托管服务中的 Python 子进程。
数值选择之前应检查共有的 EngineCore manager 边界和各平台例外。

到排空期限时，请求是得到中止确认，还是直接随进程死亡？有意关闭与崩溃在前端状态和
退出码上如何区分？超时报告应区分**调用方正在等的 future**与**worker 实际执行的操作**，
并尽可能给出 rank、阶段、owner，以及“疑似卡顿”或“终止故障”。K5 中 worker 卡在
`execute_model`，但消息报 `sample_tokens`，因为异步 EngineCore 先等待后者的 future。
这只是 [#54638](https://github.com/vllm-project/vllm/pull/54638) 的诊断背景，不是新提案。
后续测试必须记录实际 `async_scheduling` 与 concurrent-batch 配置。

### 对外健康接口

需要决定进展健康是作为 opt-in `/health` 模式、独立 readiness/progress 接口，
还是先作为指标；在负对照和运维语义达成共识前，保留默认 endpoint 契约。
本提纲不包括新的控制器或自动重启策略；还需比较现有 opt-in FT 的 `UNHEALTHY` 语义。

## 4. 与上游已有工作的关系

这是集成提纲，不替代别人的补丁。关闭半部分延续 [#24885](https://github.com/vllm-project/vllm/issues/24885)；
卡顿半部分试图回答 [#52365](https://github.com/vllm-project/vllm/pull/52365) 未决的检测范围问题。
实施前至少对齐：

- [#54638](https://github.com/vllm-project/vllm/pull/54638)、
  [#52365](https://github.com/vllm-project/vllm/pull/52365)、
  [#56816](https://github.com/vllm-project/vllm/issues/56816) / [#55700](https://github.com/vllm-project/vllm/pull/55700)、
  已合并的 [#58779](https://github.com/vllm-project/vllm/pull/58779)：分别涉及 worker watchdog、
  GPU 等待限时、仅诊断的 stack-dump watchdog、draft-token RPC 限时。需要划定观察责任及结果语义。
- [#54553](https://github.com/vllm-project/vllm/pull/54553)、
  [#43154](https://github.com/vllm-project/vllm/pull/43154)、
  [#55632](https://github.com/vllm-project/vllm/issues/55632) 及 ROCm/XPU 宽限期修复：
  承认既有嵌套预算报告，并参考其不变量测试方向。
- [#58279](https://github.com/vllm-project/vllm/pull/58279)：RPC 回应对应关系；
  [#49000](https://github.com/vllm-project/vllm/pull/49000) 与
  [#52178](https://github.com/vllm-project/vllm/pull/52178)：有意关闭与 fatal cause 传播。
- [#36258](https://github.com/vllm-project/vllm/pull/36258)：`/live` 与 `/health` 的存活/就绪区分；
  [#31252](https://github.com/vllm-project/vllm/issues/31252)：API 父进程可能截断 EngineCore 清理；
  [#36964](https://github.com/vllm-project/vllm/pull/36964)：关闭时中止请求并排空输出。
  C7 与 C8 均已有相应工作，不作为无主发现提出。
- [#36451](https://github.com/vllm-project/vllm/pull/36451)：循环 ping 与请求实际进展的区别；
  Lab 的证据已在该 PR 评论中，不重复提及审阅人。
- 可选 FT 框架中的 `UNHEALTHY`：需要对齐作用范围和语义。

以上 PR 状态仍是快照；上游文字不能把已合并、变化或被取代的工作当作待办。

## 5. 验证与分阶段实施问题

1. 先审契约：状态词汇、每类信号的 owner、`/health` 兼容性及嵌套预算不变量。
   记录备选方案，不先开宽泛代码 PR。
2. CPU 不变量：保留 K1/K2，对“零排空但非零进程清理宽限”、预算嵌套、
   有意关闭与崩溃传播补可证伪断言。
3. 只在必要时补 GPU：K5 是机制例子，不是双臂通过判定。若因其他原因重跑，
   预注册 async-on → `sample_tokens`、async-off → `execute_model`，分别评分并记录实际配置；
   不为修这份文档单独租卡。
4. 非终止健康负对照：长 prefill、首次编译、空闲、长生成、有意暂停/休眠、
   `WAITING_FOR_REMOTE_KVS`、DP dummy batch 和多 Engine 分区；先测检测延迟与误报。
5. 获得维护者设计方向后，按一个 seam 一次的方式实施、补回归测试与兼容性说明；
   不在重叠的生命周期工作等待审阅时再开并行 PR。

## 6. 入场门槛与上游出口

当每个拟议转换都有 owner、可观测输入、超时规则和可证伪测试计划时，Lab 提纲完成。
定向查重已记录在 `55de40a2fc` 的审阅文档中；实际发帖前仍要重查目标源码及线程状态，
并分开 K5 的可评分事实与源码解释。

上游短稿应只有一个状态图、一张预算表、两个具体例子和少数可二选一回答的设计问题，
链接 Lab 的完整证据，而非复制整个清单。默认出口是对已有 owner 讨论的有界补充，
不是新 issue、第五个机制或默认新 RFC。C1 是唯一候选的新上游问题；若要提出，
应优先选择现有关闭语义线程，并遵守“先推进已有 PR”的既定顺序。
