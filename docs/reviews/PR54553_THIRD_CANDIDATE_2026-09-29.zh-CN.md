# 第三个外部候选：vLLM #54553

状态：2026-09-29，有界源码预检；尚未运行测试或向上游交付。本文为准；[英文版](PR54553_THIRD_CANDIDATE_2026-09-29.md)为翻译。目前页面显示的 PR head 是 `41dddf7`；使用本记录前须刷新 head、测试和讨论。

## 消费者与决策

消费者是 [#54553](https://github.com/vllm-project/vllm/pull/54553) 的作者与 reviewer。该 PR 拟给 EngineCore **致命错误后的关停**设置上界，避免 teardown 卡住时进程永不退出。作者在 [PR 讨论](https://github.com/vllm-project/vllm/pull/54553)中明确表示“模拟卡住的 shutdown”与正常 `SystemExit` 负控尚待验证。要回答的决策是：这份精确补丁是否既保持正常关停语义，又在 teardown 无法完成时按预算以非零状态退出。Lab 不复现其底层 XPU 驱动卡死。

## 实现前冻结的比较

| 项目 | 预测与边界 |
| --- | --- |
| 普通复现 | 一个独立睡眠后退出的 helper，或复制超时逻辑的测试，几乎不能证明 PR 的 EngineCore 异常／finally 路径。作者已知道缺少这两类控制。 |
| 可能的 Lab 增量 | 对固定源码的真实 `EngineCoreProc` 路径做一个**CPU 子进程级 fixture**，基线和补丁运行同一测试：注入受控致命错误并卡住 teardown；另测 `SystemExit`／请求关停负控。记录退出码、耗时和 teardown 是否进入。如必须把新分支复制进测试 helper，就不算 Lab 价值。 |
| 预期差异 | 基线：致命错误后的 teardown 在有界观察窗口后仍未退出；补丁：在配置的截止时间附近非零退出。两版的主动 `SystemExit` 均沿既有关停路径，不出现新的强退标记。正常完成的致命 teardown 不得被新截止时间杀死。这些是**预测**，尚非观测。 |
| 推翻或停机结果 | 不替换决策逻辑就无法触达真实进程路径；上游现有测试已经证明相同的基线／补丁／负控结果；补丁进程未在声明上界内退出；或正常关停走了强退分支。停止并记录，不继续堆装置。 |

补丁[增加默认 60 秒设置](https://github.com/vllm-project/vllm/pull/54553/files)，并在 `EngineCoreProc` 中把致命路径的 `engine_core.shutdown()` 放进 daemon thread，限时 join；仍未结束时调用 `os._exit(1)`。`SystemExit` 分支则直接调用 shutdown。这是源码阅读，不是运行结果。退出后的用户可见状态与 supervisor 重启**不在**本 CPU fixture 的范围。作者拥有 XPU 复现环境；Lab 模拟测试不能被表述为复现了底层 GPU wedge。

## 立即门槛与预算

1. 刷新 PR head，并查看 [#58279](https://github.com/vllm-project/vllm/pull/58279) 的重叠、现有测试及作者的新结果。如果作者已经交付同样的进程级负控，按重复 review 收尾，不与作者竞争。
2. **最多两小时 CPU 预检**：找出让受控致命事件和阻塞的 `shutdown()` 进入真实 EngineCore 进程 `try/except/finally` 路径的最小方法。写代码前先明确基线应失败的 fixture、较短的配置截止时间，以及独立的父进程 watchdog。CPU litmus 不等默认 60 秒。
3. 只有同一 fixture 可在两个源码版本上运行，且评分规则没有案例特判，才继续；随后实现与测试最多**再用四小时**。若必须用 GPU、XPU、下载模型或大规模重建环境，就停止并重新征求决定；本文不授权硬件。

只有身份有效、基线／补丁的致命路径形成对照、正常关停和正常完成的致命 teardown 两个负控均成立，才评为 `supported`；有效且相反的进程结果评为 `refuted`；身份、进入路径／teardown 见证缺失、装置超时或两个版本不可比则为 `unscored`。首次评分运行前须冻结精确 fixture 和阈值。仅经用户审核后才考虑交付最小可运行复现；本次预检不自动发帖、@ 维护者、开 PR 或计入 Lab 交付。按 R2 分开记录装置耗时。

## CPU 预检检查点（无 vLLM 运行结果）

当前页面显示的 `41dddf7` 中，改动的判断位于 [`EngineCoreProc.run_engine_core`](https://github.com/vllm-project/vllm/commit/41dddf7)：致命 `Exception` 进入限时 daemon-thread teardown；`SystemExit` 沿旧的直接路径。它的父提交是 `810bc3250c945829c64a745b4f695ddfd8f9a598`。[#58279](https://github.com/vllm-project/vllm/pull/58279) 修改 RPC／receiver 失败处理及关停，但其页面可见的测试计划未提供这个致命 `run_engine_core` 基线／补丁加 `SystemExit` 的对照。运行前仍须刷新两个线程，因为 head 可能变化。

[拟议的标准库 runner](../../experiments/pr54553-fatal-shutdown/README.md)从两个 checkout 导入真实外层方法，仅用受控故障／teardown 对象替换新建的 EngineCore；子进程原始 stderr 留在私有目录。六项本地评分测试通过；它们验证的是装置，**不是 vLLM**。当前 Windows Lab 环境为 Python 3.14，未安装 vLLM、Torch、pyzmq 或 msgspec，WSL 也无法访问，因此两版源码均未运行子进程。剩余执行条件是一个兼容的 Linux CPU 环境和两份干净、固定提交的 checkout；不为此租 GPU／XPU。
