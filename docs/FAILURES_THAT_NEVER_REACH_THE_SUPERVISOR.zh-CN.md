# 没有到达 supervisor 的故障

推理服务常常在所有权边界上丢失故障语义：worker、rank 或 engine 已经无法继续完成
有效工作，但负责恢复的组件只收到一个更弱的信号，甚至没有收到信号。

本文中的 *supervisor* 泛指应当判断故障并采取行动的主体，包括父进程、服务管理器、
健康检查、运维人员和诊断分析器，不只指 vLLM 的 `DPSupervisor` 类。

四个已有调查呈现了同一个结构：

| 案例 | 实际发生的故障 | 穿过边界的信号 | 不安全的结论 |
| --- | --- | --- | --- |
| vLLM [#53859](https://github.com/vllm-project/vllm/issues/53859) / [#53883](https://github.com/vllm-project/vllm/pull/53883) | EngineCore 在向已满事件队列发布时停止 token 推进 | 进程仍存活，`/health` 仍返回 2xx | 服务健康 |
| PyTorch [#196968](https://github.com/pytorch/pytorch/issues/196968) / [#197232](https://github.com/pytorch/pytorch/pull/197232) | 一个 rank 在 communicator destruction 期间停止响应 dump 请求 | 另一个 rank 产生了 Flight Recorder dump，该 rank 没有 | 缺失的 rank 没有参与 |
| vLLM [#48966](https://github.com/vllm-project/vllm/issues/48966) / [#52178](https://github.com/vllm-project/vllm/pull/52178) | EngineCore 意外死亡 | 顶层 serving 进程以状态 0 退出 | 服务正常关闭 |
| vLLM DP supervisor Gate 0 | 子进程异常退出，或启动后的健康探测失败 | 父进程返回状态 0 | 被管理的进程组正常停止 |

它们不是同一个 bug 的四种命名。共同缺陷是：真正决定恢复行为的状态，没有以可安全
解释的形式穿过进程、rank 或诊断边界。

## 1. alive 与 health green 都不等于 progressing

#53859 实验在真实单卡 EngineCore 中确定性制造 KV event queue 背压。受影响组中：

- EngineCore PID 与 start-time identity 保持稳定；
- `/health` 持续返回 2xx；
- 已进入系统的请求停止 token 推进；
- 外部栈显示 EngineCore 位于
  `ZmqEventPublisher.publish -> Queue.put`。

应用 #53883 的提议修改后，同一触发条件下请求能够完成，但活性恢复存在一个实测代价：
四个 event batch 被丢弃。

这个 bug 不是 Lab 发现的。Lab 的贡献是把已有报告和修复提议转换成有边界的 base/fix
结论，并保留了简单“已修复”标签会隐藏的数据丢失权衡。

```text
process alive + health 2xx != useful work is progressing
```

## 2. 缺少证据不等于缺少参与者

在 #196968 中，Flight Recorder 生成了 rank 0 的 dump，却没有 rank 1 的 dump。如果把
文件集合直接当作成员列表，最重要的 rank 会被错误地排除在诊断之外。

外部进程状态和双 rank 栈证明 rank 1 仍然存活。隐私受限的 shutdown-stage flags 进一步
给出了更窄的顺序：rank 1 已停止 dump responder，进入 communicator destruction，并在
rank 0 发起 dump 请求之前一直阻塞在那里。

缺失的 artifact 描述的是诊断 producer，而不是 participant：

```text
producer missing != member missing
```

Lab 首先发现并提交了 #196968，并对 #197232 的 C++ 修改提议完成 base 3/3 失败、fix
3/3 通过的验证。截至 2026-09-21，该 PR 仍是开放提议。本文不声称修复已被接受或合入；
完整 case study 继续等待明确的 upstream 结果。

## 3. 内部致命故障变成了退出状态 0

对于 #48966 报告的生命周期缺口，系统级 contract 很简单：EngineCore 意外死亡必须让
顶层进程非零退出，而有意发送的 SIGTERM 仍应正常退出。

Lab 的成对验证结果为：

| 路径 | 注入 | 顶层状态 |
| --- | --- | ---: |
| baseline | `SIGKILL` EngineCore | 0 |
| #52178 提议修改 | `SIGKILL` EngineCore | 1 |
| #52178 提议修改 | `SIGTERM` API server | 0 |

TP=2 进程级验证也得到一致边界：worker 或 EngineCore 丢失时状态为 1，有意 SIGTERM
时状态为 0，并且没有遗留进程。

该 bug 由他人独立发现；Lab 提供系统级验证。这里的教训是：父进程的 clean status
只说明生命周期传播的结果，不能证明子进程执行过程健康。

## 4. supervisor 看到了失败，却仍返回成功

DP supervisor Gate 0 使用精确的 upstream `dp_supervisor.py`，对生命周期边界执行四个
cell：

| Cell | 子进程状态 | 父进程状态 |
| --- | ---: | ---: |
| ready 后有意 SIGTERM | -15 | 0 |
| ready 前子进程异常退出 | 17 | 0 |
| ready 后子进程异常退出 | 17 | 0 |
| ready 后 health probe 失败 | cleanup 中为 -15 | 0 |

真实 monitor 路径观察到了两种异常子进程退出，真实 probe failure 路径也触发了关闭，
但 `run_dp_supervisor()` 没有把故障转换为父进程状态。监督该父进程的 service manager
无法用退出码区分它们与有意关闭。

这是 Lab 已确认的薄生命周期边界结果，尚未形成 upstream issue 或修复。它与 #52178
刻意分开，因为 multi-port DP supervisor 使用不同的父进程生命周期。

## 四个案例共同改变了什么

这些误导性的信号在局部都是真的：

- 进程确实存活；
- health handler 确实返回了 2xx；
- dump 文件确实不存在；
- 父进程确实返回了 0。

错误发生在把局部观察提升为更强的系统结论时。解决方法不是“收集一切”，而是定义跨越
边界必须成立的关系，保留验证该关系所需的主体 identity 和时间窗口，并在事实不足时
拒绝下结论。

Lab 因此区分三层：

1. **Observation：**进程状态、health response、progress counter、stack、stage flag、
   child status；
2. **Relationship：**这些观察属于哪个主体和时间窗口，以及它们相互印证还是冲突；
3. **Claim：**可以离线重算的关闭 verdict；无法支持时返回 `undetermined`。

stack、GPU utilization 和诊断 artifact 可以加强 attribution，但不能静默替代 progress、
identity 或 demand。producer 缺失必须保留为显式事实，不能变成负面证据。

## 无需 GPU，重放一个完整证据回路

v0.2.0 release 将 #53859/#53883 结果封装成离线 replay：

```bash
git clone --branch v0.2.0 https://github.com/jackLei0901/vllm-runtime-reliability-lab.git
cd vllm-runtime-reliability-lab
python -m pip install .
vllm-dfx replay results/vllm-zmq-backpressure-stage1-r3-20260916
```

该命令验证已归档的文件集合、identity、四 cell base/fix contract、health 与 progress
观察、stack 结论和 event-loss 权衡。它不会重新运行 GPU 实验，也不代表生产可用。

真正有价值的质疑不是输出是否“看起来合理”，而是证据是否允许该 verdict、是否忽略了
反证，或者 verifier 本应返回 `undetermined`。

## 边界

- #53859/#53883 结果只覆盖单 GPU、单 EngineCore；
- #52178 结果验证生命周期行为，不代表 Lab 首先发现该 bug；
- DP supervisor Gate 0 使用精确 upstream 源码和真实进程/服务 primitive 验证薄边界，
  不是完整模型服务实验；
- #197232 尚无明确 upstream 结果，因此本文不把它描述为 upstream 成功；
- 这些结果都不证明自动根因分类或生产自动修复。

共同结论更窄，也更有用：进程、endpoint、artifact 或 exit code 可以各自真实，但从它们
推导出的系统结论仍可能错误。可靠性问题就存在于这条缝隙中。
