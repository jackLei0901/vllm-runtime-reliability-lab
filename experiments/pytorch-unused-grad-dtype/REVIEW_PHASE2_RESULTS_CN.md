# Phase 2 GPU 结果：中文审核入口

状态：**Gate 0 通过；Gate 1 证据不足；Gate 1b/1c 已撤回；Gate 1d runner 停止；Gate 1e 严格采集门失败。**

本文件是执行后的审核入口。实验前的预期、脚本和停止规则仍保存在
PHASE2_FREEZE.json 所固定的 15 个文件中，没有根据结果改写。

## 一、先核查预执行冻结

运行 verify_phase2_freeze.py，正式执行前和结果返回后均得到：

    PASS (15 frozen Phase 2 files)

完整实验前协议见 PHASE2_PROTOCOL.md。

## 二、Gate 0：PASS

环境：

- 2 x NVIDIA GeForce RTX 4090；
- PyTorch 2.13.0+cu130；
- PyTorch git revision cf30153c4c131c8164ee7798e5022d810682e2cb；
- CUDA runtime 13.0。

结果：

| case | 重复 | 两个 rank 实际进入 foreach_reduce 的 dtype | 退出结果 |
| --- | ---: | --- | --- |
| unused-parameter | 3/3 | 全部为 bf16，每个 rank 四个梯度槽 | exit 0 |
| forced-mixed-gradient | 3/3 | 每个 rank 均为 bf16 + fp32 | exit 1，命中精确 assertion |

验证器输出：

    PASS (2 cases x 3 trials; torch=2.13.0+cu130)

审核重点：

1. 普通 unused 参数路径没有产生混合 dtype。真实梯度和
   zeros_like(unsharded_param) placeholder 都是 bf16；
2. reduce_dtype=fp32 的转换发生在 uniform-dtype 检查之后，因此它不会
   单独制造该 assertion；
3. 正控制是主动制造 bf16+fp32，用于证明探针和分类器有效。它不是 PyTorch
   缺陷证据；
4. 6/6 均记录两个 rank 身份、不保留原始输出、无追踪进程遗留。

详细说明见 GATE0_RESULT_2026-09-12.md，六份结构化记录位于
results/pytorch-unused-grad-dtype-gate0-20260912/。

## 三、Gate 1：STOP

Gate 1 只增加四个 microbatch 的梯度累积，不增加 GPU，也不引入 PP：

- control：3/3 completed；
- affected trial 1：达到 60 秒上限，分类为 timeout；
- 冻结预期是快速出现精确 dtype assertion，因此 timeout 是失败结果，不是复现；
- 按停止规则中断后续运行，没有继续凑满三次；
- 当时 affected trial 2 已启动但未完成，明确排除；
- 中断后无遗留进程、无 GPU compute process。

这次还暴露了 runner 的一个证据保留缺陷：操作员中断下一次 trial 后，顶层清理
删除了不完整目录，因此 Gate 1 没有可发布的结构化结果集。不能根据终端输出把它
提升为与 Gate 0 同等级的证据。

需要特别注意：timeout 分类的优先级高于 assertion marker，而原始输出已经按
隐私规则丢弃。因此目前不能声称“没有 assertion”，也不能排除某个 rank 先在本地
assert、另一个 rank 随后卡在 collective 的情况。

源码给出了一个比“组内有 fp32 参数”更贴近现有证据的预测：

1. 禁止梯度同步的 microbatch 会通过 to_accumulated_grad_if_needed，把已有的
   真实梯度转换为 reduce_dtype，也就是 fp32；
2. affected 的 rank 1 始终不使用 conditional 参数，所以该参数没有 accumulated
   gradient，最终补入的 placeholder 仍跟随 unsharded parameter，为 bf16；
3. rank 1 可能以 fp32+bf16 在 uniformity check 处 assertion；
4. rank 0 两组参数都有 fp32 accumulated gradient，可能进入 reduce-scatter 后
   等待已经退出的 peer。

这个机制能解释 timeout，但本轮没有 accumulated dtype probe，仍需新版本实测。

详细偏差记录见 GATE1_STOP_2026-09-12.md。

## 四、当前结论

已经得到支持：

- 复核指出的 dtype 机制解释成立：首轮两卡 null 是因为进入检查点的梯度
  实际上全是 bf16；
- 探针和 classifier 能稳定识别真正的 bf16+fp32 assertion。

尚未得到支持：

- 普通 unused 参数本身会触发 PyTorch dtype bug；
- microbatch 累积已经通过可审计证据复现目标 assertion；
- pipeline parallelism 或四卡是必要条件；
- Gate 1 的 timeout 与原 organic workload 是同一种故障。

## 五、下一步审核问题

下一次开卡前，应先离线准备并审核新的诊断修订：

1. 把同一 dtype probe 放进 accumulated 路径，并预注册 rank 0 全 fp32、
   rank 1 为 fp32+bf16 的预测；
2. 首次出现与冻结预期不符的分类时自动停止，并保留已完成 JSON；
3. 记录每个 rank 是否进入 post-backward、进入次数及 dtype 集；
4. 将“rank-local assertion”“collective次序/数量不一致”和“hook 未触发”分开；
5. 新修订重新冻结，不能覆盖本轮 PHASE2_FREEZE.json。

在这些完成前，不建议继续租两卡，也不建议升级到四卡。

## 六、对“非统一参数 dtype”建议的处理

不把 param_dtype 设为空、再人为构造 fp32 与 bf16 参数组，可以作为独立的 FSDP
输入契约测试，但它不能解释当前 organic workload：原 workload 明确配置
param_dtype=bf16。并且 Gate 0 的正控制已经证明，bf16 parameter 也可能产生
fp32 real gradient；所以从 fp32 gradient 反推 fp32 parameter 并不成立。

当前主线应先验证 accumulation-upcast 与 unused placeholder 的组合，而不是更换
为一个与 organic 配置不同的混合参数模型。

## 七、Gate 1b/1c 撤回与 Gate 1d 审核入口

Gate 1b 已在执行前撤回。源码复核确认，`reduce_scatter_single(async_op=False)`
的 CPU 调用可能在建立 stream 依赖后返回；rank 0 真正的 CPU 等待点是随后的
`dist.barrier()`。因此 Gate 1b 要求“rank 0 reduce 不返回”的冻结条件不可能与
预期机制同时成立。旧冻结保持不变，只作为审计记录，不能执行。

Gate 1c 保留了正确的机制矩阵，但也在执行前撤回。它的 45 秒 wall bound 给两份
Flight Recorder dump 留出的写盘余量太小；同时，它会在终止预测不匹配时停止，
与协议中“机制和终止分别评分”的规则矛盾。旧冻结与撤回说明均保留为审计记录。
需要注意，Gate 1d 仍复用并冻结了 `gate1c_campaign.py`、`verify_gate1c.py` 中的
公共采集、分类和验证逻辑；它们虽属于已撤回协议，当前仍是 Gate 1d 的运行时依赖，
不能作为废弃文件清理或修改。

Gate 1d 沿用该机制矩阵：

| arm | rank | `foreach_reduce` dtype | reduce | barrier/outcome |
| --- | ---: | --- | --- | --- |
| control | 0/1 | all fp32 | 均返回 | barrier 返回并完成 |
| affected | 0 | all fp32 | 返回 | 进入 barrier，不返回 |
| affected | 1 | fp32 + bf16 | 不返回 | exact uniformity assertion |

终止路径单独评分。当前预注册预测是：rank 1 在 assertion 后进入
`destroy_process_group()` 但不返回，rank 0 尚未进入 teardown，launcher 最终达到
60 秒外部 wall bound。如果 torchrun 或 watchdog 更早结束任务，只记录终止预测
不匹配并继续完成三次 affected trial；只有机制预测不匹配才停止 campaign。
这一自动继续规则只用于 affected arm。若 control trial 出现超时、非零退出或其他
终止异常，操作者应立即停止 campaign，保留已有结果并把手工停止记录为
`operator deviation`；无需为这项不影响验证器 fail-closed 行为的限制另开冻结版本。

采集门也被收紧：20 秒时必须同时读取 rank 0 和 rank 1 的 stack，分别显示
`dist.barrier()` 与 `destroy_process_group()`；Flight Recorder 必须有两份可解码
rank dump，并在 `[0, 1]` group 的同一 logical sequence 上显示只有 rank 0 的
未完成 reduce-scatter。缺少一份 dump 与 rank 未参与 collective 分开记录，单份
dump 产生的通用 `missing_member` 不再算通过。

ptrace preflight 记录 effective UID、`ptrace_scope` 和授权模式。存在 Yama 时，每个
rank 通过 `PR_SET_PTRACER` 授权 campaign parent 及其 py-spy 子进程；不存在 Yama
时记录 `yama_absent`，不调用会返回 `EINVAL` 的 prctl。原始 launcher 输出、stack
JSON 与 Flight Recorder pickle 仍只存在于临时目录。

结果验证会先比较每份结果中的 `reproducer_sha256` 与当前 reproducer，再从源码
推导预期 stack 行号，避免运行后修改脚本导致验证器静默改变判据。

审核顺序：

1. `GATE1B_WITHDRAWAL.md`：Gate 1b 为何不能执行；
2. `GATE1C_WITHDRAWAL.md`：Gate 1c 为何在执行前撤回；
3. `GATE1D_PROTOCOL.md`：当前机制、终止、采集矩阵和停止规则；
4. `gate1d_reproducer.py`：Yama 条件授权和逐 rank marker；
5. `gate1d_campaign.py`：20/30/60 秒采集、非阻塞终止偏差和严格 FR join；
6. `verify_gate1d.py`：含 reproducer hash 前置检查的独立验证；
7. `GATE1D_FREEZE.json` 与 `verify_gate1d_freeze.py`：当前预执行冻结。

当前仍是 **pre-execution**。Gate 1d freeze 通过后才可租两卡运行；在该机制门完成
前不进入四卡矩阵。

2026-09-13 的首次 Linux 远端预检在 GPU trial 启动前发现，原冻结清单对三个复用
文件记录的是 Windows CRLF 工作树哈希，而 Git 与 Linux checkout 使用 LF。此次只
纠正这三个哈希并用 `.gitattributes` 固定 LF；协议、reproducer、预期和结果均未
改动。完整审计见 `HASH_LINE_ENDING_CORRECTION_2026-09-13.md`。四套 freeze 必须在
干净 Linux checkout 中重新通过后才能执行 Gate 1d。

## 八、Gate 1d 停止与 Gate 1e 结果

Gate 1d 在 control trial 1 正常完成后，于 control trial 2 解析两个 rank 相邻写入的
JSON marker 时触发 `JSONDecodeError: Extra data`。runner 自动停止，没有执行任何
affected trial。该失败及唯一完整 control summary 已保留，不能算作机制结果。

Gate 1e 只修复相邻 marker 的独立解码，并把 control 终止异常改为自动 fail-fast；
机制矩阵、20/30/60 秒时序、reproducer、严格双 dump 门和解释边界均未改变。Linux
干净 checkout 的五套 freeze 以 15/7/7/9/10 文件全部通过后，双 RTX 4090 正式执行：

| 项目 | control | affected |
| --- | --- | --- |
| 重复次数 | 3 | 3 |
| 机制 | 3/3 对称 fp32 并完成 | 3/3 rank 1 assertion、rank 0 barrier wait |
| 终止 | 3/3 正常 | 3/3 frozen wall-bound path |
| stack | 未要求 | 3/3 同时取得两个 rank |
| Flight Recorder | 0，符合预期 | **每次只有 rank 0；双 dump 0/3** |
| 生命周期 | 3/3 无孤儿 | 3/3 无孤儿 |

因此机制门、终止门和 stack 子门通过，但严格 capture 门失败，总结论按预注册规则为
**FAIL-CLOSED**。现有证据不能把“rank 1 没参与 collective”和“rank 1 的 dump 缺失”
区分开，不能宣称完成跨 rank Flight Recorder join。三次缺失模式完全相同，不再用
GPU 重复扩充样本。

审核入口：

1. `GATE1D_EXECUTION_STOP_2026-09-13.md`：Gate 1d runner 停止；
2. `GATE1E_PROTOCOL.md` 与 `GATE1E_FREEZE.json`：Gate 1e 的预执行契约；
3. `GATE1E_RESULT_2026-09-13.md`：结果、限制和下一步；
4. `results/pytorch-unused-grad-dtype-gate1e-20260913/`：六份 allow-listed summary。
