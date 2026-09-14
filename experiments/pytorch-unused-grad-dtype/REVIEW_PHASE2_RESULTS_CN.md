# Phase 2 GPU 结果：中文审核入口

状态：**Gate 0 通过；Gate 1 证据不足；Gate 1b/1c 已撤回；Gate 1d runner 停止；Gate 1e 严格采集门失败；Gate 1f shutdown-stage 诊断通过。**

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

本节记录的是 **Gate 1d 执行前** 的冻结状态。随后 Gate 1d 已启动并因 runner
解析问题停止，Gate 1e 也已完成；当前结果见下一节。在该机制门完成前不进入四卡
矩阵的约束，仍是当时正确的执行边界。

2026-09-13 的首次 Linux 远端预检在 GPU trial 启动前发现，原冻结清单对三个复用
文件记录的是 Windows CRLF 工作树哈希，而 Git 与 Linux checkout 使用 LF。此次只
纠正这三个哈希并用 `.gitattributes` 固定 LF；协议、reproducer、预期和结果均未
改动。完整审计见 `HASH_LINE_ENDING_CORRECTION_2026-09-13.md`。四套 freeze 必须在
干净 Linux checkout 中重新通过后，Gate 1d 才于 2026-09-13 开始执行。

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

## 九、Gate 1f：定位 rank 1 dump 缺失发生在哪个 shutdown 阶段

Gate 1e 的三次重复已经证明缺失模式稳定，不再重复同一测量。源码复核给出一个更窄、
尚未证实的解释：在 PyTorch 2.13.0+cu130 所带 NCCL 2.29.7 上，rank 1 可能先完成
`finalize()`、停止 heartbeat monitor，随后停在 communicator destruction。这样 rank 0
约 30 秒后广播 dump 请求时，rank 1 已没有存活的 Flight Recorder responder。

Gate 1f 只执行一次 affected trial，并声明唯一诊断改动
`TORCH_CPP_LOG_LEVEL=INFO`。runner 不保留原始日志，只把八条固定 PyTorch library
消息压缩成逐 rank 布尔值，同时在 preflight 中记录 `torch.cuda.nccl.version()`。
rank 前缀缺失或含糊时不抛出并丢失整次试验，而是写入有界
`library_log_scan_error`，保留其他结构化证据后由验证器失败关闭；不根据时间戳猜测
消息归属。

预注册的 rank 1 序列是：开始 shutdown、完成 flush、watchdog 已 join 且开始销毁
communicator 三项为 true；`Destroy complete.`、收到远端 dump signal、dump 成功三项
为 false。即使完全匹配，也只能支持该 shutdown-stage 解释，不能直接证明阻塞发生在
`ncclCommDestroy` 内部，更不能把 Gate 1e 的严格双 dump 失败改判为通过。

rank 0 也有冻结判据：成功广播 `exception_dump` 为 true、广播失败为 false、dump 成功
为 true，四个 shutdown 阶段均为 false。它既证明 dump 请求确实发给其他 rank，也让
`dump_success` 有正对照，并用 shutdown 阶段的全 false 防止 rank 归属颠倒。

审核入口：

1. `GATE1F_PROTOCOL.md`：诊断改动、固定消息白名单和解释边界；
2. `gate1f_campaign.py`：逐 rank 消息归属、NCCL 版本记录和单次 runner；
3. `verify_gate1f.py`：独立重算机制、终止、stack、dump-set 与阶段预测；
4. `GATE1F_FREEZE.json` 与 `verify_gate1f_freeze.py`：12 个运行依赖的预执行冻结。

### Gate 1f 实测结果

唯一一次 affected trial 在 commit `6db7069e916571a4e9a5f213bf63f881929a53a1`
的精确归档树上运行。六套 freeze 先以 15/7/7/9/10/12 文件全部通过。环境为双
RTX 4090、PyTorch `2.13.0+cu130`、NCCL 2.29.7 和 py-spy 0.4.2。

rank 0 成功广播 `exception_dump`、广播失败为 false、本地 dump 成功，四个 shutdown
阶段均为 false。rank 1 则依次记录 shutdown 开始、operations flushed、watchdog 已
join 并开始销毁 communicator；`Destroy complete.`、观察到远端 dump signal 和 dump
成功均为 false。`library_log_scan_error` 为 null，两侧冻结预测完全匹配。

机制仍为 rank 1 本地 dtype assertion、rank 0 barrier wait；终止仍达到 60 秒外部边界。
外部 stack 取得两个 rank，Flight Recorder 仍只有 rank 0 dump，生命周期无孤儿，原始
输出未保留。独立 verifier 通过。

因此可以确认：rank 0 的跨 rank dump 请求确实发出；rank 1 在停止 heartbeat monitor
之后、communicator destruction 完成之前停滞，无法响应稍后的请求。这支持提出一个
限定于 PyTorch 2.13.0+cu130 / NCCL 2.29.7 的 diagnosability issue，但没有直接证明
具体阻塞点一定是 `ncclCommDestroy`。Gate 1e 的严格双 dump 结论仍是 FAIL-CLOSED。

完整结果见 `GATE1F_RESULT_2026-09-13.md` 和
`results/pytorch-unused-grad-dtype-gate1f-20260913/affected-trial-1.json`。

## 十、Gate 1g：去掉 FSDP 后验证问题边界

Gate 1f 的结果仍含 FSDP、混合精度和 unused-gradient 路径。Gate 1g 不再扩充同一
FSDP 试验，而是构造一个独立的两 rank ProcessGroupNCCL 最小复现，回答更窄的问题：
当 rank 0 已经排队并等待异步 `all_reduce` 时，rank 1 发生本地异常并按文档进入
`destroy_process_group()`，相同的 dump-responder 缺口是否仍会出现。

两个 arm 都先由双 rank 完成一次同尺寸 warm-up all-reduce，并执行
`torch.cuda.synchronize()`，确保 communicator 及连接在分岔前已创建。affected arm
随后用原子文件在用户空间建立明确顺序：rank 0 先记录正式 all-reduce 已排队，再以
180 秒显式超时进入 `work.wait()`；rank 1 观察到该标记后才注入异常。这个标记只证明
本地事件顺序，不推断 rank 1 已参与正式 collective。control arm 则由两个 rank 正常
完成相同 all-reduce 和 teardown。

预注册期望沿用 Gate 1f 已验证的 20/30/60 秒节奏和逐 rank library-log 判据。affected
arm 预期 rank 0 stack 位于 `work.wait()`、rank 1 位于
`destroy_process_group()`；Flight Recorder 仍只有 rank 0 dump。新增的本地摘要只允许
声明“rank 0 的 dump 含 group `[0, 1]` 上恰好一个未完成的 ALLREDUCE”。warm-up 条目
必须已完成；零个或多个 pending 候选都失败关闭。候选构造只读取 rank 0 artifact，并
记录 `peer_participation_inferred: false`，不能用单 rank dump 声称另一 rank 参与了
collective。

本轮只安排 1 次 control 和 1 次 affected。任何 marker 或 library-log 解析问题均先
写入有界错误码再失败关闭；原始 torchrun、py-spy 与 Flight Recorder 内容不持久化。
若结果完全匹配，才具备把问题作为通用 ProcessGroupNCCL diagnosability gap 报告上游
的条件；仍不主张具体 NCCL 内部阻塞点，也不提出未经验证的修复方案。

执行前审核入口：

1. `GATE1G_REVIEW_START_HERE.md`：英文审核入口、已完成检查和待确认问题；
2. `GATE1G_PROTOCOL.md`：问题、受控改动、冻结预测和解释边界；
3. `gate1g_reproducer.py`：无 FSDP 的最小两 rank 复现；
4. `gate1g_campaign.py`：采集、清理、隐私边界和本地 FR 摘要；
5. `verify_gate1g.py`：独立结果验证器；
6. `GATE1G_FREEZE.json` 与 `verify_gate1g_freeze.py`：运行依赖哈希冻结。

### Gate 1g 实测结果

Gate 1g 从精确 commit `9f1c72e01616c0a59748e0b91690371da99fb3fd` 的归档树
运行；归档 SHA-256 为
`a05bbfd50ce8921f1de76002d1bd0a4ac62d2ff9472617389d9a41fb804dacd2`。
执行前七套 freeze 以 15/7/7/9/10/12/11 文件全部通过。环境为双 RTX 4090、
PyTorch `2.13.0+cu130`、NCCL 2.29.7 和 py-spy 0.4.2。

control 中两个 rank 完成 warm-up、正式 all-reduce 和正常 teardown，进程退出 0，
没有 stack 或 Flight Recorder dump。affected 中两个 rank 先完成 warm-up；rank 0
随后排队正式 all-reduce 并停在显式 180 秒 `work.wait()`，rank 1 观察到原子标记后
注入本地异常，并停在 `destroy_process_group()`。进程达到 60 秒外部边界。

rank 0 成功广播 dump 请求并生成本地 dump。该 dump 中恰好一个 group `[0, 1]` 的
本地 `ALL_REDUCE` 处于 `scheduled`；候选构造没有读取 rank 1 数据，也没有推断其是否
参与。rank 1 记录 shutdown 开始、operations flushed、watchdog 已 join 并开始销毁
communicator，但没有 `Destroy complete.`、没有收到稍后的 dump 请求、也没有生成
dump。逐 rank stack、library flags、解析、生命周期和隐私合同全部通过独立 verifier。

因此 Gate 1g 在去掉 FSDP、模型、混合精度、unused-gradient 和 accumulation 后，仍
复现了 Gate 1f 的 missing-rank dump 缺口。该结论仅适用于固定版本和本次边界实验；
不能直接证明 NCCL 内部的具体阻塞调用，也没有验证修复方案。完整结果见
`GATE1G_RESULT_2026-09-13.md` 和
`results/pytorch-unused-grad-dtype-gate1g-20260913/`。

审核后发现 `watchdog_marker_seen` 不能作为 Gate 1g 的 timeout 判据：INFO 日志中的正常
退出消息 `ProcessGroupNCCL watchdog thread joined.` 也会命中宽泛的
`"NCCL watchdog"` 子串，control 因此同样为 true。已执行的冻结代码保留不动。该修正
不改变 PASS，因为 rank 0 成功广播 dump 请求并产生可解码 dump，且 affected 达到外部
60 秒边界；后续试验改为逐 rank 精确匹配
`Watchdog caught collective operation timeout`。
