# NCCL graph-profiler `pTimer`：下一次双卡预约验证关卡

英文版：[NCCL_PTIMER_STANDALONE_ONE_BOOKING_GATE_2026-09-27.md](NCCL_PTIMER_STANDALONE_ONE_BOOKING_GATE_2026-09-27.md)。本文件预先固定**下一次**预约，不改写[已完成的 serving 结果](NCCL_PTIMER_DUAL_GPU_RESULT_2026-09-27.zh-CN.md)。本文尚无新的 GPU 观察。待检验的是 CUDA graph replay 中，分别提交、未显式分组的 NCCL collective 回调时钟是否属于其命名的执行；不是 vLLM hang、`base=0` 的直接证明或 stock Inspector 计时/带宽错误。

## 固定输入与边界

- Linux 双 RTX 4090、两 `torchrun` rank；vLLM `c8602c79062440074a018c1d5f875a5571eb6881`，Torch `2.13.0+cu130`，实际加载 NCCL `2.29.7`。NCCL `v2.29.7-1` 的 Inspector 依次使用 occurrence 补丁 SHA-256 `5cbaa80d97ecbaf79e388d74466b15522fd585c8b514953f289195c3cd2474aa` 与 `pTimer` 增量补丁 SHA-256 `8c1fd0860ff35996b6af70cd13722f48152c8d9b689fce9b47bbbc30f513aabe`。只有二进制 SHA-256 为 `ef734dc33acd60af9edc43b8ec84204c70e3d1fde53294478dc5b08e0d352c3b` 才可直接复用，否则在全新固定源码 checkout 构建，并在实验格前登记新身份。
- 上轮私有归档的**远端** SHA-256 为 `17d3c0a687e2d7f555591fbdc54dcc7512421fb1bb76305895420ba0b7849d3e`；实际路径只保存在私有运维记录。下载后、解包前独立复核摘要，并检查成员不得含绝对路径、`..` 路径穿越或链接。解包到新建、仅所有者可读目录。归档缺失或摘要不符时，旧数据审计记为不可用；独立实验仍可继续。
- 新脚本 SHA-256：`ptimer_posthoc_audit.py` `5dd5e7ac7cfcf0bc964d00869a115ca477fe43dcc200691082c3155a77c9a76f`；`ptimer_standalone_replay.py` `ca6f2b7c073266f306a7e87e3df90d31a97598fef18ee78b9aaf3552dd2ce0c8`；`ptimer_standalone_score.py` `3264be8616436aedf522991557f9515dc675a58b55649c8aeb7993417fadce99`。依赖 `ptimer_trace.py` `8400e4972e826a8b1b18633b7f298ad3e5673ba1e1da1f301af0cf3f2c255ebd` 与 `inflight_trace_v2.py` `2770b3ad3157fee56409da1416b09101ff1c3c4ddfeb5fd6913865fabef8ded8`。传输后重新核对，不在远端改脚本。runner 计算每个 rank 映射的 NCCL 库摘要，若两者不同即拒绝。
- [有界运行脚本](../../experiments/vllm-tp-dfx/run_ptimer_standalone.sh) SHA-256 为 `eeac5c73f16a65bbf7d6843fd5721d22470c441e4857f1276f238d16bc3c1779`；它在 graph 前核对五份源码、插件摘要、venv 中 `ninja` 的可用性及 eager 分数。私有路径与环境身份确定后调用：`bash run_ptimer_standalone.sh VENV_DIR PLUGIN_SO PRIVATE_BASE LAB_ROOT`。如需新编译插件，必须在运行前记录其摘要并设置 `LLR_EXPECTED_PLUGIN_SHA256`，不在远端修改 driver。driver 不负责封存私有证据或关机，第五步仍必须单独执行。
- 仅一个 eager 格、一个 graph 格，均使用新建私有目录、独立两 rank 进程、相同二进制与软件。无需模型权重。graph 捕获三次直接调用 `PyNcclCommunicator.all_reduce`，中间插入有界 `torch.cuda._sleep(1_000_000)`，随后以不同输入 replay 两次并核对数值。eager 格直接调用三次，每次同步并核对。构造器预热和 graph capture 均由私有 before/after 快照排除。它不是 transport hang 或 kernel 性能基准。

## 一次预约的顺序

1. **只读预检，最多 30 分钟。** 核对两卡、空间、选定 venv 的 Python/`torchrun`、准确 vLLM/Torch 版本、实际加载的 NCCL 库、`g++`、CUDA 头文件、`make`，以及选定 venv 的 `bin` 加入 `PATH` 后 `command -v ninja`。SSH 主机指纹须与用户独立提供的值相符。先哈希现有 Inspector，只有缺失或不符才重编。构建/加载失败属于实验装置结果，不是 profiler 结论；现场不改源码和阈值。
2. **旧归档完整性及事后审计。** 下载后独立核验上列摘要，安全解包；定位健康与 hold 格及其摘要固定的 before/after/during 快照。`ptimer_posthoc_audit.py` 各跑一次健康 `after` 和 hold `during`，从私有 manifest 传入快照摘要。脚本强制校验 rank、快照前缀及每个被选 channel 事件的时钟标记。仅公开每 rank 闭合计数：同一 `(通信器、通道、函数、事件类型)` 中共享时钟的不同 occurrence ID，以及窗口内低于同通信器/通道/事件类型历史最大值的时钟数。这是**事后**分析，不追溯改旧协议评分。按函数分层后，先前的 73 相等类可能缩小，必须照实记录。相邻逆序在 stream/顺序绑定前不评分。
3. **eager 仪器正对照。** 新目录；`NCCL_PROFILER_PLUGIN` 指向摘要合格的 `.so`，使用 `NCCL_DEBUG=TRACE`、`NCCL_DEBUG_SUBSYS=INIT,PROFILE`、`NCCL_DEBUG_FILE=<格目录>/nccl.%p.log` 及与上轮一致的 Inspector 启用/详细日志设置。两 rank 命令外部限时 180 秒。runner 保存 rank 绑定的 before/after 快照，只输出摘要与数值通过。用 `ptimer_standalone_score.py --mode eager` 评分：每 rank 恰好新增三次 AllReduce、每个 channel 的 START/STOP 配对完整、无零值/非正时长、时钟各异且严格递增、同 channel 无区间逆序。失败即**停止，不运行 graph**。
4. **graph 格。** 新目录、同一运行环境和插件身份、外部限时 180 秒。运行 `ptimer_standalone_replay.py --mode graph --sleep-cycles 1000000`；要求每 rank 新增六次 AllReduce、两次 replay 数值正确、before/after 快照均有摘要。用 `ptimer_standalone_score.py --mode graph` 评分。见结果后不调参、不重跑。
5. **封存与停止。** 原始日志、快照、命令、环境/构建清单及评分 JSON 进入新建的私有、仅所有者可读归档；哈希、复制离机、对复制件复核。公开材料只保留闭合计数、摘要与软件身份。须在用户授权下关机，并另行核对云平台电源状态；SSH 断线不能证明计费已停止。

本机准备关卡：`PYTHONPATH=src python -m unittest discover -s tests -q` 为 **315 项 OK、15 项跳过**；其中定向 pTimer 10 项通过。此结果只证明 CPU/脚本逻辑；`compileall` 通过，本机无 `ruff`。传输前记录经过审阅的提交身份，同时在远端逐项复核文件摘要；两种身份不可互相替代。

## 冻结解释

| 首个适用结果 | 允许的结论与动作 |
| --- | --- |
| 旧归档缺失或摘要不符 | 旧数据审计不可用，不引用新事后计数；独立格可继续。 |
| eager runner/评分无效或 `eager_instrument_failed` | 仪器未验证，不解释 graph 时钟；本轮实验停止。 |
| graph runner 无效、缺 channel、rank/occurrence 不一致或超时 | `unscored` 装置结果，不支持也不反驳；不挑选有利子集。 |
| eager 通过；graph 两 rank 均为 `graph_callback_clock_reuse_observed` | 在该版本/路由上独立复现回调症状；不等于证明 `base=0` 或 stock 指标错误。准备源码与复现材料；是否提上游仍单独决定。NCCL-core 插桩仅作为可选机制确认。 |
| eager 通过；graph 为 `one_rank_only_unscored` | 未形成双 rank 结论，保留两边完整轨迹，不挑选单 rank。 |
| eager 通过；graph 为 `graph_reuse_not_observed` | 本次独立格未复现；不抹掉 serving 观察。离线比较路由、捕获图与回调覆盖，再决定是否需要新预约。 |

旧数据中的“低于历史最大时钟”若非零，支持错误归属假说，但仅限通过校验的快照窗口；为零也不能反驳复用，因为 START/STOP 可能都读到更早但有效的槽位时间对。任何结果都不提升原 serving hold 的 `unscored / target_collective_ambiguous`，也不据此准入 Lab probe。
