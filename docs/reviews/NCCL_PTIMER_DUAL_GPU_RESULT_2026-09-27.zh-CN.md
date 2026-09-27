# NCCL graph-profiler `pTimer`：一次双卡能力验证

英文版：[NCCL_PTIMER_DUAL_GPU_RESULT_2026-09-27.md](NCCL_PTIMER_DUAL_GPU_RESULT_2026-09-27.md)。本记录依据[预注册协议](NCCL_PTIMER_DUAL_GPU_PROTOCOL_2026-09-27.zh-CN.md)，不修改冻结的 v0.2 verdict 或此前的 `unscored / target_collective_ambiguous` occurrence 结果。

## 结果

两 rank 的 eager 仪器正对照通过；V2 serving 健康对照通过。唯一一次 hold 确实进入，窗口内请求待完成，放行后完整生成 16 token；但原 occurrence gate 仍判为 **`unscored / target_collective_ambiguous`**。独立 `pTimer` 摘要器在健康 serving 和 hold 窗口中，都观察到同一通信器/通道组内 GPU 时钟值重复。事后源码检查使健康对照的解释更强：普通强制 PyNccl TP 调用分别提交，因此 73 个不同 occurrence 不可能全部携带各自独立执行的设备时间戳。这是源码与回调的联合推断，不是 `base=0` 机制、stock 指标错误或独立 NCCL 缺陷的证明。

评分前有一次**启动配置失败**：固定 venv 内的 `ninja` 未加入 `PATH`，引擎未完成初始化。随后仅修正 `PATH`，没有改代码、阈值或插件，并用全新目录运行健康对照。没有第二次 hold。

## 固定身份和证据

| 项目 | 身份与状态 |
| --- | --- |
| GPU | 两张 RTX 4090；每格一次运行。 |
| 软件 | vLLM `c8602c79062440074a018c1d5f875a5571eb6881`；Torch `2.13.0+cu130`；实际加载 NCCL `2.29.7`。runner 的软件及映射库检查通过。 |
| NCCL 源码 | `v2.29.7-1`，`b91894bd5b190c874d98a017f93f5daa515b65d0`；新建 detached worktree。 |
| Inspector 补丁 | occurrence 补丁 SHA-256 `5cbaa80d97ecbaf79e388d74466b15522fd585c8b514953f289195c3cd2474aa`，随后 `pTimer` 增量补丁 SHA-256 `8c1fd0860ff35996b6af70cd13722f48152c8d9b689fce9b47bbbc30f513aabe`。两次 `git apply --check` 与 `git diff --check` 均通过。 |
| 本轮 Inspector 二进制 | SHA-256 `ef734dc33acd60af9edc43b8ec84204c70e3d1fde53294478dc5b08e0d352c3b`；编译器警告为零，未覆盖上轮二进制。 |
| 摘要器 | [固定源码](../../experiments/vllm-tp-dfx/ptimer_trace.py) SHA-256 `8400e4972e826a8b1b18633b7f298ad3e5673ba1e1da1f301af0cf3f2c255ebd`；远端上传文件复核一致。 |
| 健康对照 receipt | SHA-256 `daa4513c622482c7cc378fc2033c184d6847abcc6ace825af80b5a339c44b2c0`；记录 `healthy_v2_full_replay_observed` 并绑定本轮二进制。 |
| 私有证据归档 | SHA-256 `17d3c0a687e2d7f555591fbdc54dcc7512421fb1bb76305895420ba0b7849d3e`；64 个条目，权限 `600`。摘要在远端计算，未下载后独立复核。原始日志、GPU 时钟、通信器 ID 和进程身份均不公开。 |

eager、健康对照、hold 三份摘要 JSON 的 SHA-256 依次为 `86fa4f61f1fb5136ee999ea549b864483af41dc23fc6cd22c765c37324d2d3d3`、`3669c55f0ac1691c03dfd240cacfbedef35200b2d90453eb904791051f506d37`、`b619478e4fbece22391c27120bb0aacad30c00ec89791b880a5d9601f794a521`。各摘要均来自两个绑定 rank 的私有日志；serving 摘要还要求 before/end 快照摘要匹配且前缀一致。

本轮新归档含新补丁、摘要器、eager 脚本、构建日志、二进制和四个实验格目录。未修改的 v2 runner、parser、stall 插件及 activation witness 保存在上轮 occurrence 实验的另一个私有归档，其固定摘要列于协议。因此，新归档单独并非完整源码包。

## 闭合观察

| 实验格 | Rank 0 | Rank 1 | 关卡解释 |
| --- | --- | --- | --- |
| eager PyNccl | 6 对 start/stop；唯一通信器/通道组内无零值、重复或非递增时钟，无 `stop <= start`。 | 相同。 | 数值正确，仪器正对照通过；六次包含 PyNccl 初始化 warm-up 与五次显式调用。 |
| 健康 V2，after 减 before | start/stop 各 1,273；组内最大相等类均为 73；无零时钟或配对时长非正。 | start/stop 各 1,273；最大相等类 start 78、stop 77；无零时钟或配对时长非正。 | runner 报告 `healthy_v2_full_replay_observed`、请求完整、身份稳定。观察到时钟值复用，但不评分为错误计时。 |
| 唯一 hold 窗口，during 减 before | start/stop 各 164；组内最大相等类均为 58；无零时钟或配对时长非正。 | start/stop 各 148；最大相等类均为 26；无零时钟或配对时长非正。 | Rank 1 的 replay 前 hold 已进入，窗口内请求待完成、放行后完成；runner 仍为 `unscored / target_collective_ambiguous`。计数不能确定共同目标 collective。 |

分组相等类排除了不同通信器或通道之间恰好同值的混淆。[batch 内统一打点](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/device/common.h#L318-L335)只作用于一次 kernel launch 内的 work，不能解释普通强制 PyNccl TP 调用：[它们直接调用 NCCL API，没有显式 group](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/distributed/device_communicators/pynccl.py#L166-L213)，而 [NCCL 以每次隐式 group 提交](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/enqueue.cc#L2786-L2823)。73 个相等时钟来自同一通信器/通道的不同回调 occurrence ID；对这些普通 TP 调用而言，它们不可能全部是各自执行的设备 START（或 STOP）时钟。已发表摘要没有保存相等类的函数构成及 stream 映射，故这是**事后、限于该路由的推断**，不追溯改动预注册评分。相邻时钟逆序仍需绑定 occurrence 顺序与实际 stream 才能评分。本轮未见 `stop <= start`，与 START/STOP 都读取同一早期槽位的有效时间对相容。Rank 1 在当前 replay 前被 hold，也不能仅凭它的回调计数推断当前 replay 的设备进度。

## 决策与下一关

健康对照支持更窄但更强的结论：在这条强制 PyNccl 路由上，**不是每个回调时钟都属于其命名的、分别提交的 collective 执行**。尚不能确定时钟实际来自哪个 work、是否发生了 `base=0`、stock Inspector 的 `coll_exec_time_us` 或带宽是否错误，也不能据此断言现有 producer 无法导出正确设备进度。hold 的 occurrence 结果仍为 `unscored / target_collective_ambiguous`：重复时钟不能选出目标 collective。本轮不纳入 Lab probe，也不足以单凭 serving 运行提交 NCCL issue。

下一步先取回私有归档并独立复核摘要，再做**事后**离线审计：按通信器/通道及函数统计共享 START/STOP 时钟的不同 occurrence；统计 hold 窗口内低于窗口前最大时钟的值；只有绑定顺序和 stream 后才统计相邻逆序。这些计数**尚未计算**，不能作为已观察结果。归档仍在已关机的远端；上述远端摘要不等于下载后复核。

重编 NCCL core 之前，先做最小双 rank 独立复现：用有界设备工作隔开的、未显式分组的 all-reduce，分别 eager 执行与 CUDA graph replay，并用同一 Inspector `pTimer` 仪器观察。若 eager 时钟各异且递增、graph replay 重复，即可在不依赖 vLLM 的情况下证明回调症状。若仍需确定 `base`、planner 状态或槽位身份，单独标识的 NCCL-core 插桩才作为**可选的机制确认**。若要主张 stock 指标错误，还需把完成记录绑定 occurrence，并有独立计时参照。下次开卡前须在选定 venv 的 `PATH` 中检查 `command -v ninja`，不要等到第一个 serving 格才发现缺失。
