# NCCL graph profiler `pTimer` 对照——一次双卡预约

英文版：[NCCL_PTIMER_DUAL_GPU_PROTOCOL_2026-09-27.md](NCCL_PTIMER_DUAL_GPU_PROTOCOL_2026-09-27.md)。

## 问题与边界

在 graph replay 中，不同 collective occurrence 的 Inspector 回调是否拿到了重复的 device 时钟值？尤其是在 peer 尚未 replay 的限时窗口内，rank 0 的 `KernelChStop` 是否携带复用的 `pTimer`？这检验[计数器生命周期源码假说](NCCL_GRAPH_PROFILER_COUNTER_LIFECYCLE_SOURCE_REVIEW_2026-09-27.zh-CN.md)，不把回调当作当前设备进度证明，不从重复时钟直接推断 stock 带宽错误，不改变冻结的 `unscored / target_collective_ambiguous` 结果，也不自动构成上游 issue。成本更低的 stock JSON 离线关卡已经做过：重复序列层的每记录 CPU 回调时差在两 rank 均较短，但 Inspector 只导出完成记录，无法证明 graph 位置覆盖或 device-clock 正确性。本轮验证的是当时缺失的设备时钟，不是跳过该关卡。

NCCL 源码先调用 `recordEventState(..., ncclProfilerKernelChStop, pTimer)`，再调用 `stopEvent`；Inspector 在前者保存 `stopGpuClk`。所以现有 `kernel_ch_stop` 日志点可以读取该值。新增[增量补丁](../../experiments/vllm-tp-dfx/inspector-ptimer-delta-after-occurrence-v2.29.7-1.patch)只在原 occurrence 日志末尾追加起始/结束 GPU 时钟。原值保留在私有 NCCL 日志；冻结的 v2 runner/parser 因 `EVENT_LINE` 接受 `channel=` 后的空白而忽略尾部字段，CPU 回归测试会输入带 `ptimer=` 的日志行。[独立离线摘要器](../../experiments/vllm-tp-dfx/ptimer_trace.py)只公开按通信器/通道分组的相等类计数、配对时长非正的计数及 SHA-256 身份。

**不能把 `start_(i+1) < stop_i` 计为旧时钟的违例。** 固定 NCCL 版本的[设备侧 profiler](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/device/common.h#L318-L335)先为一个 batch 的全部 `nWorks` 写 START，再执行整批、写 STOP（[kernel loop](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/device/common.h#L393-L409)）。即使每个时钟都是真实设备时钟，两个 work 的区间也可能在此表示下重叠。sm_89 不支持 programmatic dependent launch 不能排除 NCCL 批处理。此外，[PyNccl 调用](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/distributed/device_communicators/pynccl.py#L234-L268)可接受显式 stream，否则采用当前 stream；整个 serving 运行不能仅凭通信器身份推断单一 stream。摘要器的相邻时钟计数仅供描述，不是健康 graph 的通过/失败判据。

## 固定输入与本机检查

- NCCL 源码固定为 `v2.29.7-1`、提交 `b91894bd5b190c874d98a017f93f5daa515b65d0`。先应用已有 occurrence 补丁（SHA-256 `5cbaa80d97ecbaf79e388d74466b15522fd585c8b514953f289195c3cd2474aa`），再应用本轮增量补丁（SHA-256 `8c1fd0860ff35996b6af70cd13722f48152c8d9b689fce9b47bbbc30f513aabe`）。不得覆盖上轮验证的 Inspector 二进制。
- 离线摘要器 SHA-256：`8400e4972e826a8b1b18633b7f298ad3e5673ba1e1da1f301af0cf3f2c255ebd`。eager 脚本、v2 runner、parser、stall 插件和 activation witness 传到远端后也必须核对固定摘要。runner 的 control receipt 已绑定健康/hold 所用二进制、软件身份和 parser；新增补丁的摘要还需单独写入私有 host manifest，不能宣称旧 receipt 已证明它。
- 按顺序固定的文件 SHA-256：`ptimer_eager_control.py` `559720b1270f4ac3832ffa68acbea526dd236e339520002f642cdb8c31ec374f`；`serving_v2_stall_gate.py` `b120f521581de6624be99aaad8fed70a33d3d98172fda23903f8f9d5d3b7c58c`；`inflight_trace_v2.py` `2770b3ad3157fee56409da1416b09101ff1c3c4ddfeb5fd6913865fabef8ded8`；`llr_tp_v2_stall.py` `37ebb91273d73285c2c2352a5f7c347a9ffa22c13c16b59183e71ec193ddc067`；`v2_activation_witness.py` `393a2d3e14442d70d97c4caef85aa8f14c15168b7de9e0c9353c74e73f3aa900`。
- 本机 CPU 检查：两个补丁依次应用到固定 Inspector 源码的干净归档后，在 WSL2 `g++` + CUDA 12.8 开发文件下编译成功，编译器警告为零。这仅证明该环境里的语法/构建兼容，**不是**远端加载或 NCCL 行为验证。增量补丁对已有 occurrence-patched checkout 的 `git apply --check` 通过。分组摘要修改后，本机完整测试为 309 项运行、OK、15 项跳过；摘要器/parser/runner 定向子集 22 项、OK。
- 目标运行环境与[上轮单次预约 runbook](VLLM_TP_V2_OCCURRENCE_ONE_BOOKING_RUNBOOK_2026-09-27.md)一致：两张 GPU、vLLM `c8602c79062440074a018c1d5f875a5571eb6881`、PyTorch `2.13.0+cu130`、实际加载 NCCL `2.29.7`、相同模型、V2 cached-FULL 路径和强制 PyNccl。任何差异先登记为环境偏离，不能默认为等价。

## 预注册实验格与停止条件

每格使用新建、仅所有者可读的私有目录和同一个新编译插件。构建/加载预检限时 30 分钟，**不计入**各实验格 180 秒外部超时。运行期间不改 parser。完整 NCCL 日志、receipt、before/during/after 快照、构建清单与摘要只在私有归档保留；不公开原始 GPU 时钟、通信器 ID、PID、路径、主机名或模型输出。

**运行后补充，不属于冻结评分：**今后预约在启动第一个 serving 格前，须先把所选 venv 的 `bin` 加入 `PATH`，再检查 `command -v ninja`。本轮首次尝试因缺少这一预检而在引擎初始化时失败；修正 `PATH` 后使用新目录运行评分对照。

1. **主机及加载预检。** 核对两卡、CUDA toolkit 头文件和库、`g++`、`make`、模型与磁盘空间、传输文件摘要。固定 NCCL checkout 中按顺序应用两补丁并构建，计算 `.so` 摘要。确认两个 rank-bound NCCL 日志，start/stop occurrence marker 都带 `ptimer=`，并确认各 worker 只映射一个 NCCL 库。构建、加载或身份失败时，在评分格前停止。
2. **eager 仪器正对照。** 双 rank 执行[五次 eager PyNccl all-reduce](../../experiments/vllm-tp-dfx/ptimer_eager_control.py)；PyNccl 构造器还会做一次预热 all-reduce。要求数值正确，各 rank 的 `largest_paired_group_event_count >= 6`、没有零时钟、任一至少含两个事件的通信器/通道组内无重复或非递增时钟、配对的 `stop <= start` 计数为零。此格用[摘要器](../../experiments/vllm-tp-dfx/ptimer_trace.py)的 `--end-label all` 模式。严格递增要求仅适用于此脚本中分别同步的 eager 调用，**不得**迁移到 graph 对照。失败表示仪器未验证，不进入 serving。
3. **V2 graph 健康对照。** 不改 `serving_v2_stall_gate.py --mode control` 及其环境契约。要求 `healthy_v2_full_replay_observed`、回调亲子关系有效、请求完整、生成一次性 receipt。再用新摘要器统计 `callbacks-after.json` 减 `callbacks-before.json`，分别记录各 rank 的 start/stop 数、零值数、各通信器/通道的相等类和配对的 `stop <= start` 计数。相邻区间重叠或重复值**不是**这里的缺陷判据：NCCL 可批处理多个 work、其他 device work 可改写复用槽位，且 serving stream 边界尚未证明。
4. **一次限时 hold。** 仅在两项对照通过后，按旧 runbook 使用原 runner 的 `--mode hold` 和准确的 control receipt 跑一次。只有 hold 确认进入、请求在窗口内待完成、身份稳定且放行后完成，才能解释 `callbacks-during.json` 减 `callbacks-before.json`。只报告 rank 0/1 的 start/stop 闭合计数及相等类。rank 1 未 replay 是 host witness，不是 rank 0 精确 device 状态的证明。本次预约不追加第二次 hold、不改阈值、不现场修 parser。

摘要器使用 `--private-dir`、`--end-label after|during` 和对应 before/end 快照文件的 SHA-256；缺时钟、重复 marker、快照非前缀或 rank 绑定歧义均拒绝。`--end-label all` 仅用于 eager 仪器检查，不需要快照摘要。其 JSON 是闭合形状，连同原始数据私有保存；公开前仅摘取经审阅的计数/摘要。`nonincreasing_adjacent_pairs` 在 eager 格是仪器检查字段，在 graph 格仅供描述；`nonpositive_pair_duration` 是单个 occurrence/channel 的单侧异常计数，不能作为完整反驳检验。

## 运行前固定的解释

| 观察 | 允许的结论 | 后续动作 |
| --- | --- | --- |
| eager 对照未通过限定为独立同步调用的时钟检查，或 v2 健康对照失败 | 装置或路由不可评分 | 停止并保留证据；离线修复后另约。 |
| 健康 graph 中出现较大的时钟复用相等类 | 完成态 serving 中观察到 device-clock 值复用；尚不是错误指标结论 | 必须绑定准确 occurrence/channel、stock Inspector 指标，并与独立计时对照，才可主张用户可见错误。 |
| 有效 peer hold 期间，rank 1 尚未 replay，多个不同 rank-0 occurrence 共享相同 start/stop 时钟 | 在该版本与路由下支持旧槽值解释 | 提交狭窄 NCCL 报告前还需独立双 rank graph 及计数轨迹。 |
| hold 期间时钟不相同，不论相邻计数是否变化 | 不反驳源码路径；其他工作可能改写槽位，批处理可使 work 时钟区间重叠 | 若仍值得追，另建有独立身份的 NCCL core 实验构建。 |
| hold 窗口不准确，或快照无法和日志匹配 | 不可评分 | 不用最终聚合日志代替，不挑有利子集。 |

即使相等类模式符合预测，这仍只是**回调数据**，不是当前 collective 的设备执行证明，也不是 `coll_exec_time_us` / 带宽正确性结论。健康 graph 无时钟异常**不能**反驳源码路径：回调读槽位之前它可能已被改写；跨 occurrence 区间重叠也**不能**证明该路径：NCCL 本身会批处理 work。本次预约只能给出支持或未定，无法对该假说作对称的支持/反驳裁决。本轮不纳入 Lab probe、不直接提上游。用户可见错误必须把完成记录绑定到 occurrence，并有独立计时对照；更广义的计数器修正仍是源码层设计假说。
