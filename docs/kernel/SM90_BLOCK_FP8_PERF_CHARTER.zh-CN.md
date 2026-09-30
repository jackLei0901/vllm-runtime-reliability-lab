# SM90 block-FP8 GEMM decode 性能：项目章程

状态：2026-09-30，v0.7 同卡准备与冻结材料；未公开冻结，未做性能测量。以中文版为准；[English](SM90_BLOCK_FP8_PERF_CHARTER.md) 为译文。这是继[契约修复](SM90_BLOCK_FP8_PROJECT_REQUIREMENTS.zh-CN.md)之后第二个由 Lab 自主负责的底层项目。本项目首先是测量项目；只有预先确定的差距门槛通过后，才尝试一项候选修改。该候选失败只结束本实验，不结束性能方向。

## 1. 问题

在一张 SM90 GPU 上，针对一个 block-FP8 模型在 decode 规模（M = 1–64）下的 linear 层：

1. **Q0 工作负载：** 每个 block-FP8 linear 层实际执行哪个 kernel？这些 kernel 占 decode GPU 时间的比例是多少？
2. **Q1 基线：** 在明确说明的缓存条件下，单独计时的 CUTLASS SM90 blockwise GEMM 单次调用延迟是多少？拷贝带宽作为参照。
3. **Q2 分派：** 延迟在分派边界 M = 63 → 64 处是否非单调？M = 15/16/17 对照组能说明配置与 tile 数各自的作用吗？

## 2. 源码依据（固定 `7b054aca96cea8be1369d651c3434ad140580b92`）

### 2.1 分派

[`scaled_mm_blockwise_sm90_fp8_dispatch.cuh` L204–225](https://github.com/vllm-project/vllm/blob/7b054aca96cea8be1369d651c3434ad140580b92/csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm90_fp8_dispatch.cuh#L204-L225)：

| 条件 | MMA tile | Cluster | Kernel 调度 |
| --- | --- | --- | --- |
| `M % 4 == 0` | 128×128×128（M×N×K） | 1×2×1 | TMA warp-specialized cooperative |
| `M % 4 != 0` | 交换后的问题；N 方向 128，M 方向 16 | 1×1×1 | TMA warp-specialized ping-pong |

kernel 为 `GemmUniversal<Shape<int,int,int,int>, Mainloop, Epilogue>`，未显式指定 tile scheduler（L125–126）；[`cutlass_gemm_caller`](https://github.com/vllm-project/vllm/blob/7b054aca96cea8be1369d651c3434ad140580b92/csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/cutlass_gemm_caller.cuh#L34-L63) 传入默认构造的 `KernelHardwareInfo`。CUTLASS v4.7.1 固定到 `cb4247394dd82148787aed73e5dc7cef33cbf862`；[`tile_scheduler.hpp` L107–128](https://github.com/NVIDIA/cutlass/blob/cb4247394dd82148787aed73e5dc7cef33cbf862/include/cutlass/gemm/kernel/tile_scheduler.hpp#L107-L128) 将默认 tag 映射到 `PersistentTileSchedulerSm90`。[cooperative L226–246](https://github.com/NVIDIA/cutlass/blob/cb4247394dd82148787aed73e5dc7cef33cbf862/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp#L226-L246) 和 [ping-pong L229–249](https://github.com/NVIDIA/cutlass/blob/cb4247394dd82148787aed73e5dc7cef33cbf862/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp#L229-L249) 均在传入 SM 数非正时查询设备。Ping-pong 的 consumer work 按 MMA warp group 数推进；不能把逻辑 tile 等同于启动 CTA，也不能假定两个 schedule 的工作分配相同。精确 grid 大小与实际启动几何仍须运行见证：Nsight Systems 无需硬件计数器即可记录 grid 与 block 维度。

### 2.2 三个量分开表述

- **逻辑输出 tile：** 非交换为 `ceil(M/128)·ceil(N/128)`；交换为 `ceil(N/128)·ceil(M/16)`。
- **Cluster：** 非交换时 tile 沿 N 两两成组；交换时每个 cluster 为单个 tile。
- **实际启动的 CTA：** 取决于 scheduler；从 trace 记录，不靠推导。

N = 4096（N 方向 32 个 tile）时的逻辑 tile 数：

| M | 路径 | 逻辑 tile |
| --- | --- | ---: |
| 15 | 交换 | 32 |
| 16 | 非交换 | 32 |
| 17 | 交换 | 64 |
| 63 | 交换 | 128 |
| 64 | 非交换 | 32 |

**M = 63 → 64 是主要边界比较：** 多一行，配置不同，逻辑 tile 少四倍。**M = 15/16/17 为独立对照：** 15 与 16 tile 数相同但配置不同；15 与 17 配置相同但 tile 数不同。tile 数描述几何形态，不能确立任何延迟差异的原因。

### 2.3 Kernel 选择

CUDA block-FP8 kernel 列表（[`kernels/linear/__init__.py` L461–469](https://github.com/vllm-project/vllm/blob/7b054aca96cea8be1369d651c3434ad140580b92/vllm/model_executor/kernels/linear/__init__.py#L461-L469)）依次为 FlashInfer/DeepGEMM 动态 kernel、DeepGEMM、CUTLASS。`VLLM_USE_DEEP_GEMM` 默认为 1；`has_deep_gemm()` 接受 vLLM wheel 内置的副本（[`import_utils.py` L500–507](https://github.com/vllm-project/vllm/blob/7b054aca96cea8be1369d651c3434ad140580b92/vllm/utils/import_utils.py#L500-L507)）。动态 kernel 在 M < 32 时交给 FlashInfer，M ≥ 32 时交给 DeepGEMM（[`flashinfer.py` L149–170](https://github.com/vllm-project/vllm/blob/7b054aca96cea8be1369d651c3434ad140580b92/vllm/model_executor/kernels/linear/scaled_mm/flashinfer.py#L149-L170)）。**源码预期、待阶段 0 验证：默认安装的 Hopper 上，CUTLASS 是回退路径，而不是默认路径。**

### 2.4 既有分派工作与预测限制

[#44572](https://github.com/vllm-project/vllm/pull/44572) 由 yewentao256 提交，2026-06-13 合并，引入本交换路径；它延续 [#43706](https://github.com/vllm-project/vllm/pull/43706)（2026-06-01 合并）。前者在 (N,K)=(4096,7168) 的若干小 M、非整除 4 点报告约 17–18 对 50 μs。旧臂含 padding/分配/复制，计时与缓存条件亦不同；它支持 P1/P4 的动机，却不能分离纯 GEMM 加速比、确定本次 padding 成本或证明四个形状的 M63/64 台阶。本研究是确认并扩展到整除 4 的 capture 点，不是发现 swap_ab。

[#52775](https://github.com/vllm-project/vllm/pull/52775)（SM120，2026-08-19 合并）在 prefill 退化后收窄小 M 交换路径；[#40170](https://github.com/vllm-project/vllm/pull/40170)（SM120，关闭未合并）探索更细的 M tile 选择。这些是调优先例，不是 SM90 测量。大 M 修改不在本章程范围内。

## 3. 工作负载与目标策略（冻结前决定）

**模型：** `Qwen/Qwen3-8B-FP8`，revision `220b46e3b2180893580a4454f21f22d3ebb187d3`，TP = 1，BF16 激活。[固定配置](https://huggingface.co/Qwen/Qwen3-8B-FP8/blob/220b46e3b2180893580a4454f21f22d3ebb187d3/config.json) 已确认 hidden 4096、intermediate 12288、32 query heads、8 KV heads、head dimension 128、36 个 decoder layer、动态 e4m3 FP8、weight blocks [128,128]。下载预检必须计算实际 config 字节摘要、核对这些字段，并在运行前保留模型 revision 与权重文件身份。下列四个形状由此推导；每个 projection 在每个 decoder layer 执行一次，decode 测量不含 startup/prefill。

| vLLM 层 | 推导 | 预期 (N, K) |
| --- | --- | --- |
| `qkv_proj`（合并） | N = (q_heads + 2·kv_heads)·head_dim，K = hidden | (6144, 4096) |
| `o_proj` | N = hidden，K = q_heads·head_dim | (4096, 4096) |
| `gate_up_proj`（合并） | N = 2·intermediate，K = hidden | (24576, 4096) |
| `down_proj` | N = hidden，K = intermediate | (4096, 12288) |

未量化的 `lm_head` 不在范围内。这些形状替代 DeepSeek-V3 benchmark 形状；后者来自一个放不进单张 H800 的模型。

**目标策略（本有界研究已接受）：** 能力、形状与配置检查通过时，预计 Hopper 选择 FlashInfer/DeepGEMM。本章程研究**强制 CUTLASS 回退配置**。理由是：DeepGEMM 被禁用或不可用时，SM90 上的 block-FP8 linear 层由 CUTLASS 服务，且 vLLM 为这条路径维护了一个禁用 DeepGEMM 的 benchmark。所有结论仅限该配置，不对默认 serving 作任何声明。选择设置：`VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel`，同时记录 `VLLM_USE_DEEP_GEMM=0`。设置是否足够，须由选择日志加 trace 共同验证；任何一项单独都不算。若此策略被否决，就在此停止，另为默认路径写章程。

**Graph 相关性有条件：** 固定 pin 的 balanced capture grid 包含 1/2/4 和 256 以下的 8 的倍数（[config L2413–2445](https://github.com/vllm-project/vllm/blob/7b054aca96cea8be1369d651c3434ad140580b92/vllm/config/vllm.py#L2413-L2445)）。Interactivity 模式、自定义 capture size、追加的 token/decode 最大值可能引入其他点。阶段 0 固定 balanced、max_num_seqs=64、无 speculation、无自定义 capture list、token budget 高于 capture ceiling。预期统一 decode 的逻辑 M63 在 graph M64 执行；须检查最终 capture list 与实际执行，不能仅凭并发数。在该 grid 上，capture M4 与 M8..64 走 base 的非交换路径。M63/64 是微基准代理，不是两种不同的 serving graph size。被动记录 prefill M/路径作为后续线索，不扩大计时门槛。

## 4. 阶段

**阶段 0：工作负载见证。** 在 Nsight Systems 下，以固定 decode 并发 1、16、63、64，分别用默认配置和强制 CUTLASS 配置服务该模型。每种配置记录：
- `Selected <kernel>` 日志行（仅为选择见证）；
- 每个 linear 层实际 trace 到的 kernel 名称及 grid/block 维度（执行见证）；
- 实际 scheduled-token 数、逻辑 M 与 graph padding 后的执行 M；
- block-FP8 linear kernel 占 decode 步 GPU 时间的比例，在同一步窗口内对不重叠 kernel 时长求和。

每请求固定 128 个输入 token、64 个输出 token，使用固定 token-ID 输入、ignore EOS、不启用 speculative decoding、TP=1、关闭 prefix caching，并记录全部 compilation/graph 设置。并发值不能证明执行 M。分开 prefill/warm-up；每负载保留至少十个稳定 decode step。层/形状归属需要 NVTX 或同等 op 元数据见证，不能由 kernel 名猜测。记录最终 capture list；仅当观察到统一的 63-token decode step 时评估预期的逻辑 M63→执行 M64。若 trace、源码/二进制身份或形状归属不可用，Q0 记为未评分，不得当作路径已见证继续优化。

若强制配置下未观测到 CUTLASS kernel，则停止。默认配置的结果仅作为背景记录。

**阶段 1：计时工具（开卡前，CPU 侧完成）。** 用预量化输入（不含激活量化）单独计时 GEMM，使用以固定 #55537 head 构建的 R2 性能 successor，不含新增 B1/B2 修复 guard；两臂均保留 #55537 已有 operand 检查。

- **轮换权重：** 使用 `B = max(3, ceil(2·L2_bytes / 每组字节数) + 1)` 组互不相同的权重/scale，其中 `L2_bytes` 运行时从设备属性读取；按固定顺序轮流访问。
- **Graph 构建：** 每个 (shape, M, 臂) 一个 CUDA graph，内含 `R` 次连续调用，循环访问各组。激活与输出预先分配；计时区内除算子自身外不做分配或同步，两臂完全一致。
- **预热与计时：** 先 replay `W` 次，然后用 CUDA event 对每次 replay 计时；单次调用时间 = replay 时间 / `R`。
- **缓存条件：** 轮换降低缓存复用，但不保证每次调用都从 HBM 读取全部权重，也不能精确再现 serving。有硬件计数器时，在最多三个点测量 DRAM 读取字节与 L2 命中率；否则把缓存条件明确写为假设。
- **热对照：** 同一 GEMM、相同 graph 结构与 R，但只用一组权重。
- **参照：** 同一会话实测的设备内拷贝带宽。仅作解释参照，不是 GEMM 必须达到的上限。

**阶段 2：基线（H800）。** 对 §3 中四个形状，M ∈ {1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 24, 31, 32, 33, 40, 47, 48, 56, 63, 64}，BF16 输出，按 §6 程序测量。可选参照：若 DeepGEMM 已可导入，在相同点测量。

**阶段 3：一项候选修改（仅当 §5 差距门槛通过）。** 对所有 M ≤ 64 使用交换小 tile 配置的分派变体。在扫描点中，它改变 M ∈ {4, 8, 16, 24, 32, 40, 48, 56, 64}。解释任何变体计时之前，§7 正确性必须全部通过。Split-K、stream-K 与新 tile 形状不在范围内。

## 5. 预测与差距门槛（开卡前冻结）

P1/P4 是由 §2.4 先验支持的预测，不是本配置的测量；方向有依据，效应量与是否适用于全部 cell 仍未知。

- **P1 边界：** 至少一个相关形状上，t(64) − t(63) 超过 §6 的判定界。预期方向为 t(64) > t(63)。
- **P2 对照（描述性，不判定通过与否）：** 报告每个形状的 t(15)、t(16)、t(17)，不从 tile 数作因果推断。
- **P3 缓存：** 每个形状在 M ∈ {1, 16, 64} 时，热对照比轮换运行更快，且差值超过判定界。两者都不含激活量化。
- **P4 变体（仅阶段 3）：** 在通过差距门槛的形状上，检查变体在改变点是否比 base 快且超过判定界。四个形状的全部改变点均检查退化，包含未通过 P1 的形状；任何改变点不得退化超过判定界，全部未改变对照须保持在判定界内。分开报告改善、等价、退化与证据不足；等价不算加速。

**阶段 3 的差距门槛：** P1 在某个相关形状上成立，在 `b` 个配对块中至少 `q` 个复现，并超过相对 `e_rel`、绝对 `e_abs` 的最小效应。`q`、`b`、`e_rel`、`e_abs` 在冻结时确定。roofline 或拷贝带宽比例只用于解释，不能决定门槛通过或失败。

**无差距：** 若门槛未通过，记录基线并结束本实验。这是一个完成的结果。

## 6. 测量程序（开卡前冻结）

- **固定次数：** 预热 `W`、每个 graph 的调用次数 `R`、每块 replay 次数 `n`、块数 `b`。
- **配对交错的两臂：** base 与变体用不同的注册 namespace 构建，在同一进程中加载。每块内按 ABBA 顺序交错，块与块之间交替配对顺序。
- **A-A 校准：** 在相同的交错位置上做 base 对 base，先于任何跨点或跨臂比较。
- **预先确定的判定界：** `max(k · s_AA, e_abs, e_rel · t_base)`，其中 `s_AA` 为 A-A 块差值的稳健离散度（缩放后的中位数绝对偏差），`k` 在冻结时确定。可以在校准后计算判定界，但不能修改公式。
- **时钟与温度剔除：** 记录 SM 与显存时钟、温度、功耗和降频原因，包括每块内部的热态采样；按 §11.2 的固定功率例外及剔除规则处理。热态 SM 时钟偏离会话中位数超过冻结的百分比时剔除该块。被剔除的块计入 `q` 的分母，不补做。
- **重复：** 所有计划内的重复都运行。看到结果后不选择性重跑、不改阈值、不追加测量点。
- **两臂对等：** 对比 base 与变体的 `build.ninja` 和编译命令，只允许注册 namespace 与分派源码不同。两臂共用计时代码、分配方式和同步方式。已新增支持 `--namespace` 的独立性能 successor 及 CPU 测试，历史 R2 保持不变。GPU 编译与双库加载仍未验证；两臂均加 `-Wl,-Bsymbolic` 将内部 ELF 符号本地绑定。计时前必须核查各库的实际分派身份，不能仅以注册名不同证明运行了不同实现。
- **Nsight Compute：** 会话开始时探测计数器权限。没有权限时，计时与 trace 结果仍然成立，依赖计数器的项目记为未评分。

**计时前的双库见证：** 在 M64、相同合法输入上，trace 必须分别显示 base 的 cooperative/非交换 kernel 与变体的 ping-pong/交换 kernel。若符号无法归属到各 namespace，或两臂执行相同路径，则停止并记为未评分；注册名不同本身不能证明隔离。

## 7. 变体的正确性覆盖

历史 30 用例提供参考案例，不是可原样复用的 fix 臂判定：性能两臂不含 B1/B2 修复 guard，不能沿用这些输入的旧 `raises` 预期。原套件保持不变，另建性能变体套件，对 §3 全部四个形状增加：
- 改变的分派点 M ∈ {4, 8, 16, 24, 32, 40, 48, 56, 64}；
- 相邻未改变的交换路径对照 M ∈ {3, 5, 15, 17, 31, 33, 63}；
- 与浮点反量化参考结果的数值比较，使用冻结的相对误差界（0.005）；
- 在 M ∈ {4, 64} 做 CUDA graph 捕获与 replay；
- 在 (N,K) = (4096,4096) 上测全部 16 个正确性 M 的 FP16 输出。

两臂均须通过全部新增数值与 capture 检查。必测矩阵每臂为 64 个 BF16 eager、8 个 BF16 graph、16 个 FP16 eager（每臂 88 项，共 176 个臂用例），使用独立输出参考。任何必测项失败或未评分，都不解释变体性能。

## 8. 预算与会话

总计 **18 个工作小时**，含准备与构建。H800 采用一次租用中的三个有界阶段；不是三段必然全部执行：

- **开卡前：** 本地完成工具、源码变体、wheel 归档、安装/下载方案、CPU 检查和公开冻结 SHA 核对。无需为了准备再次释放稀缺 H800；模型下载和远端安装可在同卡阶段 P 完成。若有无卡环境，仍可提前下载，但不在 2 GB 无卡模式编译 CUTLASS。
- **准备 P（最多 60 分钟，含失败封存）：** 同卡检查网络和空间、下载固定模型、独立环境安装显式父 wheel、依赖/源码/扩展身份核验。调用模型预检和 `serving_preflight.py` 保存回执；失败或超时不进入 A，封存后关机。不是性能测量；不得更换 pin、模型、候选、参数或效应阈值以通过准备。
- **会话 A（最多 100 分钟）：** 身份与 Nsight 探测 10、base 构建 20、阶段 0 见证 25、marked 分派 probe/导出/检查 5、阶段 2 基线及 A-A 校准 30、封存与关机 10。
- **会话 B（最多 70 分钟，仅当门槛通过）：** 身份 10、变体构建 20、正确性 10、配对计时 20、封存 10。

一次租用上限为 P60+A100+B70=230 分钟；P 或 A 停止则立即封存关机，不等待剩余预算耗完。B 仅在完整 A 差距门槛通过后于同卡继续，不重新预约；若发生换卡，仍遵守下文身份限制。以往会话中两次构建加测试约用 42 分钟，这不保证本性能构建耗时。若某阶段超时，其余测量点记为未评分，会话按时结束。

Stage 0 次序为 forced CUTLASS Nsight、forced CUTLASS capture shapes，最后可选 default Nsight（仅背景）；不运行 default shapes。三项共用 25 分钟截止时间，超时先放弃 default 背景，不能删必需的 forced 见证。每 step 的 36×4 对应只适用于 forced CUTLASS，不要求默认 FlashInfer/DeepGEMM 使用相同启动模式。

会话 B 允许不同物理 UUID，并明确记录，但 GPU 型号、SM 数、L2、NVIDIA driver、CUDA driver API、nvcc 与 Torch 必须相同。B 的分派、正确性与计时见证须在同一张 B 卡上；B 自行重新校准并内部配对。A 的差距门槛可能来自另一张卡，这是门槛可迁移性的限制，不是跨卡加速比较。

## 9. 开始验证的条件

1. 完成重复工作检查：vLLM 中关于 blockwise FP8 小 M 分派、`swap_ab`、SM90 tile 配置的开放 PR 与 issue，以及 CUTLASS 上游。
2. 工作负载与目标策略（§3）已确定。
3. 修订后的章程、计时工具、支持 namespace 的构建器、正确性测试和预测已审阅、提交并推送。
4. 构建与测量计划符合会话预算。

先运行准备 P；通过后只运行会话 A，只有完整差距门槛通过才运行会话 B。公开冻结核验必须在 P 之前；不能等安装后或测量后才补提交。

## 10. 延续的约束

原始日志、trace、主机名、路径与 profiler 报告保持私有；只发布配置、计数、摘要表与摘要值。按现有决定，不新开上游 issue，不开第四个 vLLM PR。日后的任何 PR 遵循 vLLM AGENTS.md。正式采纳提交 10 月 26 日–11 月 1 日的复盘。

## 11. 冻结前补充记录（2026-09-30；没有 GPU 数据）

### 11.1 下次公开冻结采用的参数提案

| 参数 | 数值 | 含义 |
| --- | ---: | --- |
| W | 5 | 每个计时位置预热 graph replay 次数 |
| R | 8B | 完整轮换八遍权重组；热对照使用相同 R |
| n | 9 | 每个位置计时 replay 次数；每次时间除以 R |
| b | 7 | 计划配对块数，不补做 |
| q | 6 | 阳性主张至少需要六个有效块超过判定界 |
| k | 3 | 缩放 MAD（1.4826）的噪声倍数 |
| e_rel | 0.03 | 最小相对效应 |
| e_abs | 0.5 microseconds | 每次 GEMM 最小绝对效应 |
| Clock deviation | 10% | 热态 SM 时钟偏离会话中位数的界限 |

这些是保守的决策常数，不是实测灵敏度或统计置信度；公开冻结前仍可审阅。校准或测量不完整时记为证据不足，不算“无差距”。即使前六块已命中，也运行全部七个计划块。

### 11.2 会话 A 的配对与校准定义

对每个 (shape,M,cache condition)，以同一 base graph 的两个位置标签做 A-A 校准。每块四个位置：偶数块 ABBA、奇数块 BAAB；每位置 W 次预热后 n 次计时。每侧共 2n 个值，块值取其中位数。

随后对四个形状按相同顺序比较 M64 与 M63，`difference = t64 - t63`、参照 `t63`。两 cell 各自校准，判定取二者判定界中较大者。热/轮换也按配对位置测量，`difference = t_rotating - t_hot`，参照为轮换。A-A 校准覆盖 80 个轮换 cell 与 12 个热 cell。广泛扫描的基线来自这些校准测量，不使用另一次非配对扫描来判断台阶。

每个 cell 的 s_AA 为有效 A-A 块差值相对于其中位数的 1.4826·MAD。至少 q 个有效校准块；若 A-A 差值中位数的绝对值超过 max(e_abs,e_rel·t_reference)，该 cell 校准失败。否则判定界为 max(k·s_AA,e_abs,e_rel·t_reference)。阳性比较要求原 b 块中至少 q 个分别超过判定界，且有效差值中位数超过该界。全部原始 replay 值私有保留。

会话 B 使用同一定义，`difference = t_base - t_variant`、参照 base；在本会话重新做 A-A 校准，不将会话 A 时间减去会话 B 时间。检查全部 36 个改变 cell 和 44 个未改变 cell。任何改变 cell 的退化中位数超过判定界，或未改变 cell 的中位差值绝对值超过判定界，都不接受改善主张。每项必测退化检查至少 q 个有效块，否则为证据不足。

在热态计时块内监测时钟，不能只用计时前后 idle 时钟；保留原始端点与热态采样。固定功率设置下的软件 power cap 若全程一致且时钟通过偏差检查可接受；热降频、硬件 slowdown、功率限额变化或时钟样本缺失均使该块无效。剔除块不补做。

### 11.3 预算算术与准备状态

每 replay 含 R 次 kernel 调用；每个比较/校准 cell 固定工作量为 b·4·(W+n) 次 graph replay。会话 A 为 92 个 A-A cell、4 个边界配对与 12 个缓存配对：108·7·4·14 = 42,336 次 replay。会话 B 为 80 个轮换 A-A cell 加 80 个 base/variant 配对：160·7·4·14 = 62,720 次 replay，另有 176 个正确性臂用例。这是工作量计数，不是实测耗时估计；本 packet 的 GPU 耗时、graph capture、模型 startup 和构建均未计时。以 #44572 不同形状的 17–60 μs 粗估，R=8B 的 replay 可能约 0.5–1.5 ms，对应 A 约 21–64 秒、B 约 31–94 秒的 kernel 时间；不是主机会话耗时预测，风险主要在构建、serving、capture、同步与 trace。仅用固定次序前三个 cell 推算剩余时间，不改次数/阈值；超时后的剩余 cell 未评分。会话上限仍为 A=100 分钟、B=70 分钟，总工时 18 小时。

CPU 准备已新增独立性能[构建器](../../experiments/kernel-operand-contracts/sm90-block-fp8-perf/build_perf.py)、[协议定义](../../experiments/kernel-operand-contracts/sm90-block-fp8-perf/protocol.py)和测试；历史 R2、修复 patch、receipt 和冻结 manifest 均未改动。namespace 测试 mock 编译，只证明参数与注册隔离，不能证明 nvcc 编译/链接成功或 GPU 执行不同。

CUDA 计时、原生 serving 采集、需复核的 shape/graph 见证及 176 臂用例套件已实现并做 CPU 测试，见[runbook](../../experiments/kernel-operand-contracts/sm90-block-fp8-perf/RUNBOOK.zh-CN.md)。这不证明 GPU 或 profiler 兼容。

**仍阻塞冻结/开卡：** 审阅这些可执行工具；运行预检验证编译后注册与分派隔离；下载并绑定模型 config/权重；复核工作量预算并完成公开 manifest。本次文档修改不授权 GPU 运行。

serving Python 必须匹配 pin。只为 Stage 0 允许显式记录的父提交 `4f1451679088e5832bce0965a254295c854aaa09` cu130 预编译 wheel：pin 的唯一 C++ 差异是 `scaled_mm_entry.cu` 中已审阅的 CPU operand checks，serving 二进制不含这些检查。这种 Python/二进制混合身份不是完整 pin build，也不验证 K5；最小计时两臂仍编译 pin 的 C++ 及继承检查。开卡前须保留 wheel URL、完整 commit、归档 SHA-256、依赖 metadata、安装后扩展哈希与安装日志；要求 Torch 2.13.0+cu130，预检逐一比较安装扩展与归档。显式选择保留 wheel，不能回退 latest。用户返回的 cu130 索引已发布父 commit 的 x86-64 cp38/abi3 条目，variant 为空，相对 path 解析到根目录父 commit 下。须绑定原始索引和摘要，再解析其条目，不能要求最终 URL 包含 `/cu130/`。归档阶段的新证据见下段；CUDA/ELF 兼容性及安装仍未验证，若不兼容则开卡前停止，不静默追加完整源码构建。确切索引见 runbook。

原生 Nsight 与 Torch shapes 采用独立 fresh-server 采集，共用 Stage 0 的 25 分钟截止时间。capture-op → replay-node 的对应是需复核的证据，不是自动跨 profiler 证明。缺失 input dimensions、graph node 或完整 step 归属时记为证据不足，不能用并发数推断 M。推荐开卡前先落实此路线。

归档阶段现已完成：315,907,592 字节，SHA-256 `f50bf6c6c63785be641d8512139593ebeb20f7c1e9bf290dc7864de414909128`；声明 torch==2.13.0、Python >=3.10,<3.15，含 stable CUDA 扩展。内部标签是 generic `cp38-abi3-linux_x86_64`，不是索引的 manylinux 标签，不能推断 glibc 下限。工具的两个过严要求（精确 manylinux 标签、HIP-only 旧 `_C`）已纠正，保留两份失败回执；CUDA/ELF 加载与实际安装仍未验证。若不兼容，开卡前停止。

两个主要 stable 扩展的只读 ELF 检查确认直接依赖 `libcudart.so.13`/`libcuda.so.1`，最高已声明 GLIBC 2.14、GLIBCXX 3.4.21；这不覆盖整个 wheel 或传递依赖。agent 访问本机 WSL 返回 E_ACCESSDENIED；用户已准备 Python 3.10 venv 并确认所需 Torch 版本在索引中，但不再继续本机整套安装。随后目标主机预检确认 H800 SM90、114 SM、cgroup 内存 120 GiB，已有 Torch 2.13.0+cu130、nvcc 13.0.88 和 Nsight Systems 2025.3.1.0。基础 Torch GPU 加法通过，不是 vLLM 扩展或性能结果。旧 serving 依赖的 pip check 失败，固定模型未下载，直接访问 Hugging Face 超时。未先确认公开冻结即进入付费主机，是流程错误；没有运行性能实验，已按用户要求关机。剩余安装、空间、模型与公开冻结门槛见 runbook §7，目前不批准再次开卡。

实现常数：seed 730 与独立权重 seed 10730 保持不同 M 的 B 与 B-scales 一致；相对 L2 数值界 0.005，关闭 TF32；graph replay 检查 A 清零与还原。NVML 热态窗口每 5 ms 采样，不改变功率限额；每臂三个 marked launch 验证分派。GPU 执行前核验公开 byte manifest；原始 trace 与复核映射私有保留。这些细节不改变上文固定矩阵、判定阈值或会话上限。

### 11.4 有界重复工作检索

最初只查开放工作的检索遗漏 #44572；冻结前已补读 §2.4 的合并/关闭历史，纠正这一检索范围缺陷。2026-09-30 的 GitHub issue/PR 精确检索为：
`repo:vllm-project/vllm is:open "swap_ab"`、
`repo:vllm-project/vllm is:open "blockwise" "small"`、
`repo:NVIDIA/cutlass is:open "SM90" "small"`、
`repo:vllm-project/vllm is:open "SM90" "tile"`、
`repo:NVIDIA/cutlass is:open "blockwise" "FP8"`。
已阅读当前 #56248 文件 diff：处理 operand stride，不改变小 M 分派。#55537/#56480/#56659 处理布局检查/stride，并非本变体。相关 [#43214](https://github.com/vllm-project/vllm/pull/43214) 是带 serving 测量的 DO NOT MERGE 低延迟 blockwise FP8 CuTeDSL/PDL 实现，属于相关性能供给，不是相同 C++ 分派修改。CUTLASS [#2923](https://github.com/NVIDIA/cutlass/issues/2923) 报告 B200 的 block-FP8 延迟问题，并非相同 SM90 测量。

CUTLASS [#3596](https://github.com/NVIDIA/cutlass/issues/3596) 及修复 [#3599](https://github.com/NVIDIA/cutlass/pull/3599) 涉及 256 行 tile 的连续 M wave 重用 B scale。在固定 collective 中，`NumSplitsM = TileM / 128`（[L208–209](https://github.com/NVIDIA/cutlass/blob/cb4247394dd82148787aed73e5dc7cef33cbf862/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp#L208-L209)）；两种目标 tile 的 TileM 均为 128，只有一个 M wave。这是该连续 wave 缺陷不匹配本次两种 tile 的源码证据，不是一般正确性保证或 GPU 验证。数值门槛仍必需，两臂均不加入该依赖修复。另查 `repo:vllm-project/vllm is:pr is:open author:yewentao256`，阅读全部四项文件列表（#59084/#58845/#57443/#57053），均未列出 SM90 blockwise dispatcher；不排除私有工作、不同署名或随后 push。未发现本后续修改的精确重复；交换技术本身是既有工作。提出上游修改前刷新 thread head。
