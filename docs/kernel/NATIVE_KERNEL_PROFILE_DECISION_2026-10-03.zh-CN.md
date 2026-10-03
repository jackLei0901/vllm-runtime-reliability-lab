# 原生 CUDA kernel 项目开卡前决策

状态：2026-10-03，美西时间，v0.3 待用户核查和公开冻结；中文为权威版本，[English](NATIVE_KERNEL_PROFILE_DECISION_2026-10-03.md) 为译文。用户已批准单次发现会话，不是实现立项或一般规则修订。提交并推送前尚未冻结。

当前决定：安装及无卡检查已通过。**先公开冻结，再开卡**，随后重新核查主机、磁盘、内存、GPU 身份及空闲状态。GPU runtime 和 serving 尚未验证。增量构建不是发现会话的前置条件。

执行批准补充（2026-10-03）：用户批准单次最多 60 分钟的发现会话及准备。增量构建证明移到实现立项。提供的重复扫描不穷尽；未完成的针对性检查阻止实现，不阻止方向 profile。本例外不批准第二次租卡或全部历史日期修订。准备期间主机身份已私下核验；重新连接及 GPU 准入检查仍须完成。

## 1 要解决的问题和预期价值

选择真实模型默认 decode 路径中的一项手写 CUDA kernel，完整负责调用分析、根因、原生实现、回归测试、算子测量和模型验证。Lab 保存可复用测试和证据，修改在 vLLM 中交付；不建设独立引擎、诊断记录器或新框架。

职业价值来自可解释的 C++/CUDA 实现及工程取舍，不来自实验数量或必须发布 RFC。真实 profile 证明路径可达和成本，不自动证明用户需求、低效可修复或维护者会采用。一个小改动可以是普通 PR，不必承担主要作品集项目的定位。

先完成 #55537 当前更新与既定 review 请求；不等其合入才做离线准备，但新 GPU 工作不得挤占它。FP8 #59800 按用户确认的合入状态收尾。INT8 不参与本次 profile；原 SM90 dispatch 实验保持 [insufficient_evidence](SM90_BLOCK_FP8_PERF_RESULT_2026-10-01.zh-CN.md)，不重评或重跑。

## 2 本次源码审查基线

GitHub main 查询返回 `b0e21b308352587a6fd02f72722a2e815bfd62f0`，[提交](https://github.com/vllm-project/vllm/commit/b0e21b308352587a6fd02f72722a2e815bfd62f0)时间为 2026-10-03 06:28:02 UTC。这只是读取时的快照，不是声明一直为最新，也不是已安装 wheel 的身份。

本次读取了该 pin 的 config、QuantFP8、block linear dispatch、FlashInfer/DeepGEMM 分支、量化 helper、FlashAttention cache 更新和 CMake。没有运行模型或编译。旧本地 checkout 不是本轮 pin，且有未提交改动；没有用其工作目录状态作为洁净源码证明，也没有修改它。

## 3 默认路径可达性和模型筛选

| 边界 | 本次源码事实 | 对选题的含义及尚缺证据 |
| --- | --- | --- |
| custom ops | Inductor 且 compilation 非 NONE 时，未显式指定 all/none 会追加 none；blocked weights 且无 -quant_fp8 会追加 +quant_fp8。[config](https://github.com/vllm-project/vllm/blob/b0e21b308352587a6fd02f72722a2e815bfd62f0/vllm/config/vllm.py) | 开关不等于某个手写 kernel 一定执行；还要跟踪 backend、融合和实际图。 |
| block linear | CUDA 候选列表优先动态 FlashInfer/DeepGEMM，再 DeepGEMM、CUTLASS 等；实际选择取决于可用性和 can_implement。[dispatch](https://github.com/vllm-project/vllm/blob/b0e21b308352587a6fd02f72722a2e815bfd62f0/vllm/model_executor/kernels/linear/__init__.py) | 不强制 CUTLASS、不关闭默认融合、不用 eager 来制造可优化路径。 |
| 动态 backend | apply_input_quant=False；动态函数的小 M 分支进入 FlashInfer，另一分支显式调用 per_token_group_quant_fp8 再执行 DeepGEMM。[dynamic kernel](https://github.com/vllm-project/vllm/blob/b0e21b308352587a6fd02f72722a2e815bfd62f0/vllm/model_executor/kernels/linear/scaled_mm/flashinfer.py) | batch 1 和 32 可能执行不同的量化生产者；不能从 +quant_fp8 推出两者都运行普通 CUDA quant。 |
| 普通与 packed 量化 | QuantFP8 有普通及 packed 分支。普通 helper 在 CUDA 且 contiguous 时进入 _C.per_token_group_fp8_quant；packed helper 调用 _C.per_token_group_fp8_quant_packed。[QuantFP8](https://github.com/vllm-project/vllm/blob/b0e21b308352587a6fd02f72722a2e815bfd62f0/vllm/model_executor/layers/quantization/input_quant_fp8.py)、[helpers](https://github.com/vllm-project/vllm/blob/b0e21b308352587a6fd02f72722a2e815bfd62f0/vllm/model_executor/layers/quantization/utils/fp8_utils.py) | 算子名、真实执行 M、布局、scale 格式及 fusion 决定覆盖哪条实现；profile 必须见证，重复检查须区分普通与 packed。 |
| KV 写入 | FlashAttention backend 的 do_kv_cache_update 调用 reshape_and_cache_flash；CMake 列入原生 cache 源码。[backend](https://github.com/vllm-project/vllm/blob/b0e21b308352587a6fd02f72722a2e815bfd62f0/vllm/v1/attention/backends/flash_attn.py) | 仅条件可达；最终 attention backend、融合和符号到源码的绑定仍需核验。 |
| norm 与 sampling | RMSNorm native 进入 IR；sampler/topk 的原生文件存在于 CMake。 | 文件存在不证明所选配置使用它们。IR 优先级、fusion、V2 sampler 与 greedy 设置未完成逐项追踪，不能把这些列为已见证候选。 |

选定模型为 `Qwen/Qwen3-8B-FP8` @ `220b46e3b2180893580a4454f21f22d3ebb187d3`，TP=1、BF16 激活、动态激活量化、block [128,128]、hidden=4096、intermediate=12288、36 层。已独立读取缓存配置，SHA-256 为 `79e454d6cc36c41ab0e9418c8f78a033afa9ac88e5a58d67c1471de315c8d29a`。两份权重均通过 header 长度检查，SHA-256 与 HF 缓存 blob 名一致；未独立获取远端 LFS 元数据。显存充足仍是 GPU 准入检查。不下载第二个模型、不引入 MoE。

补充审阅提出的预期集合是：小 M 时 KV write；M≥32 时增加 group quant。独立复核确认动态函数的普通分支阈值为 `input.shape[0] < 32`，但 `VLLM_BATCH_INVARIANT` 为真会直接执行 DeepGEMM 分支。必须记录该设置，并按每步实际 padded M 分类；并发 32 不保证每一步都走 M≥32。优先列表及依赖存在只支持预计选择，不能代替 installed wheel 的选择见证。36 次 KV write、最多 144 次 group quant 是完整 dense decode 步骤下的结构预期，不是每个窗口已测出的计数。

不能暂时把该集合写成“只有两个 kernel”：norm/IR 和融合的逐项排除仍未完成。旧 forced-CUTLASS SQLite 的只读全程汇总确实包含 group quant、KV write、RMSNorm+quant 和 SiLU+quant 的手写 CUDA 符号。这不是默认 backend 的执行证据，也不是限定并发的稳定 decode 样本；它只证明旧 trace 不能直接验证新的完整集合。

冻结预测为小 M 的 KV write，以及动态 backend 执行 M≥32 时的普通 group quant。Norm/activation fusion 可能增加原生 kernel，因此不声明集合穷尽。GEMM、attention、外部库和生成 Triton 不作为候选，但保留在来源分布报告中。FlashAttention cache-update 调用链提供一条潜在合格的原生路径；是否执行、有多少剩余空间是本轮发现问题，不是已证实结论。

## 4 重复工作和需求记录

已合并先例是 [#56478](https://github.com/vllm-project/vllm/pull/56478)、[#55330](https://github.com/vllm-project/vllm/pull/55330)、[#58194](https://github.com/vllm-project/vllm/pull/58194)。它们说明量化路径已有改动，不能证明当前还存在缺口。[#47334](https://github.com/vllm-project/vllm/issues/47334)提出 CuTeDSL 迁移，不能据此声明全部手写 CUDA 即将退役或目前无人重写某个 kernel。

随后提供的审阅按 changed-file 路径扫描了 674 个唯一 open PR，查询为 `gh pr list --state open --limit 200 --search "<term> in:title" --json number,title,files`，term 为 quant/kernel/cache/fp8。三项查询达到 200 条上限，结果不穷尽，也未在本轮重新刷新。相关工作：#51939 将量化移出 opaque DeepGEMM op；#42597 试验 all-reduce/norm/quant fusion；#49928 为 TMA scales 保留内部量化；#48653 修改 group-quant kernel；#59289 修改 MLA cache write 而非 reshape_and_cache_flash。早先标题搜索无命中的说法已被替代。完整针对性重复排除仍是实现门槛。

恢复访问后，对量化、norm、cache、sampling 分别执行 `gh search prs --repo vllm-project/vllm --state open --limit 100 "<term>"`，并搜索 closed/merged 工作；检查是否截断。对预期进入 profile 的具体算子，再用文件路径、符号和相关 issue 搜索、读取候选 diff，而不是只读标题。普通量化和 packed 量化分别检查。有覆盖的候选退出；仅因 PR 沉默不把其工作据为己有。保留查询、日期、相关 head、交集和未排除项。

每个候选最多一页：工作负载及用户场景、症状、默认调用链、已有报告或没有报告、修改假设、数值契约、重复工作和上游出口。CODEOWNER 只是评审路由，不是同意参与的消费者。

## 5 二进制身份和构建路线

已在隔离环境安装 `b0e21b3` 官方 wheel：版本 `0.30.1rc1.dev618+gb0e21b308`，SHA-256 为 `2282e931a96725cbe8a20f0f7ccc77e25406fff8eee5132f75f80ddbbc503780`，来源为固定 commit 的 `https://wheels.vllm.ai/b0e21b308352587a6fd02f72722a2e815bfd62f0/`。5,305 个已安装 wheel 文件全部匹配；与固定源码 archive 比较的 2,585 个 in-tree Python 文件全部匹配。另有 225 个生成或 bundled third-party Python 文件不在源码 archive 中，只由 wheel 摘要绑定，不宣称源码相同。这不独立证明 native build provenance。GPU 准入时记录加载扩展摘要。

199 个包通过依赖检查。Torch 为 `2.13.0+cu130`，FlashInfer Python 及完整 cubin 包为 `0.7.0.post1`，Nsight Systems 为 `2025.3.1.0`。CPU Torch/vLLM 导入通过且未初始化 CUDA。失败下载/安装记录私下保留；最终镜像下载匹配 canonical package 摘要。未运行 GPU 或源码构建。

实现路线优先尝试复用兼容的外部依赖、增量构建目标扩展，但尚未成功试建。[该 pin 的 CMake](https://github.com/vllm-project/vllm/blob/b0e21b308352587a6fd02f72722a2e815bfd62f0/CMakeLists.txt)把多数 CUDA 源码放在 _C_stable_libtorch，而不是简单重建一个 _C 文件；它还包含多个外部构建。不能把一次 CUTLASS 最小扩展成功当成任意原生 kernel 都能便宜构建。

实现立项前在可用 Linux toolchain 中证明：baseline 目标构建成功、可加载；对入选目标源码作可撤销的身份变更后能增量构建，时间/内存/磁盘可承受，且加载的确为新 binary。不得修改历史 checkout；身份变更不是优化或正确性证据。若无足够 CPU/RAM 的环境，记为实现构建阻塞，不在 profile 会话现场临时全量编译。没有无卡环境时，只能另提有明确费用和时间上限的构建预检，不能视为本文件已授权。

## 6 有限发现与收益判断

用户已批准一次 60 分钟发现会话；以下设置待公开冻结。文档、下载、查询及工具修复须计入成本，不重置历史花费。4 工时离线上限未核算，延长准备如实记为预算核算偏差，不宣称合规。本次批准不资助实现。

一次发现会话建议总上限 60 分钟，计入身份/启动/编译缓存/两负载 profile/封存/清理；不开额外实现会话。分配和开始采集所需最少剩余时间须由既有启动记录与离线准备确定，未证明能容纳前不冻结。到时限保留证据并清理；支持证据失败不允许无限补采或现场换模型/backend。

冻结负载为并发 1 和 32、128 输入 token IDs/64 输出 token、greedy 且 ignore_eos、TP=1、BF16、max model length 2048、max sequences 32、禁用 prefix caching、无 speculation。保持默认 backend 和 runner。每种负载两次 warmup、一次无 profiler 参考 batch，再各采两段 profile 窗口。排除 startup/prefill 后，每窗口至少 16 个可无歧义归属的完整纯 decode 步骤；缺失或归属不明为 insufficient_evidence，保留部分数据、不补采替代窗口。记录 token 数，只结合 resolved config 与执行证据推断 padding；并发不等于执行 M。

使用 Nsight Systems 取得符号、计数、step 窗口及 CPU/GPU 重叠；关机后离线 export 和归属，不在本轮追加 Nsight Compute。无 profiler 的 HTTP batch 时长仅为负载参考，不是无插桩 step time 或 TPOT，不能作为下方模型收益公式的分母。若没有合格的 step 分母，只报告定性空间，暂缓定量实现立项。

排除 GEMM、attention、外部库与生成 Triton。约 2% 的 GPU 累计时间份额只用作排序提示，不是修改许可。最多检查三个候选，选一个；同一 kernel 各调用求和、区分重叠与关键路径，并绑定目标 csrc 符号和文件。

外部库只从候选中排除，不从报告中删除。实现来源统计保留 vLLM csrc、Inductor-generated Triton、FlashInfer、DeepGEMM、FlashAttention、其他已归属和 unknown。不将名称前缀当作充分归属证据；特别是 FlashInfer 可以调用 DeepGEMM，共享符号无法区分时保留联合标签或 unknown，不能重复计入两桶。累计 kernel 时间比例与步骤墙钟时间/重叠分别报告，不能把来源份额宣称可消除收益或端到端加速。

对每个候选分别报告乐观性能空间和具体修改的预期节省。仅适用于带宽主导时才使用 bytes/BW；同时考虑数据实际来自 HBM/L2、计算量、启动及依赖延迟。大 buffer copy 带宽只是参照，不是小 kernel 可达到的目标。[NVIDIA 诊断指南](https://docs.nvidia.com/nsight-compute/ComputeTriage/)

若无重叠且关键路径不变，`预期模型比例改善 ≈ 每步目标调用的预期节省总和 / 未插桩步骤墙钟时间`。分母必须明确，不能把 step 和 HTTP TPOT 混用；有重叠时，此式只是乐观估计。1.5% 是建议的主要作品集工程门槛，不是统计显著性或已证实收益。仅理论上限超过它不够：必须有源码支持的可消除工作。最终仍需算子与模型 A/B 才能宣称效果；较小改动可另作普通 PR。

## 7 决策和停止规则

| 阶段 | 通过要求 | 不满足时 |
| --- | --- | --- |
| 开卡门槛 | 模型/commit/配置/文件身份及匹配安装已核验；预期集合非空；重复扫描及截断已披露；工具可用；本次预算获批准 | 缺安装或资源证据为暂缓；不为增量构建而租卡。 |
| 有效 profile | 两负载取得可归属的实际原生执行及稳定窗口，足以估计成本和关键路径 | 装置或见证缺失记 insufficient_evidence，不声称 no_gap，不自动追加会话。 |
| 实现立项 | 一个候选通过针对性完整重复排查，有源码根因、明确数值契约、实质 C++/CUDA 修改、可信收益估计、已证明的增量构建及另行批准的实现预算 | 有效 profile 无合格候选则 completed_no_candidate；保留来源分布作为方向筛选，不自动启动下一项目。 |

补充一个待批准的离线提前停止规则：若候选集合已完整排除遗漏，且在可比配置、稳定 decode 窗口和明确分母下，每项候选的保守乐观收益上限均低于主要项目门槛，则关闭此模型的发现提案，记为“离线筛选不立项”，不开卡。它不是 measured no_gap 或 completed_no_candidate。旧 forced-CUTLASS 全程平均、小于 1 MB 或 launch-dominated 的猜测都不足以单独通过该停止判断；需核验布局、融合、binary/pin 差异和关键路径。只为重用旧证据作有限离线归属，不新建通用 trace 框架；证据不足保持暂缓。

本次只提议资助到发现，不自动资助修改、模型 A/B 或租卡。修改预算需在候选确定后单独提出。没有合格候选时，Lab 做维护和已有贡献，不立即换模型继续搜索。

## 8 规则修订提案和受保护工作

[LAB_REQUIREMENTS](../LAB_REQUIREMENTS.zh-CN.md)的文件状态仍写“本地提案，未批准、未发布”；其 10-12/11-01/11-30 约束曾用于对话决策，但本次未核验正式批准历史。不能假装已由本文件替代。

建议新增有界 kernel 发现入口：开卡前以代表性默认工作负载、源码可达性、重复检查和可构建性代替“已承诺的事故消费者”作为**发现**入口；只有 profile 加源码根因通过才提出实现立项。外部需求单独记；缺少报告的项目只能称工作负载驱动的优化候选，不能说已有客户需求。

交付指补丁/测试/证据可由他人运行并已提交；采用指合入或明确决定使用。维护者拒绝是评审结果，不是采用；自己的 PR 不能倒算成原“他人运行时线程三份产物”目标。

建议 10-12 改为批准/否决一个有界 kernel 决策页及预算；没有合格提案则不启动新建设，11-01 维护条件保留。11-30 的三份他人运行时线程目标建议仅对新 kernel 线不再适用，不换成必须合入或必须正收益指标。须由用户明确批准、日期化后才生效；此前旧约束保持，不因写了文档自动延期。#55700 的 10-12 关闭规则不变。

#52178、PyTorch #197232 的既定跟进，#55537 的更新/review，以及现有工作树和 SUBMIT.ps1 编码问题作为受保护维护工作；其当前状态和日期另行刷新，不在本次修改、推送或发消息。

## 9 冻结前剩余动作

1. 用户核查两份决策文件、collector 和 CPU 测试，按明确路径提交并推送。记录完整 commit SHA，采集前一次性核对公开文件与本地字节。
2. 重新连接已准备环境。安装完成，不重装、不下载第二个模型。开卡前重新检查磁盘：最近准备 receipt 仅报告系统约 3.1 GiB、数据约 2.2 GiB 空闲。若缓存及报告放不下，先解决再开始计费；未获授权不删除模型、环境或历史证据。
3. GPU 启用后立即开始 60 分钟计时，前五分钟记录 H800/SM90 身份、driver、cgroup 资源、磁盘及 GPU 空闲。准入 receipt 绑定核验过的安装、模型、解释器和 collector 摘要；准备 receipt 不是准入 receipt。
4. 使用核查过的 collector，不改变 backend、不试建源码。启动上限 20 分钟，分钟 55 停止采集，剩余不足十分钟不开始 profile。最后五分钟用于停止 server/profiler、封存和传回小 receipt；报告保留远端并核验摘要，export 和解读离线进行。超时或见证失败保留 insufficient_evidence，不自动重试。
5. 封存后关机，独立确认供应商计费状态。来源分布、步骤充分性和候选准入从封存 trace 核查，不能凭 HTTP 请求成功宣称通过。

准备新增 [独立 collector](../../experiments/native-kernel-discovery/collect.py) 和 CPU 测试，没有改旧冻结工具或无关工作树。Linux syntax/help 和 CPU 导入检查通过；Linux/Nsight 采集尚未测试。提交和推送仍由用户操作。报告采到后仅为 review_pending，不证明存在候选。
