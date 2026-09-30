# SM90 block-FP8 算子契约修复：初步需求说明书

状态：2026-09-29，v0.2，可行性进行中；本中文版为准，[English](SM90_BLOCK_FP8_PROJECT_REQUIREMENTS.md) 为译文。本文提出一个由 Lab 负责实现与验证的底层项目，尚不是已冻结的执行协议。预算、模型及测量阈值须在对应阶段开始前确定。

## 1. 要解决的问题

vLLM 的 SM90 block-FP8 `cutlass_scaled_mm` 路径依赖输入 dtype、scale 的物理布局和 operand stride。Lab 已观察到：部分不符合内核解释方式的输入通过入口后，产生明显错误或非有限输出；同一算子的标准分支对其中部分输入已有检查。

需要建立一个可执行的边界：允许的输入得到正确结果，其余输入得到明确的处理或错误。允许范围须由调用者、算子入口与内核约束共同说明，不能仅凭现有负控结果决定应当拒绝还是支持某种输入。

首个项目采用本地 **reject** 策略：为 SM90 block-FP8 的 operand dtype（B2）与 scale 布局（B1）增加检查，错误输入必须拒绝，不隐含复制或转换。这是待验证的本地契约，不代表上游已接受。支持第二种 scale 布局属于另一次范围与成本决策，不纳入本项目。

本地分支拟建立在 #55537 的固定 head 上（2026-09-29 核查为 `7b054aca96cea8be1369d651c3434ad140580b92`），继承其 operand stride 拒绝及 alignment 检查。B0 只作为继承行为的回归测试，两臂均保留 #55537，不另写 B0 修复。并行方案为 [#55537](https://github.com/vllm-project/vllm/pull/55537) 的 reject，与 [#56248](https://github.com/vllm-project/vllm/pull/56248)（SM90）、[#56480](https://github.com/vllm-project/vllm/pull/56480)（SM100）、[#56659](https://github.com/vllm-project/vllm/pull/56659)（SM120）的 honor-stride 修改。可行性检查必须刷新它们，并核查入口冲突；若不能以 #55537 为基线，先修改范围或排除 B0，不静默改换策略。

## 2. 已有证据与价值边界

| 证据 | 已证明的事实 | 尚未证明的事实 |
| --- | --- | --- |
| [B1：8 个 H800 案例](../../experiments/kernel-operand-contracts/B1_H800_RESULT_2026-09-29.md) | 规定布局的控制正确；错误物理布局的 scales 在测试路径上被接受并产生显著输出差异；布局等价控制正确 | 正常 serving 会产生这些布局；新建当前上游二进制具有相同行为 |
| [B2/B0：16 个 H800 案例](../../experiments/kernel-operand-contracts/B2_B0_H800_RESULT_2026-09-29.md) | blockwise 分支接受 e5m2/int8 并产生错误或非有限输出；标准分支拒绝所测 e5m2 输入；padded views 重现已知问题 | 所有 dtype/shape 的行为；某种拒绝策略就是上游 API 契约 |
| [源码清单](QUANT_GEMM_OPERAND_CONTRACT_INVENTORY_2026-09-29.md) | 给出已检查入口、helper 和 dispatcher 的约束与源码定位 | 清单本身不能证明修复、实际调用路径或外部需求 |

上述运行针对一张 H800 和所记录的扩展。相关源码文件身份经过检查，原二进制的完整构建来源没有独立确立；历史结果保持原有范围。新修复验证需要可追溯的 base/fix 构建。已检查的正常 vLLM 调用者提供所需输入，因此当前是潜在的算子边界风险，不能声称生产部署已发生错误。

预期价值分为三项，各自评估：

- **技术价值：** 将已观察到的静默错误输入变成明确、经过回归测试的行为，减少调用者修改和后端切换时的错误结果风险。
- **外部价值：** 潜在使用者尚未确认，当前上游出口被既定约束阻挡：自然修改位置与 #55537 重叠，但不能擅自扩大它，也不能新增第四个 vLLM PR。计划出口是本地分支和可复现材料；至多在 #55534 留一条短评论，且仍须满足现行发布节奏与授权规则，本草稿不授权发布。预计外部使用为 `not_observed`，除非 #55537 合并后出现可用提交路径，或维护者明确要求此修复；这些条件本身仍不等于采用证据。
- **项目与学习价值：** 完成调用者、PyTorch custom op、C++ dispatch、CUDA kernel、测试和正常服务验证的完整链路。用户参与源码判断、构建和 profiler 操作；这项能力成果不能替代外部采用证据。

## 3. 项目交付与需求

Lab 负责以下链路：确定输入契约 → 编写 C++ 修改 → 构建 base/fix → 检验输入矩阵 → 验证有效调用者兼容性与 eager 主机开销 → 保存可复现材料。当前主要接受理由是有界的项目与学习价值，而非已确认的外部需求；C++ patch 可能只有约十行检查，能力增量在构建、dispatch 和验证链路，不是性能优化。

| ID | 需求 | 验收方式 |
| --- | --- | --- |
| K1 输入契约 | 明确 A/B/output、activation/weight scales 的 dtype、shape、stride、device 与必要 alignment；注明约束来自调用者、入口或内核 | 一张以源码 pin 为依据的契约表，覆盖普通与 swapped dispatch；对未解决的 API 选择列出本地候选策略及理由 |
| K2 C++ 行为 | 拒绝 B1/B2 错误输入，不复制或转换；错误信息指向具体输入及要求 | 每种新增行为有修复前失败、修复后通过的测试；决定检查放在共享 helper 还是 SM90 dispatcher，并列明架构覆盖 |
| K3 回归测试 | 保留有效输入、错误 dtype、错误 scale 布局、布局等价和 swapped 控制；B0 两臂均预期拒绝 | 冻结增量 B1/B2 的 base/fix 预期；eager 调用及 graph capture 时错误输入应拒绝，有效输入 capture/replay 应通过；不声称 replay 重做入口检查 |
| K4 构建身份 | 微测试 base 为固定 #55537，fix 只增加 B1/B2；匹配工具链与依赖，记录源码、patch、命令和扩展摘要 | 优先评估包含真实入口/dispatch/kernel 的最小 extension；若不可行，重新评估完整构建成本，不用复制检查逻辑替代 |
| K5 服务兼容性 | 仅使用完整 fix build，运行一个原生配置且能放入 H800 的真实 block-FP8 模型 | 能启动并完成事先规定的请求与输出有效性检查，新检查不拒绝有效调用者；须有实际 CUTLASS SM90 执行证据，无目标路径则未评分；不做 serving 性能 A/B |
| K6 主机开销 | 对 reject 策略仅比较有效 eager 调用的 CPU 提交开销，并给出 graph 检查时机的源码论证 | 事先固定方法、重复次数和容忍范围，避免逐次同步把 GPU 时间当检查开销；差异低于分辨率时报告测量界限，不称零开销 |
| K7 可用交付 | 提供 patch、测试、构建/运行命令、摘要结果及限制，原始私有数据留在私有位置 | 非作者无需追问缺失文件即可运行或评审；另外记录交付、独立使用和维护者接受状态 |

正常 serving 验证证明有效调用者未被新检查拒绝，不能证明服务原先触发了 B1/B2。关闭 DeepGEMM（`VLLM_USE_DEEP_GEMM=0`）只是选路条件之一，不是 CUTLASS 已执行的证据；必须选定后端并记录运行时路径见证。

K3 的 B0 拒绝来自 #55537 的 entry 检查，但其现有 `test_cutlass_c3x_rejects_padded_operand` 使用单个 scale，只覆盖标准路径。Blockwise 拒绝目前是源码预测，本项目的 B0 将首次直接测试这一预测；两臂预期相同不意味着已有这项运行覆盖。

CUDA graph 的 CPU 入口检查在 capture 时执行，replay 跳过这部分 Python/C++ dispatch；因此不能检测“只在 replay 才出现”的坏输入状态。已捕获 graph 的 tensor 元数据固定，替换 dtype/stride/shape 通常需要新调用或重新 capture，不把 replay 说成可任意传入新 tensor。本项目不增加值域检查，也不保证捕获后缓冲区内容变化被发现。[PyTorch CUDA graphs 说明](https://docs.pytorch.org/docs/main/notes/cuda.html#cuda-graphs)支持这一限制；实际未被捕获的调用仍有 eager 检查成本。

当前 SM90 dispatcher 的 scale 类型与布局配置由模板选择，运行时从 shape 构造布局，而非读取 scale strides；见固定源码的 [ScaleConfig](https://github.com/vllm-project/vllm/blob/91d7324cb19d301c72d849e457221ee8dd645024/csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm90_fp8_dispatch.cuh) 。支持另一布局可能需要重排/复制或另一 kernel 实例，具体方案及编译/二进制成本须另行验证，不能按几行检查的预算接受。

## 4. 与现有 Lab 的关系

本文建议为 Lab 增加“拥有并完成一个低层代码修改”的交付目标。现行 [Lab 需求](../LAB_REQUIREMENTS.zh-CN.md) R1 和外部交付目标限定他人发起的 runtime/lifecycle 线程；本项目允许自有 kernel 修复，因此正式采纳时须明确修订这些范围及其目标，不把本项目计入旧目标。

沿用源码与构建 pin、正负控制、明确的未评分结果、隐私规则和发布前核查。已有 progress、退出传播与关闭预算检查继续作为现有能力保存；是否转入维护态由单独决策记录，本文不追溯改变历史结果。仓库已有实验与源码清单可复用，不要求新增通用框架、分类器或 profiler。

上游活动继续遵守既定发布和联系约束。评审时刷新相关 issue/PR 与重复工作状态；维护者未回应不妨碍有界的本地实现，但不得将候选 API 策略表述成已获上游同意，亦不擅自扩大 #55537。

## 5. 执行门槛、预算与结果

建议总预算 **20 个工作小时**，包括设计、源码、构建、测试、文档与分析；H800 付费时间拟限制为 **一次、最多 2 小时**。这只是待确认预算，不能据此直接开卡。先做最多 **4 小时**的源码与实现可行性检查：确认 B1/B2 仍适用、选定候选契约、说明正常调用者兼容性、找出所需构建与现有测试，评估与现有 PR 的重叠。通过后才编写修改和运行协议。

预订 H800 前，patch 和回归测试须可评审，微测试 base/fix 与完整 fix 的构建方案和资源预算可行，模型及目标后端已确定，实验规则已提交。构建时间计入总预算；昂贵的构建不能隐含挤入短 GPU 会话。K6 不要求硬件性能计数器，也不因此新增 profiler 学习或权限排查任务；K5 的路径见证方法必须提前确定。

结果按不同验收项分别记录，不再用一个模糊的成功标签包办：

- **技术交付完成：** K1–K7 通过，形成可重建的 C++ 修复、回归测试与正常服务兼容性结果；上游是否合并单独记录。
- **部分完成：** 有可复现 patch/test，但服务路径、性能兼容性或构建身份仍有缺口；列出未通过或未评分的需求，不称为端到端完成。
- **停止：** 可行性检查发现现有修复已覆盖、无法在预算内建立有效契约或构建，或候选改动无法满足数值与兼容性要求；保存原因，不以增加形状、换 kernel 家族或另写框架延长本项目。

四小时门槛后必须显式决定：如果只剩约十行检查且没有上游出口，是否仍值得为学习投入余下最多十六小时；记录继续或停止的理由，不以已投入时间作理由。GPU 前再复核一次。达到预算上限时报告已完成项和缺口；继续投入需要具体新增需求与预算。独立使用、维护者采用和真实故障减少属于外部结果；如未观察到，记录 `not_observed`，不能以本地测试通过替代。

## 6. 冻结前需要补齐

1. 刷新上述四个 PR 与源码，检查入口冲突；决定 B1/B2 的放置。共享 `scaled_mm_helper.hpp` 覆盖 SM90/SM100/SM120，不能无条件加入 SM90 专用布局限制；若修改其他架构的行为，须扩大验证范围并重新批准预算，否则隔离到 SM90。
2. 给出 patch/test 和两种构建计划：微测试匹配 base/fix，以及 K5 完整 fix build。此前 B1/B2 用现成 vLLM 扩展，最小 extension 尚未验证；它必须编译真实路径、隔离注册命名并与完整 patch 对应。完整 fix build 拟在无卡或 4090 主机完成，先确认 RAM、磁盘、CUDA toolchain 和依赖；以 `TORCH_CUDA_ARCH_LIST=9.0a` 为候选配置，核查实际 CMake/nvcc 命令确实生成 SM90a，不能只记录环境变量。构建仍计入 20 小时。
3. 选定模型、请求和实际后端见证；固定 K5 输出有效性规则及 K6 eager 主机测量规则，不恢复 serving 性能 A/B。性能分析项目另立范围，不附加在本项目上。
4. 明确接受学习价值为当前主要动机、上游出口受阻及旧 Lab 需求的范围修订；四小时后再判断剩余验证是否值得。

评审草稿的首个动作是第 1 项源码与契约可行性检查。完成后再决定是否把它冻结为执行项目。

### 可行性初查记录（2026-09-29；门槛尚未通过）

- GitHub 刷新：四个 PR 均 open；#55537 head 与上述 pin 一致。#56248、#56480、#56659 head 分别为 `107d22b8a85024435296396485de9b6dbd457ead`、`f668271bf39603d1eee6d25fe85f062f8c4a8b0e`、`b89d84a94cf54984ff42df69f0ce6bd242a5af0f`。这是当日状态，不保证后续不变。
- 初选位置：`c3x/scaled_mm_blockwise_sm90_fp8.cu` 的 `cutlass_scaled_mm_blockwise_sm90_fp8`，在 output dtype 分派之前统一检查 A/B 的 e4m3 dtype 与 scales 布局。它是 SM90 专用入口，两种 output dtype 和 swapped dispatch 都经过这里；不修改共享 helper 或 #55537 的 entry 行。仍须逐一检查并行 PR diff，不能以不同文件替代完整冲突核查。
- B1 检查设计须允许 size-one 维度的物理布局等价，不机械要求所有 singleton strides 固定。有效 padded/TMA-aligned scales 是否能与当前 packed scale 寻址兼容也须逐项判断，不能仅使用 `is_contiguous()` 作为 activation-scale 条件。
- 并行 PR 的 GitHub changed-files 核查确认：三者均未修改计划中的 SM90 blockwise `.cu`；#56248 修改其下游 dispatcher，另两者修改 SM100/SM120 dispatcher。四个 PR 均修改共享测试文件，因此新增 B1/B2 测试初期放在独立文件，不修改共享测试。这里确认的是文件级无文本重叠，不是经过 merge/build 的语义兼容性。若 #56248 而非 #55537 成为实际基线，B0 应预期 honor 且正确，不能沿用本地 reject 预期。
- 正常 caller 初查：固定 head 的 `CutlassFp8BlockScaledMMKernel` 设置 `column_major_scales=True`，未启用默认 false 的 `tma_aligned_scales`，并传入 `Bs.T`。对非空二维 scales，候选 A 条件为 `(M<=1 || stride(0)==1)` 且 `(Kblocks<=1 || stride(1)==M)`；候选 B 条件为 `(Kblocks<=1 || stride(0)==1)` 且 `(Nblocks<=1 || stride(1)==Kblocks)`，即 K-major，正常 stride `(1,Kblocks)`。此前草稿把 B 条件写反；对照原始 B1 与真实 caller 后已纠正，没有执行该错误候选。条件尚未通过 CUDA 测试；shape/dtype 的既有条件必须保留。TMA padding 大于 M 且跨多个 Kblocks 时不符合 A 的布局；不因其来自量化工具就自动放行，也不声称已发现生产触发。
- 本轮未找到可运行 CUDA 编译环境：Windows PATH 无工具链，WSL 枚举返回 `E_ACCESSDENIED`。未读取旧隧道或启动实例。编译仍为未完成；下一步需要确认一个已有 Linux/CUDA 构建环境，不为此直接预订 H800。
- 已保存 [#55537 基线 patch](../../experiments/kernel-operand-contracts/pr55537-base-7b054aca.patch)，由固定 head 的 `git format-patch -1 --stdout --no-signature` 导出；文件 SHA-256：`1f3f7c463813ef9857c32db6df03b8c179acecbaabde3bc6ebe7bafd72074887`。应用 parent 为 `4f1451679088e5832bce0965a254295c854aaa09`。这只保存该次修改，完整源码/parent 及依赖仍需另行确保可重建，不把 patch 当源码归档。
- 固定 CMake 将 SM90 blockwise `.cu` 纳入 stable extension，目标为 `9.0a`。当前 Windows PATH 未发现 nvcc/cmake/ninja，未尝试构建、访问远端或开卡；这不证明远端没有工具链。
- **下一步只继续构建可行性和契约细化，不批准余下完整验证。** 未完成项：并行 diff 冲突、真实最小 extension 编译、完整 fix build 的资源估算、正常 caller 的布局等价情况与 K5 模型。完成这些或用尽四小时后，再执行 §5 的明确 continue/stop 决策。

### H800 环境预检补充（未构建、未运行 K3/K5）

本地候选 patch、最小构建驱动与独立测试已写入[准备包](../../experiments/kernel-operand-contracts/sm90-contract-repair/README.zh-CN.md)。修正 B-scale 后，12 项 CPU 检查中 11 项通过、1 项真实 CPU torch tensor 检查因本地未安装 torch 而跳过；4 项既有索引检查通过。候选 patch 应用及 ruff 检查另行复核。这些不是 C++ 编译或 GPU 验证；跳过项不可计为通过，本文仍未冻结。

用户核实指纹与主机名后，经已授权隧道只读检查：一张 H800 PCIe，显存 81559 MiB，driver 595.71.05，capability `(9,0)`；cgroup `memory.max=128849018880`（120 GiB）。系统盘剩余约 22 GiB，50 GiB 数据盘剩余约 4.4 GiB。模型占约 32 GiB，不因空间紧张自动判定其无用。

预装 torch 为 2.12.1+cu130，但旧 `vllm-fp8-venv` 已有 2.13.0+cu130、CMake 4.4.3、ninja 1.13.2 和 stable Tensor 头文件。CUDA compiler 为 `/usr/local/cuda-13.0/bin/nvcc`，版本 13.0.88；GCC 11.4。默认 PATH 中缺 nvcc/ninja，系统 CMake 3.22.1 不满足要求；下一次构建须显式使用上述匹配工具，不能只依赖镜像名。现有环境可作为候选，无需先重新安装 torch。

没有删除环境、模型或结果，没有编译最小 extension，也未证实完整 fix build 的空间需求。旧源码 checkout 不等于固定 #55537，不能用来产生本项目结果。本轮仅完成环境预检，构建及 GPU 验证继续为未完成；用户要求封存后关机。关机命令与计费停止须分开确认。


### 2026-09-30 执行路线决定

用户选择 [H800 协议 v1](../../experiments/kernel-operand-contracts/sm90-contract-repair/SESSION_PROTOCOL.zh-CN.md)：单次 120 分钟，包括预检、两臂最小构建、隔离测试与关机。本轮撤回 AutoDL 无卡和本地 WSL 构建；用户提供的 WSL 环境为 7.7 GiB RAM、2 GiB swap，nvcc/ninja/cmake 不在 PATH，python3 无 pip，不为本次验证新搭环境。本决定替代先前路线建议，不改历史证据。

仅准备/构建/回归阶段由对应本地 commit 冻结，开卡前须推送并确认公开冻结。K5/K6 与四小时 continue/stop 决策仍待完成；本地缺依赖的 CPU 检查须在预检补过且不得 skip。不声称已编译、CUDA 已验证或已有外部采用。
