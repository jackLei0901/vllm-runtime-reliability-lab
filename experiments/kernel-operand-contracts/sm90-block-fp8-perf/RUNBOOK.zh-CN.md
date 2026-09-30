# 开卡前与现场执行清单

状态：v0.7 本地冻结材料，尚未公开冻结；不是开卡批准。以[中文 charter](../../../docs/kernel/SM90_BLOCK_FP8_PERF_CHARTER.zh-CN.md)为准。§8 为当前同卡准备方案，替代 §1/§7 中要求所有远端安装与模型下载在开卡前完成的旧顺序；公开 SHA 仍必须在开卡前核对。

## 1. 不开卡时完成

1. 审阅两种分派：base 保留原谓词；variant 仅在 M≤64 时选择已有 swapped 路径。不要把 B1/B2 检查加入性能臂。
2. 在独立、同 pin 的源码副本中应用 `dispatch_variant.patch`。先 `git apply --check`；不要修改历史工作树。构建器检查两臂源码差异范围。
3. 准备 pin `7b054aca96cea8be1369d651c3434ad140580b92` 的 Python serving 安装；Stage 0 可采用下文父提交预编译路线。两份最小 GEMM `.so` 不能代替 serving 安装，不从未知 wheel 的哈希反推构建历史。
4. 下载模型到 revision `220b46e3b2180893580a4454f21f22d3ebb187d3` 的 snapshot，保留下载命令；运行 `model_preflight.py` 绑定 config、索引和权重。目录名本身不能证明下载来源。
5. 阅读各工具 `--help`，运行 CPU 测试、`--plan`；核查新输出目录、可用磁盘、NVML、Nsight Systems、CUDA toolkit 与 Torch/CUTLASS 的匹配。
6. 评估 Stage 0：先 forced Nsight，再 forced shapes，最后可选 default Nsight；无 default shapes。各次 fresh server，共用 25 分钟截止时间。另从原 Stage 0 的 30 分钟中划出 5 分钟给 marked 分派 probe/导出/检查，不挤占封存。若必需采集无法装入预算，先停，不开卡后临时扩时。
7. 全部准备审阅后执行 `freeze_packet.py --write` 与 `--check`，提交并推送。记录公开 commit；只生成 manifest 不构成公开冻结。后续工具用 `--freeze-commit` 校验公开原文。

## 2. 会话 A（上限 100 分钟）

1. 只读确认主机指纹、实际 SM90 GPU、UUID、SM 数、L2、磁盘、Torch 与 profiler；保留计费/开始时间。所有原始材料位于私有目录。
2. 在同一个 Stage 0 截止时间内按顺序运行 `serve_trace.py --configuration forced --mode nsys`、`--configuration forced --mode shapes`，若仍有时间才运行 `--configuration default --mode nsys`。每次独立输出目录，保留安装身份、来源、resolved config、日志与 trace。default 仅背景，缺失不得作默认后端性能结论；不阻塞已完整的 forced 见证。
3. 用 `nsys export --type sqlite` 导出，不修改原报告。按 `witness_review.py` 的 schema 建立人工复核的 capture-op → replay graph-node 映射。只在 forced 配置对每种并发检查至少十个完整 decode step、每 step 的 36×4 线性投影及全部 GPU kernel；default 后端不适用同一启动数/名称规则。不能只搜到 CUTLASS 名称就算通过。
4. 并发 63 不能直接作为 M=63：需 resolved graph 配置及实际图证据支持 M=64 padding。缺少 `Input Dims`、graphNodeId、完整 step 或跨采集对应证据时停止计时评分。人工映射本身不称为自动证明。
5. 在原构建预算内运行 base builder。用明确的 5 分钟 probe 预算通过 Nsight 运行 `identity_probe.py`，再由 `trace_tools.py` 验证三次 marked launch 的实际 kernel 类型；不能只验证 namespace 字符串。
6. 见证均通过后运行 `timing.py --session A`。保存全部固定块；时钟或噪声剔除不补跑。工具内计时工作窗为 30 分钟，整个会话仍受 100 分钟上限约束。
7. 只有完整 A 的边界差距门槛通过才安排 B。`no_gap` 为有效停止；证据缺失为 `insufficient`，不能改称无差距。

## 3. 条件会话 B（上限 70 分钟）

1. GPU 型号、SM 数、L2、NVIDIA driver、CUDA driver API、nvcc 与 Torch 须与 A 一致；允许不同物理 UUID，记录两者及 A 门槛来自另一张卡的限制。环境/模型/工具/基线身份不变，构建 variant。B 的双库 marked trace、正确性和计时须在同一张 B 卡执行，排除符号互相覆盖。
2. 运行 `correctness.py` 固定 176 臂用例；数值失败或 CUDA 错误保留，不修参数后补跑。
3. 全部正确性通过后运行 `timing.py --session B`；重新 A-A 校准，不跨会话相减。保留改变与未改变 cell 的退化检查。
4. `summarize.py` 仅生成公开 allowlist 摘要。原始时钟、UUID、日志、路径、SQLite、Torch capture 与 profiler 报告私有保留。人工复核摘要再发布。
5. 下载并核验记录后按用户授权关机；到平台控制台确认停机及停止计费。工具本身不自动关机。

## 4. 学习与结论边界

开卡前的源码/设计笔记（不是 GPU 结果，先由你复核并补充自己的解释）：

- N=4096 时 M64 原路径的逻辑 tile 数为 ceil(64/128)×ceil(4096/128)=32；M63 的交换路径为 ceil(4096/128)×ceil(63/16)=128。persistent scheduler 下 tile 不等于启动 CTA；更多 tile 不保证更快，须读实际 grid 与配对时间。
- 不同 M 保持相同 B 与 B-scales、固定输入 seed，避免把权重内容或 layout 差异混进分派比较；权重生成器独立于 A 的随机数消耗。
- 热权重与轮换权重使用相同 GEMM、调用数和配对规则，隔离缓存条件。轮换 footprint 大于两倍 L2 是设计条件，不是每次 HBM 读取的实证；无 counter 时保留缓存假设。

冻结前把你的确认/修正加入本节；测量后一起读原始配对块和 kernel grid。先判断证据是否成立，再讨论分派修改。

这是 forced CUTLASS fallback 的局部性能研究。即使出现 kernel 改善，也不是默认后端或端到端 serving 的加速证明。预编译安装可用性、CUDA 编译、profiler 实际 schema 与跨采集图映射目前均未验证；若不能完成，记录阻塞，不通过降低见证标准来继续。

## 5. Stage 0 预编译安装路线（尚待可用性确认）

父提交是 `4f1451679088e5832bce0965a254295c854aaa09`。同 pin 的 setup.py 支持显式预编译来源；已读父提交到 pin 的 C++ diff，仅 `scaled_mm_entry.cu` 的 CPU operand checks 改变，dispatcher/内核不变。父 wheel 不含这些检查，不能称为完整 pin 二进制，也不能用于 K5；性能最小两臂不变。

先在不开卡的联网环境检查确切索引：

```powershell
Invoke-RestMethod 'https://wheels.vllm.ai/4f1451679088e5832bce0965a254295c854aaa09/cu130/vllm/metadata.json'
```

只有索引给出兼容 Linux x86-64 的 cu130 wheel，下载后 METADATA 指明 vLLM 父 commit 与 torch==2.13.0，且目标已安装 Torch 2.13.0+cu130，才采用此路线。URL 从 metadata 的 path 解析，不能猜测文件名或改用 latest。保留完整 wheel 与 SHA-256。

用 `wheel_preflight.py --out <新私有目录>` 完成下载和归档检查，无需 Torch/GPU，也不执行归档代码。最多下载 8 GiB；每次网络操作 socket timeout 30 秒，每份下载总时间上限 30 分钟（加最多一次在途 socket 等待）。小块读取，每两秒输出已接收量，保留失败/部分下载，不覆盖；支持已有归档的 `--index <原索引> --wheel <归档>` 离线核查。成功状态仅为 `archive_prepared_runtime_unverified`，输出 private `wheel_receipt.json`；不是安装来源记录，也不证明 ELF/CUDA 加载兼容。下载后仍要在目标 Linux 环境验证安装和扩展身份。

在同 pin 的独立 Linux serving checkout 内，显式指定本地归档，例如：

```bash
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION=/private/parent-cu130.whl uv pip install -e .
```

保留完整安装日志与环境 lock。`--serving-build` JSON 的预编译路线字段：`kind=precompiled_parent`、`source_pin`、`wheel_commit`、`wheel_url`、`wheel_path`、`wheel_sha256`、`wheel_index_url`、`wheel_index_path`、`wheel_index_sha256`、`extensions_sha256`、`build_command`、`build_log`、`build_log_sha256`。扩展哈希来自实际安装；collector 逐一与保留 wheel 成员核对，同时检查源码 parent/C++ 差异和 Torch/CUDA。原完整源码 build 路线仍可用，但不得隐含追加预算。

用户返回的 cu130 索引包含父 commit 的 `manylinux_2_28_x86_64`、cp38/abi3 条目，版本为 `0.29.1rc1.dev487+g4f1451679`。其 path 为 `../../../<parent>/<wheel>`，正确解析到根目录父 commit 下，而非 `/cu130/` 子目录；variant 为空。预检因此绑定原始索引及其 SHA-256，再用 urljoin 解析唯一条目，不把根目录 alias 当成随意 fallback。这里只确认索引发布了条目；归档下载、torch 依赖、CUDA/ABI 和实际安装仍未验证。若 metadata/ABI 不匹配，停在开卡前，另定构建计划。

已下载的归档内部 WHEEL 为 `cp38-abi3-linux_x86_64`，与索引/文件名的 manylinux 标签不同。预检只允许这两个明确的 x86-64/abi3 标签并记录差异；generic Linux 不能提供 glibc 2.28 下限保证。归档 METADATA 已读到 `torch==2.13.0` 与 `Requires-Python: <3.15,>=3.10`。这不是 ELF 动态依赖、CUDA 加载或 serving 运行验证；Linux 安装预检仍必需。首次严格标签检查失败回执原样保留，使用相同原始归档离线复核，不重新下载。

第二次离线失败来自工具误要求旧 `vllm/_C.abi3.so`。pin 的 CMakeLists.txt 将 `_C` 限于 HIP；CUDA platform 加载 `_C_stable_libtorch`，归档含该扩展。已纠正 CUDA 必需扩展集合，不通过猜测补入不存在的模块；第二次失败也保留。工具修订只影响下载/归档预检，不修改候选 C++、GPU 矩阵或性能阈值。

## 6. Linux 预检当前状态

同一归档的 `_C_stable_libtorch.abi3.so` 与 `_moe_C_stable_libtorch.abi3.so` 已做只读 ELF64 section 检查（未加载二进制）：两者 DT_NEEDED 均含 `libcudart.so.13`、`libcuda.so.1`、`libtorch.so`、`libtorch_cpu.so`、`libtorch_cuda.so`；最高已声明 GLIBC 为 2.14，GLIBCXX 为 3.4.21，CXXABI 为 1.3.9。二者没有 RPATH/RUNPATH。此结果只覆盖两个主要扩展，不覆盖所有传递依赖或整个 wheel；它支持 CUDA 13 runtime 路线，不证明目标驱动能加载。

到 Linux 后以 `readelf -d <extension>` 和 `readelf --version-info <extension>` 交叉核对；确认 Python 3.12、Torch 2.13.0+cu130、glibc/libstdc++、依赖目录，再安装显式父 wheel。无卡环境若没有 `libcuda.so.1`，需记录驱动缺失，不能将该失败判为 wheel ABI 不兼容，也不能通过 driver stub 伪造运行通过。

本机 WSL 在 agent 权限下返回 `Wsl/Service/E_ACCESSDENIED`。用户已建 Python 3.10.12 虚拟环境并确认官方 cu130 索引有 Torch 2.13.0+cu130；没有完成本机 serving 安装。这条路线不再继续下载整套依赖：本机 Python 版本不同，不能代替目标 Python 3.12 的安装与身份核查。目标主机的新证据见下一节；不能把基础 CUDA 检查升级为 vLLM 扩展加载或性能验证。

## 7. 2026-09-30 预检总结与开卡门槛

结论：硬件可用，实验未运行，尚不批准再次开卡。charter、执行工具和测试仍未提交，`FREEZE_MANIFEST.json` 尚未生成，不存在本性能实验的公开冻结。此前进入付费主机预检未先检查公开冻结门槛，是执行流程错误，不是协议通过；不得补写为预注册的 GPU 结果。

已确认的目标环境：

| 项目 | 预检结果 | 不代表什么 |
| --- | --- | --- |
| GPU/内存 | H800 PCIe，SM90，114 SM；cgroup 内存 120 GiB | 不证明性能差距存在 |
| Torch/CUDA | 旧环境 Torch 2.13.0+cu130、CUDA runtime 13.0；简单 GPU 张量加法及同步通过 | 不证明父 wheel 或候选扩展能加载 |
| 工具 | nvcc 13.0.88、Nsight Systems 2025.3.1.0、uv 0.12.9 已存在 | 未验证 profiler 采集/schema 和性能构建器 |
| 同 pin 源码 | 旧修复项目的源码 HEAD 为本实验 pin，git status 为空；真实 CMake reference 存在 | 不把历史修复库当作本次性能两臂 |
| 空间 | 系统盘约 22 GiB 空闲，数据盘约 4.4 GiB 空闲；旧模型约 32 GiB | 尚未批准删除旧模型，也未证明峰值空间足够 |

仍需在不开 GPU 时解决：

1. 网络：目标主机直接访问固定 revision 的 Hugging Face config 超时。先试 AutoDL 学术资源加速，再检查同一 revision；下载仍须保留来源并校验 config、索引与权重。不以浮动 main 或更换模型绕过失败。
2. 模型和空间：固定 Qwen3-8B-FP8 snapshot 尚未准备。先列出安装、权重、解压、编译与 trace 的峰值空间预算；只清理明确不再需要、已核对目标的内容，保留历史源码、patch、日志和结果。无卡模式仅用于下载/安装准备，不用于 CUTLASS 编译。
3. 依赖：旧环境 `pip check` 失败（FlashInfer 缺若干依赖，torchvision 期望 Torch 2.12.1）。不能因 Torch CUDA smoke 通过就直接复用为 serving 环境；保留旧环境，准备独立环境和匹配依赖，保存安装日志与依赖记录。
4. 二进制身份：父提交 wheel 已在本地准备，但未在目标环境安装或逐扩展核对。使用显式本地归档及同 pin 独立源码副本；不默默追加完整 vLLM 源码构建，不改用 latest。
5. 公开冻结：准备记录与 CPU 工具检查完成后，生成/检查 byte manifest，提交并由用户推送；再次核对远端 commit 和 manifest 原文。满足前四项但没有公开 SHA 仍不准开卡。

本次本地重新执行 `python -m unittest discover -s tests -p 'test_sm90_perf*.py' -v`：44 项通过；对 packet 和两份性能测试执行 ruff check 通过。Stage A 的 `--plan` 给出 92 个校准 cell、16 对比较、42,336 次 replay；正确性 `--plan` 给出每臂 88、两臂 176 个用例。这些都是 CPU 准备检查，不是 GPU 测量或完整仓库 CI 通过。manifest 工具仅列出 21 份待绑定文件，未使用 `--write`，未生成冻结。

再次开卡前必须明确记录：模型回执、安装来源/扩展回执、空间预算、CPU 检查结果、公开冻结 SHA 和会话 A 的 100 分钟上限。只能随后授权会话 A；新卡的实际身份/驱动加载与 profiler 能力在 A 的预检预算中复核。A 不通过不得继续 B，也不得将证据不足改称无差距。

本次没有安装新环境、下载模型、编译性能臂、采集 vLLM trace 或测量延迟，没有删除任何文件。已经按用户授权对已核对的主机发出 `shutdown -h now`，连接随后断开且重连被拒绝；平台停机和停止计费仍需用户在控制台确认。详细主机/路径/UUID/原始输出保留在私有预检 JSON，不发布到本仓库。

## 8. 当前执行顺序：本地冻结 → 同卡 P → A → 条件 B

用户选择不再为准备释放/重新寻找 H800。此变更只调整安装下载位置及预算，不改变源码 pin、模型、GPU 矩阵、缓存条件、数值界或性能判定阈值。P 最多 60 分钟（末 5 分钟留失败封存）；A100、条件 B70，整次最多 230 分钟。P 不通过或 A 无完整差距证据，立即封存关机。

开卡前在本地完成：

1. 保留原 metadata 和已经下载的父 wheel，不再下载 Torch 到本机；归档固定 SHA-256 是 `f50bf6c6c63785be641d8512139593ebeb20f7c1e9bf290dc7864de414909128`。
2. 整理协议/全部工具及 CPU 检查，生成 manifest；提交并由用户推送。核对公开 commit 的 manifest 与本地相同；记录完整 40 位 SHA 后才批准开卡。
3. 用私有上传清单绑定 packet、wheel、metadata、manifest 和源码/patch 的字节。上传工具不改历史源码或结果；独立 serving、base、variant 副本。目标机允许复用已确认的同 pin CMake reference 与工具，但复核其 hash，不盲信旧路径。
4. 预写下载、安装、清理与关闭方案。模型采用唯一固定 revision；系统盘/数据盘各自核算空间，不认为两盘空闲可以任意合并。系统盘约 22 GiB 仅是上次观测，不是空间保证。不足时只清理明确不再需要且已记录的旧模型/缓存，不删除历史源码、patch、构建回执或原始结果；不能确定目标则停下询问。

同卡阶段 P 的顺序：

1. 确认 SSH 指纹、主机和 GPU，记录开始时间；核对本地/公开/上传 manifest。探测加速网络（`source /etc/network_turbo`），固定模型 config 的 20 秒探测失败则记录，不长时间空等。加速代理只用于外部下载，不代理 localhost serving 请求。
2. 下载模型并运行 `model_preflight.py`。保留下载命令和 HF snapshot 路径，目录名不得代替身份校验。配置、索引、权重全部核验后才算完成。
3. 不修改历史 fp8 环境。新建 Python 3.12 独立环境，使用显式父 wheel 安装同 pin 的独立 serving 源码，限制 Torch 为 2.13.0+cu130。按 pin 的依赖声明解决 torchvision、FlashInfer 等依赖；保存解析、安装命令、完整日志和包清单，不使用 `--no-deps` 掩盖冲突。需要完整源码 build、版本变更或超时则停止。
4. 运行下面的安装预检。工具不安装、不 serve、不执行 GPU 运算；它验证公开冻结、Torch/CUDA 版本、pip check、固定源码、所有 tracked Python 与 installed 扩展对父 wheel 的身份。成功仍是 `installation_identity_verified_runtime_unverified`，不是 GPU 加载或 profiler 通过。

```bash
python "$PACKET/serving_preflight.py" \
  --freeze-commit "$FREEZE_COMMIT" --vllm-src "$SERVING_SOURCE" \
  --index "$INDEX" --wheel "$WHEEL" \
  --install-log "$INSTALL_LOG" --install-command-file "$INSTALL_COMMAND_FILE" \
  --out "$WORK/installation-preflight"
```

上面变量在当次 SSH shell 绑定到已经核对的绝对路径，输出必须新建。`serving_build.json` 可直接传入 collector 的 `--serving-build`；GPU 扩展加载、实际路径和 profiler 见证仍在 A 的预算内验证。准备阶段结束需同时具备模型回执、安装回执、可用空间和剩余预算；任一不足不启动 A。不得在看到性能后修参数或补冻结。

依赖资料核查：pin 同时要求 Torch 2.13.0、TorchAudio 2.11.0、TorchVision 0.28.0。TorchAudio 的版本号不相等本身不是冲突：[官方 2.11 安装文档](https://docs.pytorch.org/audio/2.11.0/installation.html) 声明 stable ABI 支持 PyTorch 2.11 及以后。目标环境仍须实际解析、pip check 和加载，不能以文档替代安装回执。本地 shell 访问 PyPI 受工具网络限制，尚未完成全量依赖解析；这属于 P 的明确风险，而非已验证通过。

结束时按本轮用户授权下载/核验私有证据并关机，不将主机关闭自动等同于平台停止计费；由用户在控制台确认。

本地冻结材料复核：49 项性能工具测试、6 项历史 R2 回归、4 项索引测试全部通过（共 59 项），ruff check/format check 通过；不是完整仓库 CI 结论。候选 patch 在固定源码 HEAD 上 `git apply --check` 通过，没有应用到历史工作树。已下载父 wheel 的 SHA-256 再次匹配，重复下载不是 P 的步骤。`serving_preflight.py` 新增成功、错误 Torch、依赖冲突、空安装记录与无有效公开冻结的 CPU 测试；mock 通过仍不证明远端安装成功。

冻结前工具修订（无 GPU 数据）：`6cf971c` 的 Python 3.10 CI 暴露 wheel 校验依赖 3.11 才有的 `hashlib.file_digest`，现改为分块 SHA-256，校验内容不变。另发现 Windows 文本写入使本地 manifest 为 CRLF，而提交为 LF；现以独占二进制写入生成 UTF-8/LF，并以精确字节检查。旧提交保留，修订后的 manifest 必须重新提交、推送并核对；实验 pin、模型、候选 patch、阈值与预算均不变。此修订不是 GPU 结果，也不追认旧冻结通过。
