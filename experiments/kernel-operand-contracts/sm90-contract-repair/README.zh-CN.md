# SM90 契约修复：构建与测试准备包

状态：2026-09-30，执行包 v1，本地提交冻结，未编译、未执行。推送并确认远端同一 commit 后才形成公开冻结，未公开前不开卡。中文版为准；[English](README.md) 是译文。对应[需求说明](../../../docs/kernel/SM90_BLOCK_FP8_PROJECT_REQUIREMENTS.zh-CN.md)。历史 B1/B2/B0 脚本不变；文件摘要见 [FREEZE_MANIFEST.json](FREEZE_MANIFEST.json)，后续语义修改须在执行前记录带日期的修订。

## 目的与边界

`candidate.patch` 只修改 SM90 blockwise `.cu`，拒绝非 e4m3 A/B 和非 packed scales。接受 singleton 维度的物理布局等价。Base/fix 都建立在固定 #55537 `7b054aca96cea8be1369d651c3434ad140580b92`，B0 两臂均预期拒绝。新增 C++ 行为未获上游授权；这里只准备本地实验，不发布 PR。

`build_minimal.py` 编译真实 entry、common.cpp、SM90 dispatch 及 FP8/INT8/AZP/blockwise kernel；只有 `binding.cpp` 的注册 namespace 改为 `lab_sm90_contract`，不复制入口判断或 kernel。不导入 vLLM，不混用旧扩展，每臂使用独立进程。最小构建保留标准路径，便于以后增加对照；本轮测试集中于 blockwise。

这不是完整 vLLM 扩展，不能计作 K5 serving 兼容性。Stable 注册、链接及编译参数仍需在 Linux 上试编译确认。编译成功只记 `built_not_validated`，失败或超时记 `unscored`。默认两路编译、45 分钟上限，超时终止编译进程组；不自动重跑或切到全量构建。超时会话须检查是否有残留编译进程。

## 本地可做的复核

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -p test_sm90_contract_build_plan.py
.venv/Scripts/python.exe experiments/kernel-operand-contracts/sm90-contract-repair/build_minimal.py --vllm-src ../vllm-55537-reject-only --arm base --plan
```

审阅 patch 时重点确认：A/B 的真实 kernel 元素类型、A-scale 的 column-major 与 B-scale 的 K-major（转置后二维 tensor 为 column-major）寻址、singleton 等价，以及 guard 是否在 output dtype/swap 分派前。B 的正确 fixture 是连续 `(Nblocks,Kblocks)` tensor 的 `.T`，stride `(1,Kblocks)`；错误 fixture 是该 view 的 `.contiguous()`，stride `(Nblocks,1)`。CPU 检查不证明 GPU 行为正确。

## Linux 构建准备（暂不执行）

在固定源码仓库里建立两个 detached worktree；不要使用远端现有旧 checkout 的 HEAD。两臂目录须是新的，fix 只应用本目录 patch、保持 unstaged，不夹带其他文件。CUTLASS 要求干净的 `v4.7.1` checkout；脚本记录其 commit 与头文件摘要，不自动联网获取依赖。保存固定源码/parent 的可重建来源，单个 patch 不替代源码归档。

```bash
git -C "$SOURCE_REPO" worktree add --detach "$BASE_SRC" 7b054aca96cea8be1369d651c3434ad140580b92
git -C "$SOURCE_REPO" worktree add --detach "$FIX_SRC" 7b054aca96cea8be1369d651c3434ad140580b92
git -C "$FIX_SRC" apply --check "$PACKET/candidate.patch"
git -C "$FIX_SRC" apply "$PACKET/candidate.patch"
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$VENV/bin:$PATH"
"$VENV/bin/python" "$PACKET/build_minimal.py" --vllm-src "$BASE_SRC" --cutlass-src "$CUTLASS_SRC" --arm base --out "$BASE_BUILD"
"$VENV/bin/python" "$PACKET/build_minimal.py" --vllm-src "$FIX_SRC" --cutlass-src "$CUTLASS_SRC" --arm fix --out "$FIX_BUILD"
```

这些变量须先设为已核对的绝对路径，不能逐字粘贴未定义变量。H800 旧 `vllm-fp8-venv` 可作为候选 `$VENV`，但先重核工具身份。脚本要求 torch 2.13.0，设置 `TORCH_CUDA_ARCH_LIST=9.0a`，保留 build.ninja 与完整日志；人工检查实际 gencode、编译/link 命令。不要安装进或覆盖既有 vLLM。构建目录必须新建，拒绝复用，原始 receipt/log 私有保存。

先只做 base 试编译；试编译时从同一 pin 的 vLLM CMake configure 所产 `compile_commands.json` 或 build.ninja 提取 SM90 blockwise `.cu` 的真实命令，与最小 extension 的 gencode、宏、语言标准、include、编译选项及 link 选项逐项比较，保存命令、文件摘要和差异。不要通过连续修 link error 来猜配置。若注册、链接、空间或时间门槛失败，记录并停止；完成 flag 复核与 patch 再评审之前不构建 fix。系统盘约 22 GiB 空闲可作为候选 scratch，但不保证两个构建能放下；数据盘 4.4 GiB 不足以批准完整构建。不得自动删除模型/源码/旧结果。使用 base/fix 的实际产物大小与耗时估算 K4/K5，再执行四小时 continue/stop 决策。

## GPU 回归（冻结后才执行）

使用[固定 H800 协议](SESSION_PROTOCOL.zh-CN.md)：从启用计费实例起最多 120 分钟，包含预检、两臂构建、测试和关机。不以本地 WSL 或 AutoDL 无卡模式作为执行备选。每项仅一次、独立新进程；每臂 collection 30 项（base 27 项实际测试加 3 项预定 skip，fix 30 项实际测试）。每个构建传 `--timeout-seconds 2100`。

`test_repair_sm90.py` 验证有效 bf16/fp16、普通与 swapped 两种 M、独立 A/B 错 dtype、独立 A/B 错 scale layout、A/B/output padded views、singleton 等价及有效 graph capture/replay。Fix 另测 dtype/A-layout/B-layout 三类 capture 拒绝，然后检查 context 可继续运行有效控制；Base 明确跳过这三类错误 capture，不捕获可能错误的 kernel。Singleton 控制只在相应 scale 的维度实际为 1 时改为 contiguous，不改动 `(1,512,1024)` 中非 singleton 的 B 布局。

冻结预测：有效输入 `rel_error < 0.005`；base 错 dtype/layout 输出非有限或 `rel_error > 0.05`；fix 必须给出对应错误；B0 两臂均拒绝。执行前仍须公开冻结并通过环境门槛。Graph replay 不重做 guard。

```bash
"$VENV/bin/python" "$PACKET/run_isolated.py" --arm base --receipt "$BASE_BUILD/build_receipt.json" --out "$BASE_RESULTS" --budget-seconds 420
"$VENV/bin/python" "$PACKET/run_isolated.py" --arm fix --receipt "$FIX_BUILD/build_receipt.json" --out "$FIX_RESULTS" --budget-seconds 420
```

运行前验证源码、CUTLASS、harness 与二进制 receipt 摘要，保存 collection 与 stdout/stderr。`harness_sha256` 包含测试和隔离 runner；测试在加载二进制之前核查自身摘要、binary 摘要、torch 版本和 arm，冻结时同时记录摘要。CUDA context 失效的案例记 unscored；下一案例使用新进程，不重跑。JUnit 是逐项结果记录，不是外部采用证明。缺失控制不能升级为修复成功。

## 尚未准备完的执行门槛

本次修订的 `test_repair_sm90.py` SHA-256：`3a3d23355696244a56f4287a98bcabf5275e31b9ff11a72b3b9a78417edd3d40`。后续修改须更新审核和 receipt。迁移后二进制通过 `binary_file` 在 receipt 所在目录定位，保留原摘要和构建机路径。pytest.ini 绑定 receipt，并通过 `-c` 固定。Singleton 控制确实需要豁免；混合 int8 是新预测。

两 suite 合计 21 项：19 通过，真实 torch tensor 和 pytest 重新收集项因本地缺依赖跳过。cwd/rootdir/config 接线与迁移二进制摘要检查通过，但不能替代真实重新收集或 binary 加载。安装 pytest 被网络策略阻止；准备环境须补跑两项且不得 skip。最小 extension/GPU 行为未验证，K5/K6 未冻结。没有远端操作、开卡或完整四小时门槛通过结论。
