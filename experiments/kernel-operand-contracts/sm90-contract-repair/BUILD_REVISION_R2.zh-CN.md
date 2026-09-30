# 构建工具偏差 R2 与复现方法

本中文版本为准。这是运行后的公开补充，不替换原冻结协议。

## 改了什么

用户在编译前授权 R2：两臂最小构建匹配固定源码
`7b054aca96cea8be1369d651c3434ad140580b92` 生成的真实 CMake 参数。
C++17 改为 C++20；补齐 stable ABI/USE_CUDA/Py_LIMITED_API 宏、生成的
patched Torch header include、ENABLE_FP8 及相关 CUDA 参数；按固定 CMake
移除四个 Torch half-conversion 禁用宏。最小扩展不含其他内核源码，因此未复制
完整扩展对应宏；独立命名空间及 Torch loader/link 参数仍不同于完整扩展。
候选 patch、测试及阈值没有改变。

实际执行的私有 builder SHA-256：
`bb624fef3235d207f8c115fdffa313c84d086203d60dc9b208757e777da7d3b9`。
公开后继 [build_minimal_r2.py](build_minimal_r2.py) 的 SHA-256：
`a16daa2bafc476a25aea35e51de439168a205516578ac9b1e24f0e2007ed2168`。
唯一差异是在自身 receipt 摘要处用 `Path(__file__).name` 替代字面量
`"build_minimal.py"`，从而绑定正确公开文件名；编译参数未变。
CPU 测试逆转这一处替换，即得到实际执行版本的精确哈希。公开后继尚未产生新的
GPU 结果。原 builder/manifest 不变；新 receipt 绑定后继，历史 receipt 不改。

## 可移植准备

使用 Linux、Python 3.12、torch **2.13.0+cu130**、CUDA toolkit **13.0**、
干净的 CUTLASS **v4.7.1** 及精确 vLLM pin。本轮为 GCC 11.4、CMake 4.4.3、
Ninja 1.13.2；两臂编译和测试都在租用的 H800 主机进行，不是在 WSL 或无卡模式。
编译用两个并发任务，每臂上限 35 分钟。

运行前将 `PACKET`、`VENV`、`CUDA_HOME`、`SOURCE_REPO`、`BASE_SRC`、`FIX_SRC`、
`CUTLASS_SRC`、`REFERENCE`、`BASE_BUILD`、`FIX_BUILD`、`BASE_RESULTS`、
`FIX_RESULTS` 设为已确认的绝对路径。参考及输出目录必须新建。先取得精确 pin；
若不能取得就停止，不换 main。保存的 #55537 patch 不是完整源码归档。

```bash
git -C "$SOURCE_REPO" worktree add --detach "$BASE_SRC" 7b054aca96cea8be1369d651c3434ad140580b92
git -C "$SOURCE_REPO" worktree add --detach "$FIX_SRC" 7b054aca96cea8be1369d651c3434ad140580b92
git -C "$FIX_SRC" apply "$PACKET/candidate.patch"
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=9.0a
cmake -S "$BASE_SRC" -B "$REFERENCE" -G Ninja \
  -DVLLM_PYTHON_EXECUTABLE="$VENV/bin/python" \
  -DVLLM_TARGET_DEVICE=cuda -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

CMake 可能获取 pin 指定的依赖；这一步只配置，不编译完整扩展。不能盲用旧缓存。
本轮旧 FlashMLA 缺少 `sparse_prefill.cpp`，在新目录取得固定的
`0eee43b12f034b657133cf2afca6a72ebb6efccf` 后，传入
`-DFETCHCONTENT_SOURCE_DIR_FLASHMLA=<新目录>`，配置成功。这是需要时可用的方式，
不是硬编码的机器路径。

`REFERENCE` 必须含真实 `compile_commands.json` 和生成的 `torch_patched_headers`。
builder 检查关键参数并记录命令文件/header 哈希；header 来自固定 CMakeLists.txt
对 torch 2.13 的修补，不能手写替代。仍须核对来源及有效命令：参数检查不能认证
任意提供的参考目录。

```bash
"$VENV/bin/python" "$PACKET/build_minimal_r2.py" --vllm-src "$BASE_SRC" \
  --cutlass-src "$CUTLASS_SRC" --cmake-reference "$REFERENCE" --arm base \
  --out "$BASE_BUILD" --timeout-seconds 2100
"$VENV/bin/python" "$PACKET/build_minimal_r2.py" --vllm-src "$FIX_SRC" \
  --cutlass-src "$CUTLASS_SRC" --cmake-reference "$REFERENCE" --arm fix \
  --out "$FIX_BUILD" --timeout-seconds 2100
"$VENV/bin/python" "$PACKET/run_isolated.py" --arm base \
  --receipt "$BASE_BUILD/build_receipt.json" --out "$BASE_RESULTS" --budget-seconds 420
"$VENV/bin/python" "$PACKET/run_isolated.py" --arm fix \
  --receipt "$FIX_BUILD/build_receipt.json" --out "$FIX_RESULTS" --budget-seconds 420
```

已提供独立复现方法，但尚未观察到非作者构建或执行，K7 独立使用不计为通过。
历史日志及 receipt 含主机路径，保持私有；builder 本身没有需要保密的部分。
外部复现会产生自己的 receipt。

## R1 证据

传输后的源码 HEAD/内容正确，但无 Git index。修复用
`git -C <source> read-tree HEAD`，随后运行
`git -C <source> diff --exit-code HEAD` 并检查干净状态。
`source-diff-r1.txt` 保留 diff 的退出码 0；`base-plan-r1.json` 和两份构建
receipt 记录后续源码验证，其中 base 门禁要求状态干净。
归档中 index 数量 **0**、HEAD tree 数量 **7266** 是 R1 **前**的失败状态，
不是修复后状态。私有归档摘要见[结果记录](RESULT_2026-09-30.zh-CN.md)。
