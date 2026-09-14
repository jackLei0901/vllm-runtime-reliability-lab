# FSDP2 dtype issue 验证手册

状态：**2026-09-14 初始矩阵已执行；最终 flag-off 单卡复核待执行。** 初始
`last-microbatch` 和 control 运行错误地统一开启了 unused-parameter API，不能用于
证明 trigger 1 与该 API 无关。最终结果以复核后的结果说明为准。

## 1. 验证目标

一个根因包含两个单卡触发方式。最终脚本只在第二类以及双卡 divergent case 中开启
`set_reduce_scatter_unused_params(True)`：

1. `last-microbatch`：某参数只在最后一个 microbatch 首次产生真实梯度；
2. `unused-placeholder`：某参数始终未使用，由
   `set_reduce_scatter_unused_params(True)` 提供零占位梯度。

二者都可能让最终同步 backward 同时包含：

- 已累积并提升为 `reduce_dtype=float32` 的梯度；
- 仍为 `param_dtype=bfloat16` 的新梯度或零占位。

`rank-divergent-unused` 用双卡验证分布式后果：rank 1 局部报错，而 rank 0
已经进入 reduce-scatter 并等待。

## 2. 执行前记录

在实际运行 PyTorch 的虚拟环境中执行：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch_git:", torch.version.git_version)
print("cuda:", torch.version.cuda)
print("nccl:", torch.cuda.nccl.version())
print("gpu_count:", torch.cuda.device_count())
PY
python -m torch.utils.collect_env > collect-env.txt
```

先使用当前 nightly。若需安装一个新的 CUDA 13.0 nightly 环境：

```bash
python -m venv /root/venvs/fsdp-dtype-nightly
source /root/venvs/fsdp-dtype-nightly/bin/activate
python -m pip install --upgrade pip
python -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130
```

若现有环境已经能导入 nightly PyTorch，不要重复下载。

## 3. 单卡验证

进入本目录后执行：

```bash
set -o pipefail

timeout -k 5s 40s torchrun --standalone --nproc-per-node=1 \
  reproducer.py --case control 2>&1 | tee single-control.log
echo "single-control rc=${PIPESTATUS[0]}"

timeout -k 5s 40s torchrun --standalone --nproc-per-node=1 \
  reproducer.py --case last-microbatch 2>&1 | tee single-last-microbatch.log
echo "single-last-microbatch rc=${PIPESTATUS[0]}"

timeout -k 5s 40s torchrun --standalone --nproc-per-node=1 \
  reproducer.py --case unused-placeholder 2>&1 | tee single-unused-placeholder.log
echo "single-unused-placeholder rc=${PIPESTATUS[0]}"
```

预期（执行前预测）：

| case | 预期 |
| --- | --- |
| `control` | 退出 0，出现 `status=completed` |
| `last-microbatch` | 非 0，出现 `expects uniform gradient dtype` |
| `unused-placeholder` | 非 0，出现同一 dtype assertion |

若 control 失败，或两个 trigger 没有命中同一 assertion，停止，不执行双卡。

## 4. 双卡验证

```bash
timeout -k 10s 70s torchrun --standalone --nproc-per-node=2 \
  reproducer.py --case control 2>&1 | tee two-rank-control.log
echo "two-rank-control rc=${PIPESTATUS[0]}"

timeout -k 10s 70s torchrun --standalone --nproc-per-node=2 \
  reproducer.py --case rank-divergent-unused 2>&1 \
  | tee two-rank-divergent-unused.log
echo "two-rank-divergent-unused rc=${PIPESTATUS[0]}"
```

预期（执行前预测）：

- control 两个 rank 都 `status=completed`，退出 0；
- affected 中 rank 1 输出 uniform-gradient-dtype assertion；
- rank 0 不输出 `status=completed`，并等待到 collective timeout 或外部 70 秒边界；
- 最终没有残留的 `torchrun` 或 reproducer 进程。

检查残留：

```bash
ps -ef | grep -E '[t]orchrun|[r]eproducer.py'
```

## 5. 是否可以提 issue

以下条件必须全部满足：

- 当前 nightly 上单卡 control 通过；
- 至少一个单卡 trigger 命中确切 dtype assertion；
- 双卡 control 通过；
- 双卡 affected 观察到一 rank assertion、另一 rank 未完成；
- 结果与 #160279、#170667、#183040、#174862、#194546 的边界已写清；
- 提交材料包含 nightly 的 `collect_env`、git revision 和加载的 NCCL 版本。

如果 nightly 已不复现，先定位修复 commit，不提交过时 issue。如果 #194546
或其依赖分支已经覆盖该结果，应向现有 PR 补充复现场景，而不是新建重复 issue。

## 6. 证据处理

先保留四份日志供人工复核。公开前删除主机名、用户名、绝对路径、IP、token 和环境中的
私有包信息。Issue 正文只引用必要的 rank 结果与 PyTorch assertion，不发布完整原始日志。

不要在验证前修改预期。若实际结果不同，保留原预期并单独记录偏差。
