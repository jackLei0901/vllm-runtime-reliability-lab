# pytorch/pytorch#194434 修复验证手册

状态：**等待维护者通知；现在不执行，不租 GPU。**

目标是验证维护者负责的
[`pytorch/pytorch#194434`](https://github.com/pytorch/pytorch/pull/194434)
是否覆盖 [`pytorch/pytorch#196996`](https://github.com/pytorch/pytorch/issues/196996)
报告的两个单卡触发方式，以及它们在 rank-divergent 作业中的影响。

本手册不要求从源码构建 PyTorch，也不把手工覆盖 site-packages 得到的结果当成
正式验证。

这次运行是维护者测试的独立 cross-check，不是 #194434 的合入门槛。默认等待 PR
合入 nightly 后再运行；只有 weifengpy 或其他维护者明确请求时，才针对合入前的指定
构建执行。

## 1. 启动条件

只有满足以下任一条件才开始：

1. 维护者提供包含 #194434 的可安装 wheel；
2. #194434 已合入，并进入官方 nightly；
3. 维护者明确要求针对某个指定构建进行复核。

当前 PR 仍为 draft，分支和提交可能变化。不要把本手册撰写时看到的 PR head 当作
最终验证对象。

若维护者只要求快速确认，可以先完成单卡矩阵。双卡影响验证不应阻塞维护者合入，
除非他们明确要求。

## 2. 预注册判定

### 单卡修复判定

| case | 修复后预期 |
| --- | --- |
| `control` | 退出 0；rank 0 输出 `status=completed` |
| `last-microbatch` | 退出 0；不出现 uniform-gradient-dtype assertion |
| `unused-placeholder` | 退出 0；不出现 uniform-gradient-dtype assertion |

### 双卡影响判定

| case | 修复后预期 |
| --- | --- |
| `control` | 两个 rank 均输出 `status=completed`，退出 0 |
| `rank-divergent-unused` | 两个 rank 均输出 `status=completed`，退出 0，不达到外部时间边界 |

本轮验证只回答“报告中的复现是否被修复”。它不证明 #194434 的全部 FSDP2 dtype
语义，也不替代该 PR 自己的测试套件。

## 3. 先确认测试构建确实包含修复

在 GPU 环境中创建一个独立虚拟环境。优先使用维护者给出的 wheel 或官方 nightly
索引，不复用带有手工修改的旧环境。

```bash
python -m venv /root/venvs/fsdp-dtype-fix
source /root/venvs/fsdp-dtype-fix/bin/activate
python -m pip install --upgrade pip
```

根据维护者提供的方式安装 PyTorch。安装后记录：

```bash
python - <<'PY' | tee fix-version.txt
import torch

print("torch:", torch.__version__)
print("torch_git:", torch.version.git_version)
print("cuda:", torch.version.cuda)
print("nccl:", torch.cuda.nccl.version())
print("gpu_count:", torch.cuda.device_count())
PY

python -m torch.utils.collect_env > fix-collect-env.txt
```

同时记录：

- #194434 当时的 head 或 merge commit；
- wheel 文件名、来源和 SHA-256；
- 安装后的 `torch.version.git_version`；
- 维护者确认该构建包含修复的链接或说明。

如果无法证明构建包含 #194434，结果分类为 `inconclusive_build_identity`，不得写成
修复失败或修复通过。

## 4. 固定复现脚本

使用本目录已经提交的 `reproducer.py`，运行前记录它的哈希：

```bash
sha256sum reproducer.py | tee reproducer.sha256
```

不要为了适应修复改动测试脚本。若新 API 要求修改 reproducer，先保存原脚本和原
预期，单独记录协议偏差，再决定是否继续。

#194434 在 no-sync 阶段会把 `.grad` 暴露为部分归约的 DTensor，但当前 reproducer
只在最终同步 backward 完成后通过 `optimizer.step()` 使用梯度，不会在 no-sync
microbatch 之间读取 `.grad`，因此不需要为该语义调整脚本。

## 5. 单卡矩阵

```bash
set -o pipefail
export CUDA_VISIBLE_DEVICES=0

timeout -k 5s 40s torchrun --standalone --nproc-per-node=1 \
  reproducer.py --case control 2>&1 | tee fix-single-control.log
rc=${PIPESTATUS[0]}; echo "$rc" > fix-single-control.rc

timeout -k 5s 40s torchrun --standalone --nproc-per-node=1 \
  reproducer.py --case last-microbatch 2>&1 \
  | tee fix-single-last-microbatch.log
rc=${PIPESTATUS[0]}; echo "$rc" > fix-single-last-microbatch.rc

timeout -k 5s 40s torchrun --standalone --nproc-per-node=1 \
  reproducer.py --case unused-placeholder 2>&1 \
  | tee fix-single-unused-placeholder.log
rc=${PIPESTATUS[0]}; echo "$rc" > fix-single-unused-placeholder.rc
```

立即检查：

```bash
grep -H -E 'REPRO_RESULT|uniform gradient dtype|Traceback' fix-single-*.log
printf 'control='; cat fix-single-control.rc
printf 'last-microbatch='; cat fix-single-last-microbatch.rc
printf 'unused-placeholder='; cat fix-single-unused-placeholder.rc
```

停止条件：

- control 非 0；
- 任一运行达到外部时间边界；
- 任一 affected case 仍出现确切 mixed-dtype assertion；
- 环境或脚本身份无法确认。

满足停止条件后保留结果，不继续双卡，并将结论限定到实际失败的 gate。

## 6. 双卡影响矩阵

单卡矩阵通过，并且有两张可见 GPU 时再执行：

```bash
set -o pipefail
export CUDA_VISIBLE_DEVICES=0,1

timeout -k 10s 70s torchrun --standalone --nproc-per-node=2 \
  reproducer.py --case control 2>&1 | tee fix-two-rank-control.log
rc=${PIPESTATUS[0]}; echo "$rc" > fix-two-rank-control.rc

timeout -k 10s 70s torchrun --standalone --nproc-per-node=2 \
  reproducer.py --case rank-divergent-unused 2>&1 \
  | tee fix-two-rank-divergent-unused.log
rc=${PIPESTATUS[0]}; echo "$rc" > fix-two-rank-divergent-unused.rc

ps -ef | grep -E '[t]orchrun|[r]eproducer.py' \
  | tee fix-processes-after.txt
```

双卡通过要求：

- 两个 case 均在 70 秒内退出 0；
- 每个 case 的两个 rank 都出现一次 `status=completed`；
- 不出现 uniform-gradient-dtype assertion；
- 没有残留 `torchrun` 或 `reproducer.py` 进程。

## 7. 结果分类

| 分类 | 条件 | 对外表述 |
| --- | --- | --- |
| `pass_single_and_multi_rank` | 单卡和双卡矩阵全部通过 | 报告的两个 trigger 与 rank-divergent impact 均未再复现 |
| `pass_single_rank_only` | 三个单卡 case 通过，双卡未执行 | 单卡 assertion 已修复；多 rank impact 未复核 |
| `fail_reproduced` | 构建身份成立，affected case 仍命中原 assertion | 修复未覆盖该 trigger；附最小日志 |
| `inconclusive_build_identity` | 无法证明构建包含 PR | 不评价修复 |
| `inconclusive_environment` | control、CUDA、NCCL 或清理异常 | 不评价修复 |

不得仅凭“没有 traceback”判定通过；必须同时满足退出码和 `status=completed`。

## 8. 证据与公开回复

私下保留完整日志；公开前只导出：

- PyTorch 版本和 git revision；
- #194434 的固定提交；
- reproducer SHA-256；
- 五个 case 的退出码与完成标记；
- assertion 是否出现；
- 双卡运行是否有残留进程。

建议在 #194434 的公开回复保持简短：

```text
Verified the reproducer from #196996 against <build identity>.

- control: <result>
- last-microbatch: <result>
- unused-placeholder: <result>
- rank-divergent-unused (2 ranks): <result or not run>

The tested reproducer SHA-256 was <hash>. Full environment and bounded raw
logs are retained; no site-packages files were modified for this run.
```

只有维护者修复合入，或明确用其他处置关闭 #196996 后，才把 lab 的 upstream
记录标记为 resolved。
