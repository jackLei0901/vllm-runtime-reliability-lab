# PR #52365：开卡前配置复核

状态：**离线可行性复核后决定不开卡**，2026-09-29；不是 A1 修订，尚未运行任何 vLLM 测试。[英文版](PRE_GPU_REVIEW_2026-09-29.md)记录相同决定。

## 已复核但未使用的单张 24 GiB GPU 配置

- 模型：`Qwen/Qwen3-4B-Instruct-2507`，revision `cdbee75f17c01a7cc42f958dc650907174af0554`。[公开配置](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/config.json)写明 BF16、36 层、8 个 KV head、head 维度 128、原生上下文 262,144 token；[模型卡](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)给出 vLLM 服务方式。Lab 过去用过这一模型架构，但没有在这些提示长度或 #52365 固定点验证过。
- `--max-model-len 65536`、`--gpu-memory-utilization 0.9`、单卡；不用量化、上下文上限覆盖、人工 sleep 或改过的 vLLM 源码。
- 固定筛选长度：`8192 16384 24576 32768 40960 49152 57344 65535`。末档加 1 个输出 token 恰好不超过 65,536。若首次 ≥70 秒出现在末档，结论是 `no_candidate`，不能加档。runner 在首个筛选点后恰好再测一档即停止。
- 显存计算只是下界，不保证能启动：BF16 权重约 7.49 GiB（[4,022,468,096 个参数](https://huggingface.co/api/models/Qwen/Qwen3-4B-Instruct-2507) × 2 字节）；65,536 token 的 BF16 KV cache 为 9 GiB（`36 层 × 2 × 8 KV head × 128 × 2 字节 × 65,536`）。按 24 GiB 显存的 90% 预算计算，只余约 5.1 GiB 给 CUDA 上下文、激活、图捕获、workspace 等。`--max-num-batched-tokens 65536` 很大，即使下界能放下也可能启动失败。81,920 token 的 KV cache 为 11.25 GiB，余量仅约 2.9 GiB，不能作为 24 GiB 卡上的临时加档。

## 计算量筛选与决定

最长档为 65,535 token，粗略 prefill 计算量：线性层约 `2 × 4.022e9 参数 × 65,536 = 5.27e14` FLOPs；因果注意力约 `2 × 65,536² × (32 个 query head × 128) × 36 = 1.27e15` FLOPs；合计约 **1.79e15 FLOPs**。线性项只是近似：并非每个参数都会对每个提示 token 做矩阵乘法。NVIDIA 给出 [RTX 4090 的 BF16 Tensor 峰值为 165.2 TFLOPS（FP32 累加）](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)。假设持续有效吞吐分别为 130、100、50 TFLOPS，纯算术时间约 14、18、36 秒。仅凭这些 FLOPs 达到整请求 70 秒筛选线，需要约 25.6 TFLOPS 的有效吞吐；达到 80 秒需约 22.4 TFLOPS。

这些是**源码与算术推断，不是延迟实测，也不是严格耗时上界**。实际吞吐、显存流量、图捕获开销和具体实现尚不确定；整请求耗时也不等于 `wait_for_gpu_event` 耗时。但这个估算无法为 A1 所需的**单次完成等待 ≥65 秒**提供可信余量；再加上启动 OOM 风险，花最多三小时开卡不划算。**决定：本配置不开卡；预测可能得到 `no_candidate`，但绝不把它写成已运行的阶段 1 结果。** 这也不能证明 PR 默认超时对所有负载都无害。

只有找到*自然发生*的另一种单步骤配置，且开卡前按模型结构、长度和硬件估算达到约 **80 秒或以上**，所选 GPU 能容纳原生上下文及权重、KV cache、激活，并且收益足以覆盖成本，才重新考虑 GPU。须先另写并提交固定模型／硬件／长度配置；不能靠共享或限速 GPU 制造长等待，也不能见到数据后修改 A1 的评分规则。若找不到，此候选保留为有源码依据的普通 review，不计 Lab 交付。

如果本配置曾运行，固定梯度未找到有足够余量的完成请求时，A1 会要求记为 `no_candidate`。下方代码与命令仅作**未执行的提案**保留，不表示任一阶段已经完成。即使整请求超过 70 秒，也只是在筛选关闭上限臂的单次实测等待。

仅在将来另行审核并批准这个具体配置后，下列才是拟用的阶段 1 命令：

```bash
python experiments/pr52365-event-timeout/runner.py sweep \
  --python /path/vllm-base/.venv/bin/python --vllm-src /path/vllm-base \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --model-revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --max-model-len 65536 \
  --lengths 8192 16384 24576 32768 40960 49152 57344 65535 \
  --work-dir /private/p52365/sweep
```

只有结果为 `candidate_found` 才进入阶段 2；`ab` 命令的 `short`、`long` 必须直接取自该回执，不能在成功档位中人工挑选。

## 如将来重新提出开卡，仍须满足

1. 确认实际显存规格；若不是 24 GiB，须在开卡前另选并提交配置，不能直接套用本序列。确认只分配一张 GPU，不是 TP=2。
2. 确认该模型 revision 可取得，完整 BF16 权重已缓存或能在总计三小时的预算内下载，磁盘和内存足够。不要输出令牌或凭据。
3. 在远端记录 `nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader`、`command -v ninja`、`df -h`、`free -h`、Python 版本和基线／PR 检出的 SHA。runner 自带的身份和现有 `sitecustomize` 检查仍是硬门槛；看到结果后不得绕过。
4. 开卡前提交选定配置，确认 Lab 提交已公开，runner、钩子、测试和 A1 都是已审核的版本。各阶段使用新的私有工作目录，服务日志及逐次等待轨迹不公开。
5. 仅一次会话，含安装最多三小时。若固定配置启动时显存不足，记录装置 `unscored` 并停止，不能在看到 OOM 后临时缩小配置重跑。

本文既不请求也不批准现在开卡。
