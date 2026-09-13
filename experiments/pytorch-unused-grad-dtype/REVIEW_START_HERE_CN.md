# 两卡 FSDP2 混合梯度 dtype 实验：审核入口

状态：**首轮两卡 GPU 验证完成；预注册复现假设被实验推翻。**

## 要回答的问题

四卡 organic-hang 对照实验在 PyTorch 2.13 中触发了：

```text
FSDP reduce-scatter expects uniform gradient dtype
```

当时同时存在 PP=2、DP=2、microbatch 累积和条件参数。本实验把问题缩小为
普通两卡 FSDP2：一个 rank 不使用 conditional 参数，另一个 rank 正常使用。
它只回答 **PP/microbatch 是否为复现所必需**，不把最小复现误写成生产根因。

## 第二阶段的核心修正

源码复核表明，`unsharded_zero_grad_data` 来自
`zeros_like(self.unsharded_param)`。当 `param_dtype=bf16` 时，unused
placeholder 是 bf16；普通真实梯度也通常是 bf16。`reduce_dtype=fp32` 的转换发生在
uniform-dtype 检查之后。因此，首轮两卡实验没有复现，很可能不是 rank 数量不足，
而是输入该检查的梯度根本没有形成 dtype 差异。

后续不直接租四卡，而按以下顺序推进：

1. Gate 0：两卡记录进入 `foreach_reduce` 的实际 dtype，并用独立的已知混合
   dtype 正控制验证探针和错误分类器；
2. Gate 1：只有 Gate 0 通过后，才测试两卡 microbatch 累积；
3. Gate 2：只有前序结果确实留下拓扑问题时，才运行冻结的四卡矩阵；
4. nightly：只有受影响版本已有非循环正控制后才运行。

## 建议审核顺序

1. `reproducer.py`：确认 control 与 affected 仅有 `use_conditional` 行为不同；
2. `campaign.py`：确认精确错误分类、三次重复、原始日志不落盘和进程清理；
3. `verify_results.py`：确认预期必须在验证命令中显式声明，不能看结果后改口；
4. `README.md`：确认结论边界没有越过实验实际能证明的内容；
5. `dtype_probe_reproducer.py` / `dtype_probe_campaign.py`：确认探针在调用原始
   `foreach_reduce` 前仅输出 rank、梯度数量和 dtype；
6. `PHASE2_PROTOCOL.md`：确认每一关的预期、停止规则和升级条件已经冻结；
7. `SOURCE_AUDIT.md`：确认源码判断和正控制的证据边界；
8. `tests/test_unused_grad_dtype_campaign.py`：确认错误分类和结果集验证会失败关闭。

## 重点审核问题

- `fully_shard` 的粒度能否稳定产生一个完整的 unused parameter 梯度槽？
- affected/control 是否只有 rank 1 是否执行 conditional layer 这一项差异？
- 对照组是否足以排除安装、NCCL、CUDA 或 FSDP API 本身不可用？
- 精确 marker 是否会随 PyTorch 文案变化而把真实复现误判为 unexpected failure？
- runner 在启动早期失败、超时和正常退出三条路径上是否都不会留下进程？
- 2.13 与 nightly 是否必须分别存放结果并分别声明预期？

## 暂不包含

- 不自动安装 PyTorch；
- 不运行四卡 PP/DP 原始复现；
- 不采集 Flight Recorder 或 py-spy；这个断言会快速失败，不是 hang；
- 不提交 PyTorch issue 或补丁；需先完成受影响版本与 nightly 的成对验证；
- 不修改已经发布的 organic-hang 历史证据。

## 下一次 GPU 阶段

下一次只租两卡，执行 Gate 0。冻结预期为：

- `unused-parameter`：两个 rank 都只观察到 bf16，3/3 正常完成；
- `forced-mixed-gradient`：两个 rank 都观察到 bf16+fp32，3/3 命中精确 assertion；
- 任一 rank 记录缺失、dtype 集不符、超时或遗留进程都判为 Gate 0 失败。

## 2026-09-12 实际结果

- PyTorch `2.13.0+cu130`，CUDA runtime 13.0，双 RTX 4090；
- control：3/3 完成；
- affected：3/3 完成，均未出现目标 assertion；
- 6/6 记录了两个 rank 的 PID 身份，且没有遗留进程；
- 原预期 `reproduced` 验证按设计失败；
- `not-reproduced` 仅作为结果集完整性检查通过，不能替代预注册结论。

因此目前只能得出：**普通两卡 FSDP2、单次 backward、一个 rank 跳过
conditional 参数，不足以复现四卡实验中的 dtype assertion。** 下一轮先验证
dtype 机制和分类器正控制，不直接增加 microbatch 或 GPU 数；此时跑 nightly
没有判别价值。

结果说明见 `GPU_RESULT_2026-09-12.md`，结构化记录位于
`../../results/pytorch-unused-grad-dtype-20260912/`。
