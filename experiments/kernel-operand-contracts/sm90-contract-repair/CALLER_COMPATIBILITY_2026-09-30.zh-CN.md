# 固定 7b054aca 的调用方布局审查——不计为 K5 服务验证

本中文版本为准。直接读取干净 checkout 的固定提交
`7b054aca96cea8be1369d651c3434ad140580b92`，没有从旧 main 外推。

| 检查点 | 固定 pin 的证据 | 含义 |
| --- | --- | --- |
| Activation 分配 | [cutlass.py L276–286](https://github.com/vllm-project/vllm/blob/7b054aca96cea8be1369d651c3434ad140580b92/vllm/model_executor/kernels/linear/scaled_mm/cutlass.py#L276-L286) 设置 column-major、use_ue8m0=False；[QuantFP8 L37–44](https://github.com/vllm-project/vllm/blob/7b054aca96cea8be1369d651c3434ad140580b92/vllm/model_executor/layers/quantization/input_quant_fp8.py#L37-L44) 默认 TMA alignment=False；CUDA group 路径将参数传给 fp8_utils | 二维 x 在 [fp8_utils L614–617](https://github.com/vllm-project/vllm/blob/7b054aca96cea8be1369d651c3434ad140580b92/vllm/model_executor/layers/quantization/utils/fp8_utils.py#L614-L617) 分配 `(K/128,M)` 后 permute，stride 为 `(1,M)`，通过 A guard |
| Weight scale | [cutlass.py L313–327](https://github.com/vllm-project/vllm/blob/7b054aca96cea8be1369d651c3434ad140580b92/vllm/model_executor/kernels/linear/scaled_mm/cutlass.py#L313-L327) 传入 `Bs.T` | 若 Bs 是 packed contiguous `(Nblocks,Kblocks)`，其转置 stride 为 `(1,Kblocks)`，通过检查。转置本身不能证明所有加载权重都是 packed |
| Token padding | Cutlass block 类及 block 父类均未覆盖 get_output_padding；[base.py L282–292](https://github.com/vllm-project/vllm/blob/7b054aca96cea8be1369d651c3434ad140580b92/vllm/model_executor/kernels/linear/base.py#L282-L292) 返回 None；group quantization 走分组路径，不走非分组 padded op | 已查调用方不会创建假设中的 token-padded scale slice |
| 旧 Hopper helper | `git grep -n _padded_cutlass <pin> -- vllm` 无匹配 | 旧 checkout 的 helper 不能充当当前调用方证据；未给这条不存在的路径编造 fixture |

CUDA 的 `process_fp8_weight_block_strategy` 返回 weight_scale 时没有强制 contiguous。
因此本审查支持已检查的正常 packed 分配契约，不覆盖所有 loader、checkpoint、
自定义调用方或后端。M=258、stride `(1,260)` 的 TMA-aligned scale 在多个 Kblock
时仍被拒绝；已查 CUTLASS block quantizer 没有开启此设置。

[test_sm90_contract_r2.py](../../../tests/test_sm90_contract_r2.py) 新增 metadata
算术控制及真实 CPU Torch 分配测试，复刻固定 pin 的二维 activation 分配表达式
和 packed Bs.T 表达式，包含单例形状。检查使用候选实际 guard，不运行 CUDA
quantizer，也不导入 serving 类。本地无 CPU Torch，真实 tensor 测试明确跳过，
不计通过；算术与公开复现检查通过。运行后新增测试没有修改或重评分冻结 GPU 矩阵。

K5 不投入、不计通过：完整构建、服务请求、加载权重及真实 CUTLASS 路径 witness
仍未完成。K6 开销未评分，K7 独立执行未观察到。记为部分完成并停止。
源码推理降低已查布局的误拒绝风险，不等价于观察真实模型在 fix build 完成请求。
