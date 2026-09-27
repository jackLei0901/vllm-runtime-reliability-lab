# NCCL `pTimer` 独立 graph replay：单次开卡结果

英文正文：[NCCL_PTIMER_STANDALONE_RESULT_2026-09-27.md](NCCL_PTIMER_STANDALONE_RESULT_2026-09-27.md)。本轮执行依据已冻结的[单次开卡门槛](NCCL_PTIMER_STANDALONE_ONE_BOOKING_GATE_2026-09-27.zh-CN.md)，Lab 提交为 `c0bd1e0d5f10421ffe7d6f50de3c895950d492e2`。eager 与 graph 各运行一次，未根据结果调整参数或重跑。

## 结论

双 rank 的 eager 仪器正控通过。最小独立 graph 用例在双 rank 上均得到 **`graph_reuse_not_observed`**：每 rank 有六个新增且 START/STOP 完整配对的 AllReduce occurrence，未发现这些 occurrence 的 START 或 STOP 时钟相等。这是对**该独立小负载**的阴性结果，不能推翻此前 vLLM 强制 PyNccl 服务运行中的观察。按预注册规则，下一步先离线比较两种图的结构、回调覆盖与路由；本轮不能据此提交 NCCL 缺陷、断言原版 Inspector 指标错误、启动 NCCL-core 改造或纳入 Lab 探针。

## 身份与封存

- 硬件：两张 RTX 4090。vLLM `0.1.dev586+gc8602c790.precompiled`，源码修订 `c8602c79062440074a018c1d5f875a5571eb6881`；Torch `2.13.0+cu130`；NCCL `2.29.7`，两个 cell/rank 的已加载库 SHA-256 为 `aa957cdfb91b516eae0d54a28e9ee5db52730d02e0ab45580efc3c19a68327a4`；复用基于 NCCL `v2.29.7-1` 构建的 Inspector。
- Inspector 二进制 SHA-256：`ef734dc33acd60af9edc43b8ec84204c70e3d1fde53294478dc5b08e0d352c3b`，与上一轮一致。
- 上一轮私有归档在远端及独立下载后均为 `17d3c0a687e2d7f555591fbdc54dcc7512421fb1bb76305895420ba0b7849d3e`；64 个归档条目，无绝对或路径穿越条目。
- 本轮私有归档在远端及独立下载后均为 `fb393ef17415ff3d9c663b3d838c98c9760a3c10ac4f91aabeab0a16767dd757`；37 个条目。原始日志、GPU 时钟、communicator 标识、PID 和主机路径不公开。

driver 在运行前核对冻结脚本哈希；两个 cell 使用独立目录和新的双 rank 进程。eager 得分是进入 graph cell 的前置条件。runner 对三次 eager 调用以及两次 graph replay 均做数值检查。私有 before/after 快照和 rank 日志哈希已封存；下列结果由[冻结评分器](../../experiments/vllm-tp-dfx/ptimer_standalone_score.py)生成。

| Cell | Rank 0 | Rank 1 | 结果 |
| --- | --- | --- | --- |
| Eager：三次显式 AllReduce | 3 个新增 occurrence、3 个配对 channel；START/STOP 最大等值类均为 1；排序、零时钟、非正配对违规均为 0。 | 相同。 | `eager_instrument_pass` |
| Graph：每次 replay 三次未分组 AllReduce，两次 replay | 6 个新增 occurrence、6 个配对 channel；START/STOP 最大等值类均为 1；零时钟、非正配对违规均为 0。 | 相同。 | `graph_reuse_not_observed` |

graph 得分**没有**应用 eager 的流内顺序谓词；其中 `eager_order_violations` 为 null，不是 0。这个独立图包含三次直接 AllReduce 与有界设备 sleep，不能视为 Qwen3 TP 服务图的等价替身。

## 旧服务证据的事后审计

独立校验旧归档后，[冻结审计脚本](../../experiments/vllm-tp-dfx/ptimer_posthoc_audit.py)读取了哈希固定的快照及未变化的 rank 日志。以下为**事后计数**，不回填成旧实验的预注册判据。

| 旧窗口 | Rank 0 | Rank 1 |
| --- | --- | --- |
| 健康服务，after − before | START/STOP 各 1,273；按 communicator、channel、function 分组后，最大等值类各为 73；2,546 个新增时钟中有 14 个小于窗口前该组最大值。 | 各 1,273；最大等值类 START 78 / STOP 77；低于历史最大值 14/2,546。 |
| 服务 hold，during − before | START/STOP 各 164；最大等值类 58/58；低于历史最大值 62/328。 | 各 148；最大等值类 26/26；低于历史最大值 14/296。 |

旧 hold 仍为 **`unscored / target_collective_ambiguous`**。这些计数支持特定路由上的回调时钟归属疑点，但未证明 `base=0`、目标 collective 或原版时间/带宽指标错误。独立用例未复现，意味着在提出 NCCL 报告前，还需要更贴近服务图的复现或源码支持的差异解释。

## 下一步与关机状态

先用两份固定归档离线比较服务图和独立图：capture/replay 路径、collective 的数量与顺序、张量尺寸和 channel 数、回调覆盖，以及重复时钟类的 occurrence/function 构成。只有差异能给出可检验预测时，才预注册下一次 GPU 验证；否则保持“未解决”，不补造探针或提交 issue。

两份归档完成下载与二次哈希校验后，已发送 `shutdown -h now`，SSH 随即断开。云平台的电源及计费状态须独立确认，不能由 SSH 断开推断。
