# vLLM #55700 首个外部候选：最后一次 CPU 装置修订 A4

状态：2026-09-28；须在任何可评分挂起前提交。本文件为规范文本。[English](PR55700_FIRST_CANDIDATE_ADDENDUM_A4_2026-09-28.md)。PR 固定点、watchdog 设置、挂起与评分规则不变。

使用 A3 缓存的模型快照后，TP=1 启动仍在挂起前结束为 `unscored / server_not_healthy`。模型预热时，CPU Torch 的 Inductor 路径查询 Triton/CUDA 后端，报错 `Torch not compiled with CUDA enabled`。这是 CPU 装置兼容问题，不是 watchdog 观察。保留第四份未评分启动回执。

最后一次有界装置尝试给 CPU 服务命令加入 `--enforce-eager`。这会改变执行模式，因此任何结果只适用于 CPU eager，不证明编译或 CUDA graph 服务模式。若 TP=1 启动或空闲控制组再次失败，本候选按装置 `NO-GO` 停止，不再修订配置，也不运行 TP=2。只有 TP=1 启动和控制组通过，才运行 TP=2。旧的未评分回执绝不重新评分。
