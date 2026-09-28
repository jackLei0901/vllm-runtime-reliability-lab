# vLLM #55700 首个外部候选：模型下载补充规则 A3

状态：2026-09-28；须在任何可评分挂起前提交。本文件为规范文本。[English](PR55700_FIRST_CANDIDATE_ADDENDUM_A3_2026-09-28.md)。PR 固定点、装置代码、watchdog 设置、挂起及评分规则均沿用选择记录、A1 和 A2。

首次 A2 的 TP=1 启动也在挂起前结束为 `unscored / server_not_healthy`：镜像虽然可达，但 Hugging Face Xet 的 CAS 请求返回 HTTP 401。这是模型获取失败，不是 watchdog 证据。此前两份启动失败回执也继续保留。

下一次使用全新工作目录前，设置 `HF_HUB_DISABLE_XET=1`，经镜像预下载 `Qwen/Qwen2.5-0.5B-Instruct`；模型快照 `7ae557604adf67be50417f59c2c2f167def9a775` 的 10 个文件已下载完成。服务继续使用 `HF_ENDPOINT=https://hf-mirror.com`、`HF_HUB_DISABLE_XET=1` 和同一模型缓存；A2 的 CPU 内存利用率保持 0.5。指标、评分及挂起规则不变。若启动或控制组仍失败，下一个单元仍记 `unscored`。
