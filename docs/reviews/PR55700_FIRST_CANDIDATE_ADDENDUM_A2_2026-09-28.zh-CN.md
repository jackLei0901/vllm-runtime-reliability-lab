# vLLM #55700 首个外部候选：装置补充规则 A2

状态：2026-09-28；须在任何可评分挂起运行前提交。本文件为规范文本。[English](PR55700_FIRST_CANDIDATE_ADDENDUM_A2_2026-09-28.md)。原选择记录及 A1 的 PR 源码固定点、问题、评分规则和超时设置不变。

两次 TP=1 启动尝试均为 `unscored / server_not_healthy`，都未进入挂起。首次无法连接默认 Hugging Face 端点；第二次镜像可达，但 CPUWorker 拒绝默认的 `gpu_memory_utilization=0.92`：90 GiB 容器启动时可用 81.73 GiB，而该值要求 82.8 GiB。

下一次使用全新工作目录，设置 `HF_ENDPOINT=https://hf-mirror.com`，并给 `vllm serve` 传入 `--gpu-memory-utilization 0.5`。虽然参数名称提到 GPU，此 CPU 后端也用它控制内存预留。这只是使服务启动的装置修正，不改变 watchdog、挂起、指标或评分。保留前两次未评分回执及私有日志；若修订后的控制组失败，不评分挂起。
