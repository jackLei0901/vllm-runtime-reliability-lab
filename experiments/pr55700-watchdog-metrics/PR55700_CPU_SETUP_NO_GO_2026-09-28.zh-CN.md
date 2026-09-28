# PR #55700 CPU watchdog 探针：装置 NO-GO

状态：本轮 CPU 装置 `NO-GO`，2026-09-28/29 UTC。[English](PR55700_CPU_SETUP_NO_GO_2026-09-28.md)。这**不是**对 [vLLM #55700](https://github.com/vllm-project/vllm/pull/55700) 或其 watchdog 指标的结论。

## 身份与范围

五份结果回执中，源码和已安装 CPU vLLM 的全部 13 个 PR 修改的运行时文件均与 PR head `b274bf04dd4c6d54807a136babce5b5d17dd74be` 一致。安装版本为 `vllm==0.1.dev1+gb274bf04d.cpu`、`torch==2.13.0+cpu`；容器有 12 个 CPU 核心、AVX512 和 90 GiB 内存上限。模型为 `Qwen/Qwen2.5-0.5B-Instruct`，后来预下载固定在快照 `7ae557604adf67be50417f59c2c2f167def9a775`。探针在 `d647a44` 冻结；仅装置修订 A2、A3、A4 分别在 `b04092c`、`996d8c7`、`bf75ed6` 中于对应尝试前提交。本地评分测试 23 个通过，Ruff 通过。

| TP=1 启动尝试 | 挂起前结果 | 装置发现 |
| --- | --- | --- |
| A1，默认模型端点 | `unscored / server_not_healthy` | 无法连接 Hugging Face 直连端点。 |
| A1，镜像端点 | `unscored / server_not_healthy` | CPUWorker 拒绝默认 0.92 内存预留：要求 82.8 GiB，启动时可用 81.73 GiB。 |
| A2，预留比例 0.5 | `unscored / server_not_healthy` | 经 Xet CAS 下载权重返回 HTTP 401。 |
| A3，禁用 Xet 并预下载模型 | `unscored / server_not_healthy` | 预热到达 CPU Torch 后报 `Torch not compiled with CUDA enabled`。 |
| A4，`--enforce-eager` | `unscored / server_not_healthy` | 虽然接受了 `enforce_eager=True`，预热阶段仍出现同一 Torch/CUDA 错误。 |

五份回执均通过身份核验，但都没有控制组结果、进入挂起标记或可评分的指标窗口。按 [A4](../../docs/reviews/PR55700_FIRST_CANDIDATE_ADDENDUM_A4_2026-09-28.zh-CN.md) 不运行 TP=2。因此，本轮**不能说明**持续挂起时 EngineCore/worker watchdog 计数是否增长，也不能说明非输出 rank 的指标是否导出。

## 处置

按 A4 的上限停止本候选的 CPU 装置工作；目前没有可用于 #55700 的验证评论。重启该问题须另有充分理由，例如已确认可工作的 CPU 服务路径或新的 GPU 原生协议；本轮启动失败本身不授权继续试错。私有归档保留原始日志、五份回执、各版探针副本及安装日志，SHA-256 为 `f7fc4c9f2dceb4d3612f2db42cab410e254d03321ae37b27a36297aa223b783b`。原始日志和主机路径不公开。
