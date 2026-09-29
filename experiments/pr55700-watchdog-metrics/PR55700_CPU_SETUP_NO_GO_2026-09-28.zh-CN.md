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

## 事后解释修正（源码复核，非评分运行）

本轮启动失败**不能归因**。没有在同一安装环境中运行去掉 worker 替换和 watchdog 设置的普通 `vllm serve` 对照组，因此无法区分 `--worker-cls`、watchdog 配置、PR 改动与固定版本 CPU 后端各自的影响。A4 所写的“CPU 装置不兼容”只应理解为停止该装置的名称，而非预热异常的已证实原因。

原计划的 TP=1 EngineCore 单元还存在独立的设计缺陷。冻结的 runner 虽请求 `--distributed-executor-backend uni`，[该版本 CPU 平台会在 `VLLM_ENABLE_V1_MULTIPROCESSING` 取默认值 `1` 时将 `uni` 改为 `mp`](https://github.com/vllm-project/vllm/blob/b274bf04dd4c6d54807a136babce5b5d17dd74be/vllm/platforms/cpu.py#L316-L324)。因此，原装置会挂起独立 worker，而不是 EngineCore 内的 UniProc worker。[EngineCore watchdog 在 `EngineCoreProc` 中启动](https://github.com/vllm-project/vllm/blob/b274bf04dd4c6d54807a136babce5b5d17dd74be/vllm/v1/engine/core.py#L1203-L1205)；简单关闭 V1 多进程也不能保留同一可评分的 EngineCore 进程配置。冻结 runner 中的 `uniproc` 说明以此处修正为准。这是基于源码的装置局限，**不是**观测到的超时或 PR 指标结论。

## 处置

按 A4 的上限停止本候选的 CPU 装置工作；目前没有可用于 #55700 的验证评论。若将来重新设计装置，需要新协议来区分启动失败原因，并证实存在能评分 EngineCore 问题的运行路径；本轮失败不授权继续运行。仅凭源码提出的问题属于普通 PR review，而非 Lab 验证交付。私有归档保留原始日志、五份回执、各版探针副本及安装日志，SHA-256 为 `f7fc4c9f2dceb4d3612f2db42cab410e254d03321ae37b27a36297aa223b783b`。原始日志和主机路径不公开。
