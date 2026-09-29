# PR #55700 CPU watchdog probe: apparatus NO-GO

Status: `NO-GO` for this CPU apparatus, 2026-09-28/29 UTC. [中文](PR55700_CPU_SETUP_NO_GO_2026-09-28.zh-CN.md). This is **not** a verdict on [vLLM #55700](https://github.com/vllm-project/vllm/pull/55700) or its watchdog metric.

## Identity and scope

The source and installed CPU vLLM matched PR head `b274bf04dd4c6d54807a136babce5b5d17dd74be` on all 13 PR-changed runtime files in every result receipt. The installation was `vllm==0.1.dev1+gb274bf04d.cpu` with `torch==2.13.0+cpu`; the host offered 12 CPU cores, AVX512 and a 90 GiB container limit. The model was `Qwen/Qwen2.5-0.5B-Instruct`, eventually prefetched at snapshot `7ae557604adf67be50417f59c2c2f167def9a775`. The probe files were frozen in `d647a44`, then setup-only amendments A2, A3 and A4 in `b04092c`, `996d8c7` and `bf75ed6` before each corresponding attempt. The local scoring suite passed 23 tests and Ruff.

| TP=1 startup attempt | Pre-hold result | Setup finding |
| --- | --- | --- |
| A1, default model endpoint | `unscored / server_not_healthy` | Direct Hugging Face endpoint was unreachable. |
| A1, mirror endpoint | `unscored / server_not_healthy` | CPUWorker rejected the default 0.92 memory reservation: 82.8 GiB requested, 81.73 GiB available. |
| A2, reservation 0.5 | `unscored / server_not_healthy` | Model weight download through Xet CAS returned HTTP 401. |
| A3, model prefetched without Xet | `unscored / server_not_healthy` | Warm-up reached CPU Torch but failed with `Torch not compiled with CUDA enabled`. |
| A4, `--enforce-eager` | `unscored / server_not_healthy` | The same Torch/CUDA failure remained during warm-up despite `enforce_eager=True` being accepted. |

All five receipts report identity verified. None has a control result, entered-hold marker or scored metric window. TP=2 was not run, as required by [A4](../../docs/reviews/PR55700_FIRST_CANDIDATE_ADDENDUM_A4_2026-09-28.md). Therefore this campaign says **nothing** about whether the EngineCore or worker watchdog counter rises during a continuing hold, or whether a non-output rank is exported.

## Post-hoc interpretation corrections (source review, not a scored run)

The startup failures do **not** attribute a root cause. No plain `vllm serve` control on the same install, without the worker substitution and watchdog settings, was run. The contribution of `--worker-cls`, watchdog configuration, the PR changes and the pinned CPU backend therefore remains unresolved. A4's phrase "CPU apparatus incompatibility" names the stopped apparatus, not a demonstrated cause of the warm-up exception.

The original TP=1 EngineCore cell also had an independent design flaw. The frozen runner requests `--distributed-executor-backend uni`, but at this pin [the CPU platform changes `uni` to `mp` when `VLLM_ENABLE_V1_MULTIPROCESSING` has its default value of `1`](https://github.com/vllm-project/vllm/blob/b274bf04dd4c6d54807a136babce5b5d17dd74be/vllm/platforms/cpu.py#L297-L305). It therefore would have held a separate worker, not a UniProc worker inside EngineCore. The [EngineCore watchdog is started in `EngineCoreProc`](https://github.com/vllm-project/vllm/blob/b274bf04dd4c6d54807a136babce5b5d17dd74be/vllm/v1/engine/core.py#L997-L1107); simply disabling V1 multiprocessing would not preserve the same scored EngineCore-process setup. The frozen runner's `uniproc` description is superseded by this note. This is a source-based apparatus limitation, **not** an observed timeout or a conclusion about the PR metric.

## Disposition

Stop CPU apparatus work for this candidate under A4's cap. No validation comment is ready for #55700. A future apparatus would need a new protocol that separates startup causes and demonstrates an EngineCore-scoring execution path; no further run is authorized by these failures. A source-based question on the PR remains an ordinary review, not a Lab validation delivery. Private logs, five receipts, the frozen runner copies and setup logs were retained in an archive with SHA-256 `f7fc4c9f2dceb4d3612f2db42cab410e254d03321ae37b27a36297aa223b783b`. Raw logs and host paths are not published.
