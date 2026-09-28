# vLLM #55700 first-candidate setup addendum A2

Status: 2026-09-28; to be committed before any scored hold. [中文](PR55700_FIRST_CANDIDATE_ADDENDUM_A2_2026-09-28.zh-CN.md) is authoritative. The PR source pin, questions, scoring rules and timeout settings from the selection record and A1 remain unchanged.

Two TP=1 startup attempts produced `unscored / server_not_healthy`; neither entered a hold. The first could not reach the default Hugging Face endpoint. The second used a reachable mirror but CPUWorker rejected the default `gpu_memory_utilization=0.92`: it requested 82.8 GiB while 81.73 GiB was available at startup in a 90 GiB container.

For the next fresh-work-directory run, use `HF_ENDPOINT=https://hf-mirror.com` and pass `--gpu-memory-utilization 0.5` to `vllm serve`. Despite its name, the option controls CPU memory reservation on this backend. This is an apparatus correction to start the server, not a change to the watchdog, hold, metrics or scoring. Keep both earlier unscored receipts and their private logs. If the revised control fails, do not score a hold.
