# vLLM #55700 first-candidate model-download addendum A3

Status: 2026-09-28; committed before any scored hold. [中文](PR55700_FIRST_CANDIDATE_ADDENDUM_A3_2026-09-28.zh-CN.md) is authoritative. The PR pin, apparatus code, watchdog settings, hold and scoring rules remain as frozen in the selection record, A1 and A2.

The first A2 TP=1 startup also ended `unscored / server_not_healthy`, before any hold: Hugging Face Xet's CAS request returned HTTP 401 through the reachable mirror. This is a model-acquisition failure, not watchdog evidence. The first two startup receipts remain preserved as well.

Before the next fresh-directory run, prefetch `Qwen/Qwen2.5-0.5B-Instruct` through the mirror with `HF_HUB_DISABLE_XET=1`; the download completed for all 10 files at model snapshot `7ae557604adf67be50417f59c2c2f167def9a775`. Keep `HF_ENDPOINT=https://hf-mirror.com`, `HF_HUB_DISABLE_XET=1`, and the same model cache for the server. A2's CPU memory utilization setting remains 0.5. No metric, scoring or hold rule changes. If startup or the control still fails, the next cell remains `unscored`.
