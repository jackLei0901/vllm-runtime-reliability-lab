# vLLM #55700 first-candidate final CPU setup addendum A4

Status: 2026-09-28; to be committed before any scored hold. [中文](PR55700_FIRST_CANDIDATE_ADDENDUM_A4_2026-09-28.zh-CN.md) is authoritative. The PR pin, watchdog settings, hold and scoring rules remain unchanged.

With A3's model snapshot cached, the TP=1 startup still ended `unscored / server_not_healthy` before a hold. During model warm-up, CPU Torch's Inductor path queried a Triton/CUDA backend and failed with `Torch not compiled with CUDA enabled`. This is a CPU apparatus incompatibility, not a watchdog observation. Preserve this fourth unscored startup receipt.

The final bounded setup attempt adds `--enforce-eager` to the CPU server command. This changes execution mode, so any result applies only to CPU eager mode; it cannot establish behavior under compiled or CUDA-graph serving. If TP=1 startup or its idle control fails again, stop this candidate as apparatus `NO-GO`, without another configuration revision or TP=2 run. Run TP=2 only if TP=1 startup and control pass. Earlier unscored receipts are never rescored.
