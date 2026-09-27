# TP=2 profiler-v5 start-event capability check (2026-09-26)

Follow-up source and callback review: [VLLM_TP_SERVING_AND_CALLBACK_SOURCE_REVIEW_2026-09-26.md](VLLM_TP_SERVING_AND_CALLBACK_SOURCE_REVIEW_2026-09-26.md). It keeps the real-serving route unverified.

## Decision

**No new NCCL acquisition probe is admitted.** In this bounded two-rank PyNccl CUDA-graph experiment, NCCL profiler-v5 callbacks reached a lab-local Inspector debug build *before* the delayed collective completed. The existing Inspector JSON export still showed only completed collectives. This narrows the demonstrated gap to Inspector's retention/export policy, not the profiler interface's ability to acquire start events. It does not establish behavior in a real vLLM serving graph or the semantics of every profiler callback.

The main positive result is the in-window `KernelChStart` asymmetry: rank 0 had a start marker for the pending collective while rank 1 did not, in **both** graph A and B. The existing profiler interface therefore localized the lagging rank in these cells where stock RAS and Inspector JSON did not. It did not distinguish *why* rank 1 lagged. `CollStart` separated A from B at a different layer: NCCL schedules that callback through a captured CUDA host function before the collective kernel, so it is not a device-start witness. The earlier [A/B result](VLLM_TP_RAS_INSPECTOR_AB_RESULT_2026-09-26.md) remains the stock-tool record; this note adds lab-local callback visibility.

## Frozen inputs and controls

- Two RTX 4090 GPUs; vLLM source `c8602c79062440074a018c1d5f875a5571eb6881`; PyTorch `2.13.0+cu130`, CUDA 13.0, NCCL 2.29.7.
- Inspector source `v2.29.7-1` at `b91894bd5b190c874d98a017f93f5daa515b65d0`. The [small lab-local patch](../../experiments/vllm-tp-dfx/inspector-start-trace-v2.29.7-1.patch) logs only `CollStart` and `KernelChStart` callback arrival through NCCL's existing debug channel. Patch SHA-256: `dad43473ed3894c7d8cf5b5de6866de68d665e0a570056e312b00fa49ed7af89`; rebuilt plugin SHA-256: `e49215462fb80f58d27c39155a5fc8008178e1b11a579c3ee95d259a1b81a315`. The patch was committed before the capability runs (`0c8facd`). It is **not** an admitted vLLM/c10d probe.
- Healthy 20-replay graph control: `result=pass`, changing-input checks passed, and the debug logs contained 42 `CollStart` and 42 `KernelChStart` markers across the two ranks (two initial callbacks per rank plus 20 replay callbacks per rank). Output SHA-256: `0330b7218850ed3fb39bd7ec5ee21da4be99df5cf9cce7d9efa38bea75bf0878`.
- A runner: [ras_peer_hold.py](../../experiments/vllm-tp-dfx/ras_peer_hold.py) at `c10762f`, SHA-256 `e10363c71ce52c8c87aaf21aadfd28ed04a79c1824da67c63cee0a34fbd5a5d6`. Rank 1 waits three seconds **before** replay; rank 0 launches replay and waits for completion.
- B runner: [ras_graph_all_issued_delay.py](../../experiments/vllm-tp-dfx/ras_graph_all_issued_delay.py) at `c10762f`, SHA-256 `7757e1cb79029586935d66edf3d962351abe796f1838b493b825fdb897149046`. Both ranks issue replay; rank 1 has a bounded `torch.cuda._sleep(5_000_000_000)` ahead of its collective on the same stream. Its completion event was pending both before and after the in-window RAS snapshot. The output assertion passed after release.
- Both runners bind each debug log to the live rank PID through `all_gather_object` inside the same run. Unmatched or duplicate log files produce `unscored`. Raw logs, RAS JSON and Inspector JSON stayed in a private mode-0700 directory; a mode-0600 private archive has SHA-256 `72626b1312513451cf26e9596ae9c5da7a1f140051c8bd5816a402a0d3b8a8ec`. The archive contains host/PID-bearing data and is not for publication.

## In-window observations

Each pair below is `CollStart / KernelChStart` callback count. The in-window snapshot occurs before rank 1's A replay and while B's rank-1 completion event remains pending.

| Cell | Before, ranks 0 / 1 | During, ranks 0 / 1 | After, ranks 0 / 1 | Stock Inspector JSON during |
| --- | --- | --- | --- | --- |
| A: peer before replay | `2/2`, `2/2` | `3/3`, `2/2` | `3/3`, `3/3` | one completed all-reduce per rank |
| B: both replayed; peer device delay | `2/2`, `2/2` | `3/3`, `3/2` | `3/3`, `3/3` | one completed all-reduce per rank |

A output SHA-256: `f71e26ce7f9bc6589c79ee3c6e9ba2374fc8f5a3015a3d8c27e83f17453d6b51`. B output SHA-256: `0e7dd96d075adb9229ca3a4ed3b2266f71a54ca4452d15f9a67b8b913afa8990`. Both processes exited successfully and each released collective returned the asserted changing-input result. A and B were each run once at the final pinned runner revisions; this is a capability check, not a reliability or latency estimate.

The `KernelChStart` counts (`3` versus `2`) identify rank 1 as the rank whose NCCL kernel-channel start had not been observed in **either** pending window. Source explains why this is more than an event-name inference: the [device kernel writes](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/device/common.h#L292-L305) the `workStarted` counter and GPU timestamp before work execution; NCCL's [profiler proxy transport](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/transport/profiler.cc#L19-L40) polls that counter before calling `ncclProfilerStartKernelChEvent`, which [delivers the descriptor](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/plugin/profiler.cc#L584-L595). This is a channel-start observation, **not** proof of completion or useful forward progress. It was seen in one run per cell, not characterized for reliability.

Rank 1's `CollStart` was absent in A and present in B while its `KernelChStart` was absent in both. NCCL calls `ncclProfilerStartTaskEvents` from a [captured host-stream callback](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/enqueue.cc#L1293-L1315), enqueued with `cudaLaunchHostFunc` [ahead of the collective kernel](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/src/enqueue.cc#L1517-L1535). Thus B's extra marker reflects execution of that graph's host-side task callback, **not** rank 1 reaching its NCCL kernel. The A/B device-start observation remains identical. The current one-collective graph cannot tell whether `CollStart` for a *later* collective would appear before work between collectives finishes; that requires a separate two-collective graph check.

A separate healthy one-GPU PyTorch profiler smoke observed one `cudaGraphLaunch` event and a correct changed output. That only confirms that a host-layer producer can report a launch in a simple graph; no synchronized host A/B trace was retained in this gate.

## Boundary and next gate

No claim is made that default vLLM serving captures these particular PyNccl calls, that custom all-reduce behaves the same, or that an operator can obtain these in-flight facts from unmodified Inspector. `CollStart` (host-task execution), `KernelChStart` (device-written start observed by proxy), and completion must stay separate from progress verdicts. The next useful serving check binds one actual collective call to its backend and graph placement. A later bounded two-collective graph check can test whether the second `CollStart` appears before intervening device work completes. Do not add a new Lab acquisition probe when the missing operator-facing fact is already acquired by the profiler interface and merely hidden by Inspector's export policy.

## 明日工作（2026-09-27）

1. **上午：复核本结果。** 独立检查两个最终 runner 的提交、输出哈希、rank 绑定以及 A/B 的窗口顺序；如有矛盾，先降级结论，不补做无边界试验。
2. **下午：读一条真实 vLLM serving 调用路径。** 在固定版本上确认某个 TP all-reduce 的路由、tensor size 与 CUDA Graph 捕获位置。形成一页 source map；若无法确认，则将 serving 推论保持为 `unverified`。
3. **底层能力建设：** 阅读 profiler-v5 中 `CollStart` 与 `KernelChStart` 的时序语义，并写出两者各能/不能证明什么。今天的 lab-local C++ 插桩作为能力练习保留，不将其包装成 upstream probe PR。
4. **退出条件：** 明日不租卡、不扩展 fault 类别、不新开 issue。若 source map 显示当前工具已足够，准备“无需新探针”的 TP 诊断说明；若只缺导出，则设计 Inspector 侧最小、受限的 in-flight 导出契约，先审查隐私与流控，不直接编码。#197232 和 #55537 仍作为独立上游/C++ 轨道跟进。
