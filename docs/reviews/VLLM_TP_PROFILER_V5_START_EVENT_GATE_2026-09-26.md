# TP=2 profiler-v5 start-event capability check (2026-09-26)

## Decision

**No new NCCL acquisition probe is admitted.** In this bounded two-rank PyNccl CUDA-graph experiment, NCCL profiler-v5 callbacks reached a lab-local Inspector debug build *before* the delayed collective completed. The existing Inspector JSON export still showed only completed collectives. This narrows the demonstrated gap to Inspector's retention/export policy, not the profiler interface's ability to acquire start events. It does not establish behavior in a real vLLM serving graph or the semantics of every profiler callback.

This result corrects the earlier hypothesis that graph-mode A and B are indistinguishable to every NCCL-level producer: their `CollStart` callback counts differed in this cell. Their `KernelChStart` counts did not. The earlier [A/B result](VLLM_TP_RAS_INSPECTOR_AB_RESULT_2026-09-26.md) remains a record of stock RAS and Inspector behavior; this note adds a new lab-local callback visibility check rather than rewriting that result.

## Frozen inputs and controls

- Two RTX 4090 GPUs; vLLM source `c8602c79062440074a018c1d5f875a55571eb6881`; PyTorch `2.13.0+cu130`, CUDA 13.0, NCCL 2.29.7.
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

The rank-1 `CollStart` callback was absent in A and present in B during the window, even though rank-1 `KernelChStart` was absent in both. Thus the callback interface exposed a distinction that stock RAS counts and Inspector JSON did not expose in these runs. A callback arrival is **not** proof that a device kernel made progress or a collective completed. The rank-1 `KernelChStart` absence in B is consistent with the queued device delay; it is not a general rule about kernel execution without further source/runtime validation.

A separate healthy one-GPU PyTorch profiler smoke observed one `cudaGraphLaunch` event and a correct changed output. That only confirms that a host-layer producer can report a launch in a simple graph; no synchronized host A/B trace was retained in this gate.

## Boundary and next gate

No claim is made that default vLLM serving captures these particular PyNccl calls, that custom all-reduce behaves the same, or that an operator can obtain these in-flight facts from unmodified Inspector. `CollStart` and `KernelChStart` must stay separate from progress/completion verdicts. The next useful check is one pinned vLLM serving route: identify the actual collective call and graph placement, then compare stock RAS/Inspector with the existing callback interface under a bounded fault. Do not add a new Lab probe unless that case reveals a fact missing at acquisition rather than merely hidden at export.

## 明日工作（2026-09-27）

1. **上午：复核本结果。** 独立检查两个最终 runner 的提交、输出哈希、rank 绑定以及 A/B 的窗口顺序；如有矛盾，先降级结论，不补做无边界试验。
2. **下午：读一条真实 vLLM serving 调用路径。** 在固定版本上确认某个 TP all-reduce 的路由、tensor size 与 CUDA Graph 捕获位置。形成一页 source map；若无法确认，则将 serving 推论保持为 `unverified`。
3. **底层能力建设：** 阅读 profiler-v5 中 `CollStart` 与 `KernelChStart` 的时序语义，并写出两者各能/不能证明什么。今天的 lab-local C++ 插桩作为能力练习保留，不将其包装成 upstream probe PR。
4. **退出条件：** 明日不租卡、不扩展 fault 类别、不新开 issue。若 source map 显示当前工具已足够，准备“无需新探针”的 TP 诊断说明；若只缺导出，则设计 Inspector 侧最小、受限的 in-flight 导出契约，先审查隐私与流控，不直接编码。#197232 和 #55537 仍作为独立上游/C++ 轨道跟进。
