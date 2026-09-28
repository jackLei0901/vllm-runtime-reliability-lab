# First external candidate: vLLM #55700

Status: local selection record, 2026-09-28; no test result or upstream comment. [中文](PR55700_FIRST_CANDIDATE_2026-09-28.zh-CN.md) is authoritative. Source pin: PR head `b274bf04dd4c6d54807a136babce5b5d17dd74be`; recheck the head before testing or posting.

## Consumer and decision

The consumer is the author and reviewers of [vllm-project/vllm#55700](https://github.com/vllm-project/vllm/pull/55700), an active PR authored by someone else. A reviewer [requested a timeout counter for operator alerts](https://github.com/vllm-project/vllm/pull/55700#pullrequestreview-5263026590); the author added counters. The decision is whether the counter reports a **continuing, unrecovered** timeout while the affected step is blocked. This is narrower than whether it reports a timeout after recovery.

## Frozen incremental comparison

| Question | Before testing |
| --- | --- |
| Ordinary reproduction | A missed feed produces a stack dump. It does not establish when `/metrics` changes. |
| Lab addition | Compare the timeout counter during the hold, after release, and after a terminal RPC timeout; establish which event crosses the worker/EngineCore/frontend boundaries. |
| Source inference | The watchdog records a timeout in its background thread. On the EngineCore path, [`update_from_output()` calls `Scheduler.make_stats()`](https://github.com/wenjinhust/vllm/blob/b274bf04d/vllm/v1/core/sched/scheduler.py#L2343-L2355), which [takes its count](https://github.com/wenjinhust/vllm/blob/b274bf04d/vllm/v1/core/sched/scheduler.py#L2874-L2876). On the multiprocess worker path, [`enqueue_output()` attaches the count to a model output](https://github.com/wenjinhust/vllm/blob/b274bf04d/vllm/v1/executor/multiproc_executor.py#L993-L1013), after `get_output()` returns. [Prometheus increments from `SchedulerStats.watchdog_stats`](https://github.com/wenjinhust/vllm/blob/b274bf04d/vllm/v1/metrics/loggers.py#L1110-L1115). A step that never returns may not publish either count. This is untested, not an observed defect. |
| Refuting observation | The counter rises through `/metrics` while the step remains held and no step output has completed. |

Score EngineCore and worker visibility separately: in single-process TP=1, the EngineCore path is the relevant one; in multiprocess TP>1, a worker hold exercises the worker path as well. Secondary, separately scored question: [multiprocess `execute_model` and `sample_tokens` request only `output_rank` replies](https://github.com/wenjinhust/vllm/blob/b274bf04d/vllm/v1/executor/multiproc_executor.py#L341-L362). Test whether another rank's watchdog statistic reaches the frontend. Do not present rank loss as established from the source path alone.

## Bounded next step

Budget: up to four hours of CPU work on the exact PR head. First trace the complete metrics path and write a test that distinguishes (a) held/unrecovered, (b) held then released, and (c) normal work shorter than the watchdog timeout. The fake hold must release the GIL (for example, `threading.Event.wait()`), so the Python watchdog thread can run. Explicitly set a nonempty `dump_dir` and short `timeout`/`check_interval` (with the hold below the RPC deadline): [an empty `dump_dir` disables the watchdog](https://github.com/wenjinhust/vllm/blob/b274bf04d/vllm/utils/watch_dog.py#L223-L249), and its default check interval is 10 seconds. Record a pre-hold counter baseline. A dump or a successful `feed timeout` dump log **during** the hold is required for either scored outcome. For each EngineCore or worker question, `supported` means that witness exists while `/metrics` stays at baseline; `refuted` means that witness exists and the counter rises during the continuing hold; `unscored` includes no witness, setup failure, or an unrepresentative CPU fake. Do not substitute a test of copied logic. No GPU booking or upstream post follows automatically. Before a public draft, re-pin the head, check current contribution and AI-disclosure rules, and let the user review the claim and wording.

This work does not assert that a watchdog is a progress detector, that a post-recovery counter has no value, or that a reviewer required real-time reporting for every hang.
