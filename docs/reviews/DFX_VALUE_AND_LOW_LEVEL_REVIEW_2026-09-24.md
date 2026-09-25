# Review entry — DFX value and low-level capability, 2026-09-24

Lab source baseline: `8c6b9d788fe283d6011441c44f2a28d15621376c`.
PyTorch source baseline: `10909dd81964c2497f1f2657c1132dbc1991e582`.
These are review documents, not a new issue, PR, GPU campaign, verifier rule,
or change to the frozen v0.2 bundle/verdict contract.

## Response to the four follow-up findings

| Finding | Resolution |
| --- | --- |
| Blocking-wait reason-propagation test hides an empty communicator reason | Accepted. The audit now follows `blockingWait_` → no watchdog → `checkTimeout()`/`wait()` → reasonless `WorkNCCL::abort()`. The original timeout exception/log still exists; adding storage alone does not solve later communicator attribution. |
| Recorder cannot trigger on the flagship V1 failure | Accepted and elevated to a named gap. The ring exists, but `record` has neither a no-progress trigger nor a generation-token field in its closed observation. An integration needs a separate demand/freshness/identity design gate, not a one-line v0.2 edit. |
| Taxonomy lacks a coverage denominator | Accepted as a measurement question. A proposed preregistered 40-issue sample labels only V1/V2/V3/none/insufficient; it does not admit new categories or establish production prevalence/DFX effectiveness from issue text. No sample has been drawn. |
| Audit communicator lifecycle state next | Completed as a second read-only source audit. `nonBlocking_` is a mode, not a lifecycle flag. `destroy()` holds a recursive mutex, so a concurrent `isAborted()` reader may block; the defensible gap is absent public in-progress stage and clean-destroy/failure-abort conflation after completion. No enum or PR yet. |

The subsequent source review narrows three questions further: four of five
inspected abort paths pass no reason (the completion-hook path supplies one);
the existing shutdown logs can bracket the PG-wide destroy interval when
retained, leaving *per-communicator* attribution as the open question; and
the proposed 40-issue sample explicitly excludes install/build, accuracy-only,
feature-request, and performance-only reports without reliability failure.
At that size its nominal uncertainty is roughly ±15 percentage points near
50%, before frame or labeling bias. No sample has been drawn.

## Read in this order

1. [`C10D_COMM_FAILURE_REASON_SOURCE_AUDIT_2026-09-24.md`](C10D_COMM_FAILURE_REASON_SOURCE_AUDIT_2026-09-24.md)
   — the blocking-wait reason-propagation test stores a timeout as a Work
   exception/log but calls communicator `abort()` without a reason. This is a
   **producer-side gap**, not evidence that an added enum would help.
2. [`C10D_COMM_LIFECYCLE_STATE_SOURCE_AUDIT_2026-09-24.md`](C10D_COMM_LIFECYCLE_STATE_SOURCE_AUDIT_2026-09-24.md)
   — the second, read-only candidate. Clean destroy and failure abort both
   yield `isAborted()==true`; during destroy the getter may block on the held
   mutex. No new state API or PR is justified yet.
3. [`VLLM_53859_DFX_BASELINE_ABLATION_2026-09-24.md`](VLLM_53859_DFX_BASELINE_ABLATION_2026-09-24.md)
   — what health, client progress, process identity, stack, and four-cell
   evidence each permit on the same published case. Metrics-only is honestly
   **not scored** because the R3 public bundle has no metrics time series.
4. [`../FAULT_TAXONOMY_V0_2026-09-24.md`](../FAULT_TAXONOMY_V0_2026-09-24.md)
   — three admitted vLLM boundary classes and two explicitly cross-stack
   transfer cases. It now names the missing no-progress trigger in `record`
   and a proposed, not-yet-run coverage-denominator sampling protocol.
5. [`../../PRODUCT_ROADMAP.zh-CN.md`](../../PRODUCT_ROADMAP.zh-CN.md)
   — parallel Lab C++ DFX track at the state owner and an independent
   vLLM CUDA/kernel career track. Neither is measured by collector count.

## Decisions for reviewers

- Is `commFailureReason_` ever a *necessary* independent fact in a named
  incident after existing logs, errors, FR, and NCCL RAS are retained? The
  audit has not shown this. In particular, most inspected abort callers pass
  no reason; an enum added only beside the optional string cannot fix that.
- Would a communicator lifecycle field change a real post-fault decision?
  Source shows an unrepresented per-communicator transition, but mutex-held
  `destroy()` makes a concurrent `isAborted()` reading impossible to assume.
  PG-wide destroy already has start/end log markers; test whether existing
  log, stack, FR and NCCL RAS evidence can identify the particular
  communicator before proposing another state producer.
- Does the #53859 ablation credit existing vLLM metrics fairly? The answer
  must stay “not scored from R3,” not “metrics could not detect it.” The
  health-only and client-progress rows do have retained case evidence.
- Are V2 and V3 different enough to stay separate? They use distinct parent
  lifecycle boundaries and controls despite the same misleading exit code.
- Should X1 or X2 be imported as vLLM fault categories? Not without a
  vLLM-specific reproducer and negative control.
- Does the C++ plan avoid choosing a vLLM csrc target merely to showcase C++?
  A state owned by PyTorch/NCCL should be investigated there; kernel work is
  separate from Lab DFX.
- Is the V1 recorder gap correctly scoped? It has a 300-sample default ring,
  but no no-progress trigger and no generation-token field in its closed
  observation shape. A future integration needs its own demand/identity/
  freshness contract rather than a single added `if`.
- Would a preregistered sample of closed issues estimate coverage of V1–V3
  without admitting new categories, after the stated exclusions? At n=40 it
  permits only a coarse estimate of that reporting frame. Issue text alone
  cannot establish which existing DFX methods failed in production.

## Checks and claim limits

- `vllm-dfx replay` for the published #53859 R3 directory passed locally
  through the checkout CLI entry. It validated the closed file set, controls,
  health/progress/stack claim, fix-arm progress, and four dropped batches.
  It did not rerun GPU serving or obtain a new metrics baseline.
- The c10d audit used pinned-source inspection and a bounded duplicate search,
  not compiled/ran new C++ or captured a new failure. The second c10d audit
  has the same limit. An issue-search miss is not proof of novelty.
- The taxonomy cites prior campaigns only. DP Gate 1 remains unrun and Stage C
  remains `join_contract_required`; neither can be upgraded by this review.
- Raw logs, stacks, request contents and private artifacts were not copied
  into these documents.

To rerun the published-case check from an installed Lab checkout:

```bash
vllm-dfx replay results/vllm-zmq-backpressure-stage1-r3-20260916
```

To challenge the C++ source table from the pinned PyTorch checkout, search
`commFailureReason_`, `getNcclCommFailureReason`, `abortComms(`, and
`getNCCLCommDumpMap` in `torch/csrc/distributed/c10d/`, then compare each
listed line range. No GPU is required for this source audit.

## 中文审阅入口

本轮审查了两个 PyTorch c10d 底层可观测性候选，用 #53859 已发布证据做同案
DFX 消融，并将已有案例整理成有限的故障分类。第一个候选的更具体结论是：
blocking-wait 测试路径生成并记录了 timeout exception，却没有将 reason 传给
communicator abort；只加 enum 解决不了生产者缺口。第二个候选确有内部阶段
表示不充分，但 `destroy()` 持锁，不能声称 getter 会并发读到“健康”；仍需证明
现有日志、栈、FR、NCCL RAS 不足。R3 未保留指标时间序列，不能声称 Lab 胜过
Prometheus 的检测能力。recorder 有有界历史，却不能自动由 V1 无进展触发；
这需要独立设计 gate，不是本轮代码改动。没有新 issue、GPU 运行或 upstream 修改。

请重点审阅上面的五个判断边界，而不是文档数量。若其中一个问题被反例推翻，
应先收窄结论，再讨论下一步代码。
