# DFX claim-delta ledger — existing cases only, 2026-09-25

Status: **review draft, not a benchmark or market-coverage claim**. This
ledger asks the same question for each already investigated incident: what
could the retained baseline evidence conclude, and which *additional*
claim did the Lab's bounded experiment justify? Tool capability, tool
deployment, and retained data are different. “Not scored” must never be
rewritten as “the tool missed the fault.” No new issue was mined for this
ledger.

Source baseline: Lab `1b28aff`. The named PyTorch source-audit boundary is
`10909dd`; these are pinning references, not claims about today's upstream
PR state.

| Case / relationship | Baseline evidence that was actually retained or audited | Lab claim added and its discriminator | Still not established |
| --- | --- | --- | --- |
| vLLM #53859 / proposed #53883 — Lab independently validated a reported issue, did not discover it | `/health` was 2xx in the affected cell. Existing vLLM token and request metrics may be capable, but R3 has no preserved fresh `/metrics` series: **not scored**. | An admitted request stopped content-bearing progress while its bound EngineCore stayed alive; a bounded stack implicated publisher queue wait. Four cells distinguished trigger from control and showed proposed-fix progress **and** four dropped event batches. | Metrics inability; all-request/service-wide stall; production drop rate; accepted upstream fix. |
| vLLM #52178 — existing independently found lifecycle bug, Lab system-level validation | EngineCore death and top-level status can be observed separately; a status of 0 alone mislabels unexpected failure as clean exit. | Process-level trials compared unexpected failure with intentional SIGTERM and late-SIGTERM controls, testing whether fatal cause reaches the API-process exit boundary. | Lab discovery of this bug; a general DP-supervisor guarantee; current merge/review status from this document. |
| vLLM DP supervisor Gate 0 — Lab-originated bounded finding | A child status and top-level supervisor status can be observed, but a top-level 0 alone conceals child/probe failure. | Four-cell thin Gate 0 distinguished intentional signal, child crash, and post-ready probe failure; retained child status 17 versus supervisor status 0 in the failing cell. | Full-package Gate 1 reproduction, upstream confirmation, or a proposed fix. |
| PyTorch #196968 / proposed #197232 — cross-stack transfer case, not a demonstrated vLLM occurrence | Flight Recorder produced one rank's dump while another was absent. A missing artifact by itself does not establish absent membership. | External process/stack and per-rank stage evidence showed the missing-dump rank remained alive in teardown. Base/fix validation supports the proposed responder ordering correction; `producer_missing != member_missing`. | A completed Stage C joined claim from historical artifacts; a merged upstream fix; any vLLM-specific rank occurrence. |
| PyTorch #196996 — cross-stack transfer case | The organic symptom was a distributed hang. That symptom alone does not locate the failed layer. | Reduction and a uniform-dtype negative control isolated a local FSDP2 mixed-gradient-dtype assertion on one GPU. | A vLLM correctness category or proof of a collective fault. |

The strongest *currently scored* DFX comparison is deliberately narrow:
for #53859, repeated healthy HTTP responses did not prove useful request
progress. The Lab's value was the admitted-demand and progress rule, bound
process and stack evidence, negative controls, and the fix trade-off. It is
**not** an established superiority claim over all existing vLLM metrics.
For #196968, the distinct value is a cross-subject inference rule: absence
of a diagnostic producer is not absence of the participant. It transfers
as a method, not as an already demonstrated vLLM defect.

#196968 is the existing case that supports a *limited* same-run method
comparison: the controlled runs retained the Flight Recorder result (one
rank's dump, another absent) alongside Lab external stack and stage
evidence. That supports the stronger inference about a missing diagnostic
producer. It does **not** repair the missing identity/manifest inputs for
the separately specified Stage C closed join, and it says nothing about a
vLLM occurrence. None of the admitted vLLM cases currently clears this
same-case bar against retained existing metrics output.

Before making a stronger “existing DFX cannot diagnose X” claim, require
one of: (a) retained outputs from both methods on the *same* controlled
case and version, or (b) an exact source/contract proof that a named fact
cannot be represented, with a negative control for alternative outputs.
The [#53859 same-case ablation](VLLM_53859_DFX_BASELINE_ABLATION_2026-09-24.md)
meets neither bar for metrics-only detection because its metric series was
not retained. The
[V1 integration gate](V1_RECORDER_VERDICT_INTEGRATION_GATE_2026-09-25.md)
describes a possible future recorder trigger, not a capability already
shipped. The admitted category and transfer boundaries are in the
[fault taxonomy](../FAULT_TAXONOMY_V0_2026-09-24.md).

## Review questions

1. Does any row promote request-scoped evidence into service-wide failure?
2. Does any row confuse an unmeasured baseline with a failed baseline?
3. Are the Lab's discovery, independent validation, and cross-stack
   transfer roles kept separate?
4. Does #196968's same-run Flight Recorder/external-evidence comparison
   support the method claim at its stated limit, without laundering it
   into a completed Stage C join or a vLLM occurrence?

## 中文审阅摘要

这不是“Lab 比所有现成 DFX 更强”的榜单。#53859 只能确认 health-only 的盲区，
不能从未保存的 metrics 序列推断 Prometheus 失效；Lab 的增量是需求与进展
规则、主体绑定、外部栈、负对照和修复代价。#196968 的方法增量是拒绝从缺少
dump 推断 rank 未参与，但它仍是 PyTorch 跨栈案例，不能当作已在 vLLM 发生。
#52178 是独立发现的问题，Lab 提供验证；DP supervisor 目前只到 Gate 0。
