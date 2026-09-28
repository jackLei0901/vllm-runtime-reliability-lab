# Lab requirements: external use before expansion

Status: local proposal, 2026-09-28; not approved or published. [中文](LAB_REQUIREMENTS.zh-CN.md) is authoritative; this English text is a translation. This does not change v0.2, historical scores, or Q4 preregistrations.

## Need and baseline

vLLM/PyTorch maintainers and incident responders need to decide what a stalled or failed runtime observation proves, what it does **not** prove, and what to check or test next. Existing logs, metrics, profilers, and Flight Recorder remain the baseline. The Lab is useful only when its bounded evidence or decision rule changes an external diagnostic, test, or fix assessment beyond an ordinary reproduction comment.

There is external **delivery** already: [#52178](https://github.com/vllm-project/vllm/pull/52178) carries base/fix exit evidence, [#36451](https://github.com/vllm-project/vllm/pull/36451#issuecomment-5844585790) received a Lab-backed validation comment, and [#197232](https://github.com/pytorch/pytorch/pull/197232) links the Lab investigation. The first and third are our own PRs; only #36451 is another author's thread. These are upstream outputs, not proof that a non-author independently used a Lab artifact. Track **delivered** (under our control) and **taken up** (under others' control) separately. An absent response is `not_observed`, not a rejection.

## Requirements for the next work item

| ID | Requirement and check |
| --- | --- |
| R1 Consumer | Before *delivery work* starts, name a live thread authored by someone else or a consenting incident owner, their decision, and the artifact they could use. A bounded architecture study may precede this, but cannot become Lab product work until anchored; no consumer means stop at private notes. |
| R2 Process cost | Budget the item. Track time spent on instruments, validators, schemas, and protocol documents separately; if it approaches 25% of the item budget, stop and justify further tooling against a decision it enables. This is a review trigger, not a reason to skip necessary correctness tests. |
| R3 Predictions | Before GPU booking or freezing a success condition, link each decisive prediction to source, a merged PR, or a measured control. Record assumptions that remain unverified. |
| R4 Closure | Write mutually exclusive, collectively exhaustive `pass`, `fail`, and `unscored` outcomes. Test one deliberately failing example before freezing. Apparatus failure is `unscored`, not favourable evidence. |
| R5 Decision edges | Exercise every branch of a new classification rule on fixtures, including missing downstream observations and conflicting producers, before reading real cases. Do not invent a new classifier merely to satisfy this row. |
| R6 Evidence grade | Mark each claim as observed output, reporter narrative, source inference, or hypothesis; state the claim's limit and a plausible counterexample. |
| R7 Contract boundary | List a check's inputs and the one contract it judges. Do not widen `derive_verdict` to encode exit propagation or shutdown budgets. |
| R8 Revisions | Version substantive rule changes before new data, use unique amendment IDs, and preserve old scores. At most one planned instrument redesign per quarter. A correctness repair must name the defect and a fixture that fails before repair; disclose it and never silently rescore. |
| R9 Displacement | Record dated upstream and kernel commitments before scheduling Lab work. Review weekly whether Lab work displaced them; cut or pause Lab scope if it did. |
| R10 Publication | Check the target project's current `AGENTS.md` and disclose AI assistance where required. Review private paths, credentials, unsupported claims, and links; read back the live post. No new issue, fourth vLLM PR while two cannot run CI, extra maintainer ping beyond the dated plan, or request for a `ready` label. |

## Value test and stop rule

Before a decisive run or public draft, freeze a short comparison: the decision the thread faces, what an ordinary reproduction would establish, the additional fact or rule the Lab artifact should supply, and one outcome that would **fail** that claim. Judge the delivered result against this record, not a comparison invented afterward. Deliver the smallest runnable repro, test, validation, or review note with a negative control and an explicit `unknown` case. If the same decision follows from an ordinary reproduction comment, do **not** attribute incremental value to the Lab. No framework layer joins the existing progress, termination, and budget checks.

Name the first qualifying candidate and consumer by **2026-10-12**. If none qualifies, start no further Lab build work and enter maintenance mode by **2026-11-01**. At the **Oct 26–Nov 1** review, record the candidate, frozen comparison, first artifact's delivery status, process cost, and protected upstream/kernel work; do not defer the first assessment to November's end.

By **2026-11-30**, target three *new* such artifacts in active runtime/lifecycle threads **authored by others**; count delivery only when a non-author can inspect or run the material without asking for missing files. This is deliberately harder than supporting our own PRs, and review in this area may be slow. Separately record any non-author citation, rerun, requested follow-up, or use in a fix, along with the exact action. Delivery is the controllable target; uptake is the outcome to assess, never retroactively fabricated. If uptake is absent, distinguish inactive threads from active threads that found the artifacts unnecessary. Repeated delivery without useful uptake calls for maintenance mode, not another framework proposal. A failed ten-hour boundary study stops that study, not the whole Lab.

Quantized-GEMM dispatch belongs to the separate kernel-correctness track. M1–M6 remains an evidence index, not a general issue classifier. The 40-report scope failure concerns its fixed keyword-limited frame and neither proves nor disproves external Lab value.
