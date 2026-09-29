# PR #55700 first-candidate disposition

Status: CPU apparatus closed as `NO-GO`; upstream answer pending, 2026-09-29. [中文](PR55700_CANDIDATE_DISPOSITION_2026-09-29.zh-CN.md) is authoritative. No claim about the PR's runtime behavior is scored here.

## Result against the frozen comparison

The [selection record](PR55700_FIRST_CANDIDATE_2026-09-28.md) asked whether a watchdog timeout counter reaches `/metrics` during a continuing, unrecovered hold, beyond what an ordinary stack-dump reproduction shows. The [CPU campaign](../../experiments/pr55700-watchdog-metrics/PR55700_CPU_SETUP_NO_GO_2026-09-28.md) never reached a hold or a scored metric window: all five startup attempts were `unscored`. Therefore the incremental Lab claim is **`unscored`** and **no runnable Lab validation artifact was delivered upstream**. Neither support nor refutation can be inferred from the failed apparatus.

One [source-based question](https://github.com/vllm-project/vllm/pull/55700#issuecomment-5887551476) was posted on the PR on 2026-09-29. It is an ordinary Q5 review, not a Lab validation delivery. Evidence grade: source inference at PR head `e67ec698d3c9de90ffc23ffab17b0dfd92ac849e`. Non-author uptake: `not_observed` as of this record; no reply is not a rejection.

## Bounded decision paths

- If the author confirms alerting during an ongoing hang, or disputes the source reading, a **new** A5 GPU protocol may be prepared and committed before booking. Cap it at one session of about three hours. Score only a witnessed timeout, held-window `/metrics` samples, a normal control and post-release export; otherwise report `unscored`. No booking follows automatically from this record.
- If the author confirms post-recovery statistics, optionally suggest one clarifying sentence in the metric description and close the candidate without a GPU run.
- If there is no substantive reply by **2026-10-12**, or the PR merges or closes first, close this candidate with uptake `not_observed`. Do not keep polling or broaden the question.

## Process cost and lesson

The frozen CPU budget was four hours. Work produced the probe, scoring tests, A1–A4 setup amendments and five unscored startup attempts; the upstream delivery was one source-based question. Active hours were **not logged separately** for tooling/setup versus delivery. Commit times and attempt counts are not labor hours, so an actual hour split and the R2 25% fraction cannot be reconstructed honestly. Record R2 as **unmeasured**, not passed; review this process cost at the Oct 26–Nov 1 checkpoint before authorizing A5 or more apparatus work.

The apparatus lesson for R3 is specific: executor predictions must include platform-level configuration overrides. The CPU platform changed a requested `uni` executor to `mp`, invalidating the planned TP=1 EngineCore cell even if startup had succeeded. Future source predictions must check both the general default and the active platform's override before a run.
