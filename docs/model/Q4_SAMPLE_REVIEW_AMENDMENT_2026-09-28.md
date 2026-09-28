# Q4 pain-point sample: review-process amendment A1

Status: awaiting a public commit and push; do not read candidate 11 until both are publicly accessible. [中文](Q4_SAMPLE_REVIEW_AMENDMENT_2026-09-28.zh-CN.md) is authoritative. Recorded on 2026-09-28 before sampling resumes, following the user's decision to review the completed sample rather than blind-label candidates ahead of the AI.

## Preserved and changed

- Preserve the classification rules in §§1–3 of the [v2.0.0 Chinese guide](../PAIN_POINT_M_PART_LABELING_GUIDE_V2_2026Q4.zh-CN.md), `guide_version=2.0.0`, the frozen snapshot and order, six exclusion codes, and `n=40`. The v1 ten-candidate prefix and its failed gate remain unchanged. The public v2.0.0 guide remains historical; this amendment overrides only its §§4–6 human-before-AI procedure and associated reliability gate.
- Resume at candidate 11. The AI labels in frozen order and retains the full reviewed prefix, exclusions, source digests, evidence grades, and labeller provenance in a **private v2 ledger**. Use the existing validator with exact Chinese guide bytes, without `--human-ledger`. Do not change the frame, reshuffle, select favorable cases, or report `k/40` before 40 eligible reports are complete.
- After 40 eligible reports, the user reviews all 40 labels and evidence pointers. Preserve original AI labels and review notes separately. Corrections and recomputed counts are a **post-hoc sensitivity analysis**, not replacements for the original ledger or preregistered proportions. If 40 are not completed, retain the prefix and use `sampling_in_progress`, or `NO-SAMPLE` only on candidate exhaustion.

## Evidence and stage gates

Retrospective review may catch misread reports or rule defects, but it is not independent blind agreement. Record the original v2.0.0 human–AI `model_relation ≥8/10` and median human time ≤15 minutes as `unscored/not_attempted`; neither AI self-repeat, post-hoc agreement, nor AI processing time may substitute. The original execution plan's “8/10 to continue” condition therefore **cannot be reported as passed**. This round may yield only descriptive coverage in the keyword-limited frame, a disagreement ledger, and evidence for the next-PoC decision—not a claim that independent label reliability has been established.

Retain the existing post-40 checks: `guide_issue ≤5/40`, `outside_model + insufficient_information ≤24/40`, decision impact, cumulative effort ≤45 hours, and protection of Q6 kernel work. A failure follows the original revise-or-pause path. Passing these checks without independent reliability only permits consideration of a next PoC; it does not automatically expand `src/dfxlab` or establish a mature taxonomy. User review is an additional quality check, not a retroactive change to the numeric thresholds.

## Resumption condition

Commit and publish this amendment, then verify the remote commit and both language versions are publicly accessible. Create/validate the private v2 ledger and reclassify the already-read v1 prefix before reading candidate 11. `guide_commit` and `guide_sha256` still identify the public v2.0.0 **Chinese guide**; record this amendment's commit separately in private operations notes and the eventual result. The validator checks ledger structure, not retrospective-review quality or whether this amendment is public.
