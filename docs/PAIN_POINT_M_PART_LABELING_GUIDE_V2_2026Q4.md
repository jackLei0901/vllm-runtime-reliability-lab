# Q4 pain-point M-part labelling guide v2

Status: `2.0.0-rc1`, **review candidate, not frozen**. This is a candidate successor to the [frozen v1 guide](PAIN_POINT_M_PART_LABELING_GUIDE_2026Q4.zh-CN.md), not an edit to its labels, frame, review order, exclusions, or `n=40`. [中文](PAIN_POINT_M_PART_LABELING_GUIDE_V2_2026Q4.zh-CN.md). **Do not read candidate 11 until this guide and its validator are publicly frozen.**

## Version boundary

The first ten candidates yielded eight eligible reports and seven `guide_issue` notes, already breaking v1's cap of five among 40. The v1 revise-or-pivot outcome remains. v2 is a separate analysis pass with a separate ledger over the same snapshot and order; it cannot retroactively make v1 pass. Report the two passes side by side, never as one pooled `k/40`.

Use only the issue body and comments permitted by the original protocol. Do not import source knowledge, linked PRs, or the reporter's root-cause inference as label evidence. V1–V3 labels and exclusion codes stay unchanged. The M1–M6 conditions in v1 §3 also stay unchanged: label a boundary only when the report shows its **independent abnormality**. Retain an evidence pointer and log/output versus reporter-narrative grade for every label.

## Two independent facts

- `fault_domain` is the earliest *observed* fault site: `leaf` (kernel, model, allocator, collective, IPC/resource implementation, output, and so on), `lifecycle` (serving process-tree state, signal, budget, or failure propagation), or `unknown`. This is not a source-derived root cause. A leaf failure can still have an independently abnormal M6 propagation boundary.
- `downstream` is required only for `outside_model`: `observed_normal` or `unobserved`. The first requires a pointer to explicit evidence of correct client/process boundary behavior. Silence is `unobserved`, not normal. Correct handling of a request error does not require the service process to exit.

## Exhaustive relation rule

Apply in order, using report evidence alone:

1. An independently abnormal M1–M6 boundary is shown: `mapped`, with nonempty `model_parts`. A normal downstream timeout or 500 does not add another part.
2. Otherwise, a *lifecycle-related* abnormality is shown but M1–M6 cannot express it: `model_gap`, with the missing element and its evidence pointer. An adjacent M-part may be noted without claiming it covers the gap.
3. Otherwise, the earliest fault is identifiable as `leaf`: `outside_model`, empty parts, and a `downstream` value. `unobserved` does not claim normal propagation; it says no independent lifecycle abnormality was shown.
4. Otherwise: `insufficient_information`. A detailed leaf-fault report does not become insufficient merely because it lacks a client status or exit code.

Cross-process resource or handle lifetime *with a lifecycle-related consequence* may be a `model_gap`; a bare IPC implementation bug is not automatically one. For #51258, the IPC counter leak has direct output, whereas wake OOM is reporter narrative, not a retained wake-transition trace. The calibration prediction is provisionally `model_gap` for cross-process resource lifetime, with narrative evidence grade. A reviewer may disagree; preserve that disagreement rather than treating this as a guaranteed positive example.

## Labellers, reliability, and cost

Each v2 entry records `guide_version=2.0.0-rc1`, `fault_domain`, and, for `outside_model`, `downstream`. Record the pass's labeller kind, model identifier (`not_exposed` if the UI does not supply one), SHA-256 of *visible* labelling instructions, and prior exposure to item-level v1 labels. This digest does **not** attest to hidden system prompts or model state. This conversation has seen v1 labels, so its v2 pass is not blind.

The v1 seven-day same-labeller check remains part of v1. Re-running an AI in the current context is not automatically blind. For v2, the preferred check is the user's separate assessment of the first ten **eligible** reports without seeing the AI's item-level labels or the private expectation file. Record the user's prior exposure to aggregate findings, actual review time, and evidence. Candidate threshold: at least 8/10 agreement on the four-way `model_relation`, with M-part-set disagreements also reported. Without a human pass, reliability is `unscored`, not replaced by AI self-repeat.

Only the median of the ten human review times is compared with a candidate ≤15-minute cost threshold. AI seconds are descriptive and not comparable. Complete 40-report coverage is still required before reporting frame-scoped M-part or outside-model fractions. Candidate thresholds are `outside_model + insufficient_information ≤24/40` and `guide_issue ≤5/40`, to be publicly frozen before further sampling. The v1 gate failure is not erased, and hours spent on this direction since September 28 still count against the original total budget.

## Calibration and execution order

Before reading a new candidate, exercise the rule on the seven v1 disputed reports and v1's open-issue and synthetic examples outside the sampling frame. Item-level predicted labels and reasons are in a **private** manifest with SHA-256 `13aab5888b88bc569d1fc74b6226e6446617ade4c483a53d873e612739055008`. These predictions were written *after* the cases were seen: this is a rule-regression check, not a blind holdout, forecasting, or independent agreement. A human reviewer should not read it until their own labels are finished. The public digest commits to its bytes without publishing the manifest.

Review this guide and the validator; verify v1 remains readable and v2 mutations fail; pin the private manifest digest and public guide/code commit; have the user push and verify public access; create a separate v2 ledger and reclassify the already-read prefix under that version; only then read candidate 11 in the original frozen order. Publish v1/v2 prefixes, rule versions, and labeller provenance separately.

## Validator boundary

The v2 ledger uses `schema_version=q4-label-ledger-v2` and binds `guide_commit`, the guide file's actual `guide_sha256`, the original v1 reviewed prefix's `v1_prefix_sha256`, and `labeller`. The prefix digest hashes the entire v1 `entries` array serialized as JSON with sorted keys, `(',', ':')` separators, and UTF-8 encoding; later `relabels` are excluded. Labeller metadata has `kind`, `model_id`, `visible_instructions_sha256`, and `prior_v1_exposure` (`none_declared`, `aggregate_only`, `item_labels_seen`, or `unknown`). Separate `human_reviews` cover the first ten eligible reports, with `human_reviewer` recording exposure to AI labels. The validator rejects changes to inclusion, exclusion, title/body digests, matched terms, or update time within v1's already-read prefix. The private v1 ledger continues to validate under its original command.

V2 validation requires the snapshot, both private ledgers, and the **exact** guide file:

```text
python scripts/validate_pain_point_labels.py --snapshot data/pain-point-sample/candidate_snapshot_2026-09-28.json --ledger PATH_TO_PRIVATE_V2_LEDGER --v1-ledger PATH_TO_PRIVATE_V1_LEDGER --guide docs/PAIN_POINT_M_PART_LABELING_GUIDE_V2_2026Q4.zh-CN.md
```

The validator checks shape, order, version, and evidence pointers. It cannot establish interpretive correctness, whether a human actually avoided the AI's item-level labels, or whether the commit is public. Verify the public URL separately before resuming sampling.
