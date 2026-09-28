# Lab runtime model: execution plan, Sep 28–Nov 1

Status: Lab-internal execution plan, 2026-09-28. [中文](ARCHITECTURE_EXECUTION_2026-09-28.zh-CN.md). This refines the [Q4 plan](../PLAN_2026Q4.zh-CN.md) without changing the released v0.2 verdict contract. From Oct 5–25, sampling temporarily replaces the usual half-day-per-week discovery budget. This is an explicit capacity deviation, not a change to the sampling method.

## Objective and baseline

Use the [M1–M6 runtime model](RUNTIME_MODEL.md) to guide one full cycle of discovery, evidence location, replayable conclusion, and next-work selection. The end-of-October test is not how much model code exists: it is whether the model and index locate evidence, expose gaps, and justify one PoC or an explicit `no_candidate`.

As of Sep 28, model `0.1.0`, the [index](../INDEX.md), experiment status headers, and their test exist. The candidate-ID snapshot contains 823 frozen numbers, but no candidate body has been read and no set of 40 reports has been labelled. `src/dfxlab` does not consume the model. The v0.2 verdicts, external schema, and historical scores remain unchanged. The architecture work is on `lab/runtime-model`, not merged into `main`.

## Schedule

| Period | Engineering and evidence work | Exit gate |
| --- | --- | --- |
| Sep 28–Oct 4 | Build an offline validator and synthetic tests for the frozen sample record. Inputs are the frozen review order and manually entered closed fields. Check exclusion codes, reviewed prefix, stop at 40 eligible reports, M-part relation and evidence, duplicate clusters, and version pointers. Finish the planned G0 CPU identity matrix and verify the #53859 one-page case links. Audit the difference between `lab/runtime-model` and `main` and the dirty worktree. | Valid synthetic records pass; reordered entries, invalid exclusions, missing support, and incorrect stopping fail. Do not read candidate bodies or edit the frozen protocol/guide. The archival tag has been created and the redundant snapshot script removed: confirm its target and remote state. Decide which untracked groups become public, stay private, or are archived; do not mix them into a merge. Merge to `main` only after a publishability review and safe isolation of dirty work. |
| Oct 5–18 | Make sampling the main Lab work temporarily: reserve about 8–10 hours per week to read and label in frozen order, retaining the full reviewed prefix and exclusions. Record private per-report classification time for capacity review, not as a frozen label. Clean-install #53859 replay and ping preregistration use the remaining Lab time; do not displace dated upstream commitments. | Record weekly reviewed/eligible counts and remaining effort; reaching 40 by Oct 18 is not promised. For the preregistered first 10 eligible reports, wait **at least seven days after the 10th receives its first-pass label** before blinded same-labeller relabelling. Keep both original passes and disagreements. Report replay only for commands and versions actually tested. |
| Oct 19–25 | Continue the same frozen-order sample and any relabelling whose wait has elapsed; do not reduce n=40 to meet a date. Close #53859 replay and preregister the normal long-step ping control. If source, CPU, and budget gates pass, run one bounded single-GPU control; otherwise record `NO-GO`. | A full sample conclusion requires 40 eligible reports or candidate exhaustion; only exhaustion is the protocol's `NO-SAMPLE`. If time alone runs out, record `sampling_in_progress`, no k/40 claim, no new keywords, and no favourable selection. Defer or cancel the GPU control if necessary. **Commit the preregistration before the GPU run.** Limited controls do not establish zero false positives. |
| Oct 26–Nov 1 | Summarize completed descriptive M1–M6 and `model_gap` results; build the evidence traceability table, close #53859, and hold the first pain-point review. Choose at most one non-overlapping next PoC, or `no_candidate`. If sampling remains incomplete, review other completed evidence only; do not present it as 40-report coverage. Only if chosen work requires `src/dfxlab` changes, first write a test that fails today and a contract-change note. | Record the choice, counterevidence, budget, and upstream exit. Do not prebuild unchosen abstractions. Version verdict-semantic changes separately; never rescore historical results silently. Every claimed model item links to versioned evidence; blank cells are explicit. |

## Capacity and existing upstream commitments

Normal discovery gets about half a day per week. For the preregistered 40-report sample, reserve roughly 8–10 hours per week from Oct 5–25 (about 24–30 hours total, including exclusions and relabelling). Use the first 10 reviewed items to calibrate actual minutes per report. If the estimate exceeds that budget, retain the reviewed prefix and record `sampling_in_progress`; move nonessential Lab work, not the denominator or rules.

| When | Parallel commitments that must not disappear into the Lab schedule |
| --- | --- |
| Sep 28 | Check #55537's current state and follow the existing plan to contact tlrmchlsmth on Slack; this document does not send that message. |
| Sep 28–Oct 4 | Check #197232 and follow the plan to add tushar00jain as reviewer. Around Oct 1, check #52178 and the DM state before deciding whether the planned comment is due. |
| During October | Make Q5's two substantive lifecycle PR reviews, one in each half of the month. Do not open an opportunistic new upstream topic during peak sampling. |
| By Oct 31 | In the separate kernel track, prepare Q6's blockwise FP8 padded-view negative-control vector for #55537; claim only validation actually run. |

The liveness C1 design question is **not an active upstream post this cycle**. Preserve its evidence and duplicate check; decide after the month-end review whether an existing thread merits a comment. Recheck live thread state before any upstream action, and do not make a maintainer reply a completion gate.

## Preregistered gate for Oct 26–Nov 1

**Baseline and timing:** Freeze this section before reading the first candidate body. There are 823 candidate numbers, but no 40-report labels, blind relabels, or placement times yet. Log effort on this direction from Sep 28 onward (sampling, model-related tools, G0, and technical documentation). List earlier unmeasured work as `unmeasured_prior_work`, not zero hours. Track the existing #53859 PoC, Q5 upstream reviews, and Q6 kernel work separately. Privately time each reviewed item from opening its body to completing exclusion/labels, excluding interruptions. Keep untimed items without estimating their duration.

| Signal | Threshold and interpretation |
| --- | --- |
| Same-labeller reliability | At least **8/10** agreement on the four-way `model_relation` for the preregistered first 10 eligible reports; at most **5** `guide_issue` notes among the 40 eligible reports. Delayed self-agreement is not independent inter-rater agreement. |
| Model scope | At most **24/40** eligible reports labelled `outside_model` or `insufficient_information` in total. Exceeding this limits what the model explains in this fixed reporting frame, not in all vLLM incidents. |
| Placement cost | Median active placement time across the 40 eligible reports is at most **15 minutes**. If more than five lack trustworthy timing, mark this measure `unscored`; do not present the remainder as the full sample. |
| Decision impact | The written next-PoC or `no_candidate` decision cites an M-part count or recurring `model_gap` and explains how it changes an issue-by-issue choice. Merely naming the model does not count. |
| Upstream reuse | If a relevant Q5 lifecycle PR review benefits from a model-derived fact, cite the supported fact and record the use. Do not force a model citation into an unrelated review or make a maintainer reply a gate. |

**Continue:** The complete 40 and delayed blind relabel are scorable, the first four signals pass, newly logged effort on this direction is at most **45 hours** through Nov 1, and this direction has not displaced Q6's #55537 deliverable. Continuing permits consideration of one PoC with a currently failing test; it does not authorize automatic `src/dfxlab` expansion.

**Revise or pivot:** With **6–7/10** agreement, more than five `guide_issue` notes, or a failed scope, cost, or decision-impact signal, do not start new model code. Preserve original labels and versions, then decide whether to narrow the model, revise the next guide version, or retain only an evidence index. Do not rewrite labels mid-sample. An `unscored` cost measure follows this path.

**Pause:** With **≤5/10** agreement, more than 45 newly logged hours, or documented displacement of Q6, pause model expansion and write a short retrospective; retain the model as reference. If sampling or relabelling is incomplete at the review, do not calculate 40-report gates or call a time overrun `NO-SAMPLE`. Authorize a bounded finish only if its remaining estimate fits within 45 hours; otherwise pause. Scope failure alone triggers revision, not a claim that the deliberately liveness-bounded model invalidates the whole Lab.

**Pre-mortem and retrospective:** The likely failures are broad keywords causing many exclusions, labels requiring source knowledge unavailable in issue reports, and sampling displacing #55537. Check reviewed-item rate, `guide_issue`, and protected Q6 time weekly. At month-end, record each threshold, observation, deviation, and continue/pivot/pause decision here rather than adding an OKR hierarchy or dashboard.

**Post-gate procedural review amendment R1 (2026-09-28, before candidate 11):** The [review-process amendment](Q4_SAMPLE_REVIEW_AMENDMENT_2026-09-28.md) records the user's choice to finish AI labelling first and review all 40 afterward. R1 is separate from sampling-protocol amendments A1 and A2. It supersedes the prospective human-first step for v2 without changing the frozen frame or classification rules. The original 8/10 agreement and human-time thresholds remain visible above but will be `unscored/not_attempted`, not passed by retrospective review. Therefore the original full “Continue” gate cannot pass; only descriptive sampling and a bounded next-PoC consideration remain available under the other checks.

## Code to write now, and code to defer

- **Now:** offline sample-record validation with mutation tests, G0 CPU controls, fixes to demonstrated bundle/replay defects, and tests for model-index and evidence-link integrity. None changes model labels or verdict semantics.
- **Defer:** a generic model registry, new collector, automatic health controller, placing M1–M6 directly inside `progress.py`, or a sixth v0.2 verdict. Pause/sleep and EngineCore loop responsiveness are real input gaps, but they do not yet have an admitted new contract and failing test.
- **Boundary:** C++/kernel capability remains a separate Q4 track. Neither #55537 nor c10d work is presented as implementation of the Lab runtime model.

## Minimum technical documentation to add

1. **Producer contract table (by Oct 25):** for M3/M4 signals actually used in this cycle, record owner, unit of observation (request/engine/process), identity binding, timestamp and freshness, missing/conflicting outcome, and current code entry point. Describe facts and unknowns; do not add schema fields. G0 and ping controls determine which rows can claim availability. If sampling consumes the budget, keep only verified rows rather than rushing unproved ones.
2. **Evidence traceability table (before the month-end review):** link each M-ID actually used to a pinned experiment, controls, result, and remaining boundary; mark untested entries `unverified`. This validates the [architecture mapping](ARCHITECTURE_MAPPING_2026-09-27.md) without duplicating its source walkthrough.
3. **Conditional design note (decide at month-end):** if the selected PoC needs a verdict or schema change, specify version, inputs/outputs, failure semantics, compatibility, and a currently failing test first. Otherwise do not write another framework proposal.

The frozen [labelling guide](../PAIN_POINT_M_PART_LABELING_GUIDE_2026Q4.zh-CN.md) remains authoritative during sampling. New technical documentation must not silently redefine labels mid-sample. Review privacy, commit scope, and untracked material separately before publication.
