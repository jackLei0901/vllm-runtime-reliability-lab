# Focused adoption plan

The first adoption cycle aims for ten genuine external stars, two external
replay users, and one external contributor or issue reporter. These are
distribution signals, not the primary technical score. Upstream outcomes,
reproducible claims, and reuse of the evidence method rank above adoption
counts.

## Who this is for

- vLLM maintainers and operators who have seen health-green no-progress;
- PyTorch distributed engineers debugging incomplete cross-rank evidence;
- inference-runtime contributors who need auditable base/fix validation.

The project is not marketed to general observability users, and it does not add
new issue targets to create content.

## Three publication units

1. **Replay launch:** one short post showing the #53859 base/fix output and the
   one-command CPU replay. Ask readers to run it and report friction, not to star
   the repository.
2. **vLLM result:** a focused Slack or discussion post explaining
   `health green != making progress` and the measured liveness/event-loss
   trade-off. Link an immutable release result.
3. **PyTorch case study:** after #197232 has an explicit upstream outcome,
   publish “Why Flight Recorder missed the most important rank.” Share the same
   before/after in an appropriate PyTorch distributed venue, a Chinese technical
   article, and LinkedIn without changing the technical claim.

## Two-week launch sequence

| Time | Action | Evidence of value |
| --- | --- | --- |
| Release day | Tag v0.2, verify links, run the replay on a clean machine | immutable release and clean transcript |
| Days 1–3 | Attach the immutable replay only where an existing upstream thread or focused community discussion directly benefits from the evidence | completed runs, citations, and concrete questions |
| Days 4–7 | Publish the #53859 technical note | referrals to the result/replay, not impressions alone |
| Days 8–14 | Fix onboarding friction reported by users; recognize contributors | external issue, PR, or documented feedback |
| After #197232 outcome | Publish and cross-post the Flight Recorder case study | upstream/community citation or repost |

## Rules for upstream references

- Per-thread communication holds outrank this launch sequence. A thread in a
  deliberate quiet-wait period is not an eligible release venue until its own
  follow-up condition is satisfied.
- Comment on an upstream issue only when the lab adds direct evidence relevant
  to that issue.
- Link a tagged, immutable result or case study rather than the moving `main` branch.
- State whether the lab discovered the bug, independently validated it, or only
  validated a proposed fix.
- Never promote the lab under unrelated issues.
- Never describe #49869 as a lab finding.

## Scorecard

Track the following without collecting personal data:

| Signal | Initial target | Counts when |
| --- | ---: | --- |
| External replay users | 2 | a non-owner reports command, platform, and outcome |
| External contributor/issue reporter | 1 | a useful issue, PR, or evidence-format discussion is opened |
| Upstream references | 2 | an issue or PR directly links a tagged lab result |
| Third-party base/fix use | 1 | someone applies the matrix to a fix outside the original author’s run |
| Genuine external stars | 10 | organic accounts star after encountering the technical material |
| Fork | 1 | an external fork is created for use or adaptation |

Do not buy, trade, automate, or directly solicit stars. A star without a replay,
question, citation, or reuse signal does not establish technical value.

Do not cold-message strangers to manufacture trial users. The distribution
strategy is to place a citable, executable artifact where engineers already
encounter the relevant failure. External runs remain useful validation, but
they are not a prerequisite for the method to have technical value.

## What adoption should reuse

The desired reuse is not necessarily installation of every collector. It can
also be:

- citing the fault taxonomy in an upstream issue;
- applying the health-versus-progress distinction to another incident;
- using the base/fix and contradiction structure to validate a patch;
- challenging a claim with a reproducible counterexample;
- reusing the closed evidence shape or fail-closed verifier behavior.

Collector count, metrics volume, and dashboard coverage are not adoption
goals. Every added field must have a declared decisional or non-decisional
role, and every new producer must discriminate an already named failure mode.

## Four-week stop rule

If all external-reuse signals remain zero at week four, do not broaden outreach
or cold-message more people. The next distribution action must be another
directly relevant upstream evidence attachment, a focused technical talk, or a
new citable case artifact produced by existing work. If none is available,
pause promotion and continue upstream engineering rather than manufacturing
activity.
