# #52365 candidate close-out

Status: closed on 2026-09-29 without a Lab delivery. [中文](PR52365_SECOND_CANDIDATE_CLOSEOUT_2026-09-29.zh-CN.md) is authoritative. This closes the Lab candidate, not the upstream PR.

The [selection record](PR52365_SECOND_CANDIDATE_2026-09-29.md) asked whether the proposed default-on 60 s GPU-output-event deadline could fail a legitimate, completing step. The [measurement addendum](PR52365_SECOND_CANDIDATE_ADDENDUM_A1_2026-09-29.md) required a measured wait and base/default/disabled comparison. The [offline feasibility review](../../experiments/pr52365-event-timeout/PRE_GPU_REVIEW_2026-09-29.md) found that the proposed 24 GiB RTX 4090 cell had no credible timing margin and could fail startup on memory. An 80 GiB H800 passes only a rough memory screen: its single-step event-wait duration, activation fit, and representativeness remain unmeasured. Neither cell was run. A shutdown command was issued to the opened 4090 instance without running a vLLM cell; the cloud-console billing state was not verified and is not experimental evidence.

## Frozen comparison and outcome

| Item | Close-out |
| --- | --- |
| External decision | Whether to introduce a fatal event-local 60 s default, and whether polling is the right mechanism in [#52365](https://github.com/vllm-project/vllm/pull/52365). |
| Ordinary reproduction | [#52247](https://github.com/vllm-project/vllm/issues/52247) already reports a never-completing GPU event wait. The PR's fake-event tests exercise its helper. |
| Intended Lab addition | A naturally slow but completing serving step, with the same workload succeeding on base and PR timeout-disabled and failing only on PR default. |
| Observed outcome | No serving run, per-event wait, or base/default/disabled comparison. `supported`, `not_reproduced`, and runtime `no_candidate` are all **unscored**, not negative results. |
| External delivery and uptake | No Lab artifact posted to the PR. Delivery: none; uptake: `not_observed`. A later source-backed design comment, if separately approved, is an ordinary Q5 review, not this candidate's measured artifact. |
| Process cost | Tooling and review time were not logged reliably enough for a numeric R2 ratio. This is a process-record gap, not evidence of low cost. The runner, hook, tests and addenda remain unused. |

Source review supports a narrower policy question: the PR activates a previously unread 60 s setting by default; the copy event can include unfinished forward work. It does **not** establish that a legitimate production step exceeds 60 s. The 4090/H800 arithmetic is a booking screen, not a latency measurement or a verdict on the PR.

No GPU booking, further runner work, upstream post, or maintainer ping follows from this close-out. Reopen only upon a measured, naturally slow successful single-event wait or an explicit author/reviewer request for the missing negative control; then freeze a new bounded protocol before booking. Otherwise the source note stays available as Q5 review material. The next Lab candidate is [#54553](PR54553_THIRD_CANDIDATE_2026-09-29.md), subject to its own preflight gate.
