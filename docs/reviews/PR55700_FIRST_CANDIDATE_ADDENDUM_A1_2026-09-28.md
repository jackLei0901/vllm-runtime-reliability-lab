# First external candidate vLLM #55700: addendum A1

Status: draft, 2026-09-28; **must be committed before any data is acquired**; the probe has not run. [中文](PR55700_FIRST_CANDIDATE_ADDENDUM_A1_2026-09-28.zh-CN.md) is authoritative. This supplements the committed [selection record](PR55700_FIRST_CANDIDATE_2026-09-28.md) (commit `1f7428e`) without changing it. Source pin remains PR head `b274bf04dd4c6d54807a136babce5b5d17dd74be`.

## Why

A pre-run review found that the original rules could turn an apparatus failure into `supported`: a metric series that never appeared counted as zero, the post-release export check did not affect scoring, and identity was checked for one interpreter while the server could start from another. These rules tighten scoring without changing the questions, the refuting observation, or the candidate.

## Added rules

1. **Export control:** for the EngineCore and worker continuing-hold questions, `supported` also requires the counter to exceed its pre-hold baseline after release; otherwise `unscored` (`export_control_failed`). `refuted` is unchanged: the counter rises after the witness while the hold continues.
2. **Series presence:** after warm-up, the EngineCore series must exist for TP=1, and the EngineCore and output-rank worker series for TP=2. "Series absent" is recorded separately from "series at zero"; an absent baseline is `unscored`.
3. **Hold apparatus:** hold entry, completion of the held request, the release marker, successful post-release requests, and success of the baseline, post-release and **every** scheduled post-witness scrape are all required; for the continuing-hold questions, every post-witness scrape must also contain the required series. Any missing item is `unscored`; the remaining samples are never used instead. A witness counts only during an entered, unreleased hold.
4. **Identity fails closed:** the server is launched by the same interpreter that was verified. All 13 PR-changed runtime files must match the pinned source byte for byte, the source tree must be at the pin, and `git status` must itself succeed and show no tracked changes; a failed command never counts as clean. Otherwise the server is not started and only an `unscored` receipt is written.
5. **Non-output rank:** `supported` when that rank has a witness, its series stays absent throughout, and the output-rank export control holds; a rise is `refuted`; a series that appears without rising is `unscored`.
6. **Idle feed constraint:** from source, an idle EngineCore and an idle shared-memory reader feed about every 5 s. The probe rejects timeouts not above 5 s plus two check intervals, and holds must stay well below the 300 s RPC deadline.

## Prediction before running (source inference, not scored)

With TP=2, EngineCore waits for worker replies inside the shared-memory reader, which feeds the same process-wide watchdog. An EngineCore witness is therefore **not** expected during a worker hold, and that cell should be `unscored` (`no_witness`). An EngineCore witness would be recorded as an observation contrary to this inference.

## Apparatus and limits

The probe and tests are `experiments/pr55700-watchdog-metrics/` and `tests/test_pr55700_watchdog_probe.py`; the apparatus commit is recorded with each result. Each run uses a new, empty work directory; the probe refuses reuse and removes only the hold markers it creates, so earlier evidence is preserved. This addendum books no GPU and produces no upstream post. Before the first run, do a read-only dependency and memory preflight.
