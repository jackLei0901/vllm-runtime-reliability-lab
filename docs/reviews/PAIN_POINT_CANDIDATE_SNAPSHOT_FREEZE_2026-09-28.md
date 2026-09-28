# Q4 pain-point candidate snapshot: execution freeze

Status: **method frozen before candidate acquisition; no issue bodies read**.
Protocol and guide public commit: `62123d08b0a6bce7dbba75617014b99b776ce879`
on `origin/lab/runtime-model`. Runtime model version: `0.1.0` at that commit.

The executor is `scripts/freeze_pain_point_candidates.py` in this repository.
It calls GitHub REST `GET /search/issues` through `gh api`, with 100 results
per page. The repository is `vllm-project/vllm`. The exact fixed query prefix
is `repo:vllm-project/vllm is:issue is:closed in:title,body`; one of the eight
terms in the protocol is added per query, followed by one created month from
January through August 2026 and `closed:2026-01-01..2026-09-25`. Search terms,
month order, and request strings are retained in the generated record.

For every shard, the record keeps the UTC retrieval time for each page,
`total_count`, `page_count`, `incomplete_results`, sorted issue numbers, and
the SHA-256 of the ASCII decimal numbers with one trailing newline per number.
No title, body, comment, or raw response is retained. Any incomplete page,
count mismatch, duplicate within a shard, PR, or missing date stops acquisition.
Shards near GitHub Search's 1,000-result ceiling are divided by creation day
without changing the search terms. A partial output has `status=incomplete`
and cannot be used for sampling; it can be resumed with the same public
protocol commit. A completed output is immutable to this executor.

After all shards pass, the executor records the de-duplicated numeric union,
its digest, and the fixed pseudorandom order using
`SHA256("dfxlab-coverage-20260926-v1:" + decimal issue number)`.
The generated output is
`data/pain-point-sample/candidate_snapshot_2026-09-28.json`.
Snapshot retrieval is **not atomic**: every shard has its own UTC time, and
later issue edits or state changes will not be silently substituted into it.

No inclusion/exclusion decision, issue-body reading, V-label or M-label is
part of this acquisition step. Those are the next step only after the
snapshot is complete and checked.
