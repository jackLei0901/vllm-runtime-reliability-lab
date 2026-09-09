# Paired overhead harness

This directory prepares the post-alpha overhead gate without making a GPU
performance claim.

## Contract

- Trials alternate `disabled, enabled` and start with the disabled control.
- Server, workload and recorder commands are argument arrays, not shell strings.
- Every valid workload emits a final JSON line containing one
  `workload_signature` and numeric `metrics`.
- All valid signatures must match and the valid arm counts must be equal.
- Each trial has its own logs and process lifetime.
- Enabled arms stop the recorder gracefully so its private run summary includes
  per-source collection timing.
- Adjacent disabled/enabled trials produce pairwise relative deltas; the report
  retains every pair rather than comparing only two aggregate medians.
- The result JSON is private until commands, paths, metadata and measurements
  have been reviewed.
- A CPU fake-server run validates the harness only. It must not be cited as
  recorder overhead on vLLM.

## CPU validation

Run from the repository root:

```bash
python experiments/overhead/paired_overhead.py \
  --plan experiments/overhead/config.cpu-example.json \
  --output results/self-test-paired-overhead
```

The command should produce four valid alternating trials and a matching
workload signature. On Linux it also samples recorder CPU time and peak RSS from
`/proc`. Windows leaves those fields empty rather than inventing values.
Recorder-initiated stop codes are diagnostic only: validity requires the
recorder to be alive before the harness requests shutdown, to write its private
summary, and to leave no live process group afterward.

## GPU preparation

Copy `config.gpu-template.json`, replace every `REPLACE_*` value, and review the
resolved plan before renting a GPU. Use a clean pinned vLLM worktree: do not mix
PR #52178 or the lab-only OOM injection patch into an overhead run.

`openai_completions_workload.py` uses the OpenAI-compatible streaming
completions endpoint and records TTFT, TPOT, end-to-end latency, request
throughput and output-token throughput. It depends only on the Python standard
library. Its workload signature covers the model name, prompt digest, request
count, concurrency, warmup count, output length and seed.

Before execution, freeze:

- vLLM commit and exact server command;
- model path/revision;
- seed, request set, concurrency and arrival pattern;
- warmup policy and number of valid repetitions;
- metrics and statistical summaries;
- invalid-trial and rerun rules.

The first GPU run should be treated as a smoke test. Do not set an acceptable
overhead threshold after reading its measurements.
