# Gate 1e pre-execution protocol

Status: **pre-execution**. Gate 1d ran one complete control trial and then stopped
on a runner JSON parsing failure during control trial 2. No affected trial ran.
See `GATE1D_EXECUTION_STOP_2026-09-13.md`.

Gate 1e preserves Gate 1d's mechanism, termination, capture, privacy and timing
contracts without reinterpretation. The only runner changes are:

1. every marker JSON object is decoded independently, so two complete rank
   writes that appear on the same launcher-output line cannot be captured as one
   JSON payload;
2. the adjacent-marker form is covered by a local unit test;
3. any control-arm mechanism or termination mismatch stops the campaign after
   retaining that result;
4. an affected-arm termination-only mismatch is retained and does not stop the
   remaining affected repetitions.

## Frozen mechanism matrix

| arm | rank | `foreach_reduce` dtype | reduce | barrier/outcome |
| --- | ---: | --- | --- | --- |
| control | 0/1 | all fp32 | both return | barrier returns; completion |
| affected | 0 | all fp32 | returns | enters barrier; does not return |
| affected | 1 | fp32 + bf16 | does not return | exact uniformity assertion |

The affected termination prediction remains
`wall_bound_rank1_teardown_wait`. A different recognized affected termination is
reported independently and does not invalidate a matching mechanism and capture
result. The control prediction remains `normal_completion` and is fail-fast.

Timing remains stack capture at 20 seconds, process-group timeout at 30 seconds
and external wall bound at 60 seconds. The strict two-rank Flight Recorder join,
two-rank stack requirement, Yama authorization modes, source-hash check, cleanup
contract and raw-evidence exclusions are unchanged from Gate 1d.

## Stop and interpretation rules

- stop and retain the first mechanism mismatch in either arm;
- stop and retain the first control termination mismatch;
- continue after a recognized affected termination-only mismatch;
- never interpret a single-rank dump as collective non-participation;
- mechanism success without capture success is not joined incident evidence;
- publish runner failures, misses and lifecycle failures;
- do not expand to four GPUs to reinterpret this two-GPU contract.

Gate 1e does not establish root cause, general hang diagnosis, vLLM hot-path
coverage, cross-host behavior, production safety or unknown-issue resolution.

## Commands after review

```bash
python experiments/pytorch-unused-grad-dtype/verify_gate1e_freeze.py
python experiments/pytorch-unused-grad-dtype/gate1e_campaign.py \
  --output results/pytorch-unused-grad-dtype-gate1e-YYYYMMDD
python experiments/pytorch-unused-grad-dtype/verify_gate1d.py \
  results/pytorch-unused-grad-dtype-gate1e-YYYYMMDD
```
