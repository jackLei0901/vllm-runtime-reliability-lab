# From one distributed hang to two upstream findings

Status: **outline with unresolved sections; not ready to publish**

## One-sentence result

A known-answer distributed hang campaign did more than confirm its original
case: strict cross-rank evidence rules exposed a missing Flight Recorder
producer, and minimization separately isolated an FSDP2 gradient-dtype defect
that now has a maintainer-owned fix in progress.

## 1. The operational symptom

The investigation began from an organic distributed-training failure in which
one rank diverged and the job stopped making progress. The initial question was
whether independently captured rank evidence could reconstruct a known
collective mismatch without relying on a failure-time collective.

This was not initially a PyTorch bug-discovery campaign. The known answer was
used as an oracle for evaluating the evidence method.

## 2. What a single Flight Recorder dump could not prove

Rank 0's dump contained a pending collective. Rank 1 produced no dump. A loose
analysis could label rank 1 as a missing collective participant, but that claim
does not follow from one artifact.

Two absence classes must remain separate:

- `producer_missing`: the expected rank produced no diagnostic artifact;
- `member_missing`: the rank produced an artifact, but its artifact has no
  matching entry at the joined process group and collective position.

Treating the first as if it were the second would convert a capture failure into
a distributed-execution claim.

## 3. The evidence chain

The lab combined four kinds of bounded evidence:

1. per-rank application markers describing only allow-listed lifecycle events;
2. external stack samples taken without a failure-time collective;
3. local Flight Recorder artifacts, tracked by expected producer identity;
4. source-pinned shutdown-stage flags, retained without raw user logs.

That join showed:

- rank 1 raised locally and entered `destroy_process_group()`;
- rank 0 later timed out and broadcast a dump request;
- rank 0 wrote a valid dump;
- rank 1 had already stopped the monitor that polls for the request and did not
  produce an artifact before the external bound.

The result became
[`pytorch/pytorch#196968`](https://github.com/pytorch/pytorch/issues/196968).
It identifies a current diagnostic availability gap in selectable legacy
`ProcessGroupNCCL`; it does not identify the exact NCCL blocking call or yet
provide a fix.

Source reading shows a different but related boundary in the default `nccl2`
backend. Its current test contract expects only a rank that detects a failure
itself to write a dump; a rank that observes no backend failure is not asked to
produce one (`test_c10d_nccl2.py:728-731` at the audited PyTorch `main`). This is
not an experimental claim about every `nccl2` failure mode. It shows why an
evidence plane outside one backend is still needed when the first failure is a
rank-local exception that occurs outside NCCL.

## 4. Minimization exposed a second defect

The same campaign reduced the workload from torchtitan and four GPUs to a
standalone FSDP2 reproducer. It showed that the final synchronized backward can
mix gradients accumulated in `reduce_dtype` with fresh gradients or zero
placeholders still represented in `param_dtype`.

Both single-GPU triggers reach the same assertion:

- a parameter first used on the last microbatch;
- a never-used parameter represented by
  `set_reduce_scatter_unused_params(True)`.

In a rank-divergent two-rank job, only one rank may raise while the job as a
whole appears stalled. This became
[`pytorch/pytorch#196996`](https://github.com/pytorch/pytorch/issues/196996).

In a
[dated issue comment](https://github.com/pytorch/pytorch/issues/196996#issuecomment-5671214251),
weifengpy connected the report to
[`pytorch/pytorch#194434`](https://github.com/pytorch/pytorch/pull/194434) and
stated that the reported unused-placeholder case would be tested. The lab will
verify the maintainer build rather than open a competing fix.

## 5. Before and after linked evidence

| Evidence available | Defensible conclusion |
| --- | --- |
| rank 0 dump only | rank 0 has a pending local collective; peer participation is unknown |
| rank 0 dump plus no rank 1 file | rank 1 is a missing producer; still no claim about its collective participation |
| both external stacks | rank 0 and rank 1 are blocked at different CPU execution points |
| stacks plus shutdown-stage flags | rank 1 passed the heartbeat-monitor stop point but did not finish communicator destruction |
| source-pinned minimal reproducer | the gap is independent of FSDP, model code, mixed precision, and gradient accumulation |

The value is not the number of files collected. The value is that the joined
contract rules out a wrong inference and changes the next engineering action.

## 6. Resolution status

| Finding | Discovery | Upstream triage | Fix | Independent verification |
| --- | --- | --- | --- | --- |
| Missing legacy ProcessGroupNCCL teardown dump | complete | labeled `triaged`; no maintainer response as of 2026-09-15 | pending design agreement | pending |
| FSDP2 mixed gradient dtype | complete | labeled `triaged`; maintainer response on 2026-09-14 | maintainer PR in progress | pending maintainer build |

This table must be updated rather than rewritten after outcomes are known. A
triaged issue is not a resolved issue, and a draft fix is not a verified fix.

## 7. What the lab contributed

- a fail-closed distinction between missing producer evidence and missing
  collective membership;
- reproducible, source-pinned narrowing rather than a production-scale trace;
- a one-GPU reproducer for the FSDP2 assertion;
- a two-GPU reproducer for the diagnostic gap;
- explicit non-claims about root cause and peer participation;
- a verification plan for the maintainer-owned fix.

## 8. What remains unresolved

- #196968 needs an agreed lifecycle design and a validated fix or documented
  limitation;
- #194434 must be tested with the exact #196996 triggers and merged before the
  dtype finding is closed;
- no independent external user has yet run the lab workflow;
- the result is PyTorch-native and does not yet establish a vLLM-native hang
  diagnosis.

The investigation also corrected its own protocol and implementation errors:

- Gates 1b and 1c were withdrawn before execution after source review found an
  incorrect CPU wait site, an overly tight wall bound, and a coupled stop rule;
- the original `"NCCL watchdog"` substring also matched the healthy
  `watchdog thread joined` message, so the marker could not prove a timeout in
  Gates 1f or 1g; the retained timeout evidence came from the exact dump-signal
  and decodable-dump observations instead;
- Windows CRLF checkout conversion changed three reused files' working-tree
  SHA-256 values, so the freeze record was corrected to distinguish line-ending
  conversion from a semantic source change.

These corrections remain part of the case rather than being removed from the
publication narrative.

## 9. Transfer to vLLM

The next vLLM case should test a different capability: a process remains alive
while request progress stops. The candidate is
[`vllm-project/vllm#53859`](https://github.com/vllm-project/vllm/issues/53859),
where a full event queue can block EngineCore while health remains green.

The pre-registered product question is whether the lab can detect no progress
and place the blocked EngineCore thread at the queue operation, then show that
the signal disappears with the existing fix. That work is intentionally not
claimed by this case study yet.

## 10. Publication gate

Do not publish this draft as a success story until at least one of these occurs:

1. #194434 is independently verified and merged, closing #196996; or
2. #196968 gains an accepted fix or explicit maintainer resolution.

Before publication:

- replace status language with dated, linked outcomes;
- add the compact before/after validation table;
- link only reviewed, privacy-bounded public artifacts;
- keep failed gates and protocol corrections in the limitations section;
- add a five-minute reproduction path for the capability being claimed.
