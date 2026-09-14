# Gate 1g review entry

Status: **frozen locally; not executed; no GPU result exists**

## Review question

Does this two-rank, FSDP-free experiment isolate the ProcessGroupNCCL lifecycle
claim established by Gate 1f without inferring rank-1 collective participation
from a rank-0-only Flight Recorder dump?

## Why this gate exists

Gate 1f matched its source-derived shutdown prediction on PyTorch 2.13.0+cu130
and NCCL 2.29.7, but the workload still contained FSDP, mixed precision,
unused-gradient handling and accumulation. Gate 1g removes all four. It is the
standalone reproducer needed before deciding whether the observed missing-rank
dump is suitable for an upstream PyTorch issue.

## Read in this order

1. `GATE1G_PROTOCOL.md` — hypothesis, frozen observations and non-claims;
2. `gate1g_reproducer.py` — the two-rank control and affected paths;
3. `gate1g_campaign.py` — process lifecycle, stack capture, log reduction and
   rank-0-local Flight Recorder summary;
4. `verify_gate1g.py` — independent result contract;
5. `GATE1G_FREEZE.json` and `verify_gate1g_freeze.py` — 11-file execution
   freeze;
6. `tests/test_gate1g_campaign.py` — parser, classifier, evidence-boundary and
   synthetic-verifier tests.

## Frozen experiment

- one control and one affected trial;
- exactly two identical GPUs;
- PyTorch `2.13.0+cu130`, NCCL `2.29.7`;
- stack capture at 20 seconds, process-group timeout at 30 seconds, external
  wall bound at 60 seconds, explicit `Work.wait()` timeout at 180 seconds;
- both ranks first complete a same-size warm-up `all_reduce` and synchronize the
  GPU, creating the communicator and connections before divergence;
- control: both ranks complete the same asynchronous NCCL `all_reduce`;
- affected: rank 0 enqueues and waits; only after an atomic marker exists does
  rank 1 inject a local exception and enter `destroy_process_group()`.

The atomic marker proves user-space ordering only. The result schema never says
that rank 1 entered the collective.

## Evidence boundary to inspect closely

The affected-arm Flight Recorder contract requires one decodable rank-0 dump
containing exactly one non-completed `ALLREDUCE` for global group `[0, 1]`.
The completed warm-up entry is excluded. Zero or multiple pending candidates
fail closed. The derived summary also requires:

```json
"peer_participation_inferred": false
```

The verifier rejects a result that changes this field to true. More importantly,
the candidate builder reads only rank 0's artifact. Missing rank-1 dump data is
never converted into participation or non-participation evidence.

## Local verification completed

- repository lint: PASS;
- full CPU suite: 125 tests OK, 2 platform skips;
- Gate 1g focused tests: 7 OK;
- compileall: PASS;
- diff whitespace check: PASS;
- Gate 1g freeze simulation with repository LF bytes: PASS, 11 files.

The Windows working tree can still show historical CRLF variants for reused
files. The frozen hashes are the repository LF hashes used by a Linux checkout,
consistent with `.gitattributes` and the earlier correction record.

## Questions for the reviewer

1. Does the completed two-rank warm-up establish the communicator precondition
   that Gate 1f had before divergence?
2. Can `async_op=True` returning plus the atomic marker support the narrow claim
   that rank 0 enqueued its local operation before rank 1 injected the failure?
3. Are the control and affected marker matrices strict enough to reject an
   early rank-0 return or an unobserved rank-1 injection?
4. Does the rank-0-local Flight Recorder summary avoid every cross-rank inference
   that would require the missing rank-1 dump?
5. Are the expected stack sites and ProcessGroupNCCL log flags appropriate for
   the pinned PyTorch/NCCL versions?
6. Is one control plus one affected trial sufficient for this narrow boundary
   check after Gate 1e repeated the original pattern three times?

Do not review this file as a result report. No Gate 1g trial has run yet, and no
outcome may be added to the frozen prediction after GPU execution begins.
