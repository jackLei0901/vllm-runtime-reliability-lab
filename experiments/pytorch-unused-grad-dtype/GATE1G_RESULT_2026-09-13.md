# Gate 1g result: FSDP-free ProcessGroupNCCL shutdown gap

Status: **PASS for the frozen boundary experiment**

Gate 1g ran from commit
`9f1c72e01616c0a59748e0b91690371da99fb3fd`. The exact source archive had
SHA-256 `a05bbfd50ce8921f1de76002d1bd0a4ac62d2ff9472617389d9a41fb804dacd2`.
Before execution, all seven historical and current freeze verifiers passed with
file counts 15/7/7/9/10/12/11.

## Environment

- two NVIDIA GeForce RTX 4090 GPUs;
- PyTorch `2.13.0+cu130`;
- PyTorch git revision `cf30153c4c131c8164ee7798e5022d810682e2cb`;
- CUDA runtime 13.0;
- NCCL 2.29.7;
- py-spy 0.4.2;
- Linux Yama `ptrace_scope=1`, with each rank authorizing the campaign parent.

## Frozen result pair

| observation | control | affected |
| --- | --- | --- |
| communicator warm-up | both ranks entered and returned | both ranks entered and returned |
| main all-reduce | both ranks enqueued and returned | rank 0 enqueued; no return |
| injected failure marker | none | rank 1 observed the atomic marker, then recorded the declared failure |
| stack capture | not triggered | rank 0 at `work.wait()`; rank 1 at `destroy_process_group()` |
| termination | normal, exit 0 | external 60-second wall bound |
| Flight Recorder | no dump | one decodable rank-0 dump; no rank-1 dump |
| lifecycle | both ranks completed teardown; no orphans | rank 1 entered but did not return from teardown; no orphans after cleanup |

The independent verifier reports:

```text
PASS (Gate 1g: FSDP-free ProcessGroupNCCL shutdown/dump gap reproduced)
```

## Flight Recorder boundary

The rank-0 dump contained exactly one local non-completed collective candidate:

```json
{
  "group_members": [0, 1],
  "observed_rank": 0,
  "operation": "ALL_REDUCE",
  "sequence_number": 2,
  "state": "scheduled"
}
```

The completed warm-up did not enter the pending set. Candidate construction read
only the rank-0 artifact and retained
`"peer_participation_inferred": false`; the missing rank-1 dump was not treated
as evidence for or against rank-1 collective participation.

## Shutdown-stage observation

Rank 0 recorded a successful cross-rank dump-signal broadcast and a successful
local dump. It recorded none of the four shutdown stages. Rank 1 recorded:

1. shutdown started;
2. operations flushed;
3. watchdog joined and communicator destruction started;
4. no `Destroy complete.`;
5. no observation of the later cross-rank dump signal;
6. no successful dump.

This reproduces the Gate 1f missing-rank dump behavior after removing FSDP,
model code, mixed precision, unused-gradient handling, and accumulation. In the
pinned environment, a rank following `destroy_process_group()` can stop its
Flight Recorder responder before a later shutdown step blocks, leaving the rank
that raised the local exception absent from the timeout dump set.

The exact NCCL internal blocking call remains an inference. This experiment does
not establish behavior on other PyTorch/NCCL versions and does not validate a
fix.

## Timing note

The atomic file establishes that rank 0 returned from local enqueue before rank
1 injected the failure. Per-rank `elapsed_seconds` values have independent
process origins and are not compared to establish cross-rank temporal order.

## Retained public evidence

- `control-trial-1.json` SHA-256:
  `59104e6640b4ef4725c39e6e4c6517a5a82dbc972b1f03394ab67c1f05c6b59a`;
- `affected-trial-1.json` SHA-256:
  `7a23d1fa8615e0395e9e2a623dff43f899ad5fa354bffbf5023f76ee97ee29e8`.

The summaries retain bounded markers, hashes, stack locations, per-rank library
flags and lifecycle status. Raw torchrun output, raw py-spy output, and raw
Flight Recorder pickles were not persisted.

## Next step

Before filing upstream, search for an existing PyTorch report covering a Flight
Recorder responder that stops before communicator destruction completes. If no
duplicate exists, extract a standalone script from this reproducer and report
the observed diagnosability gap, not a proposed fix.
