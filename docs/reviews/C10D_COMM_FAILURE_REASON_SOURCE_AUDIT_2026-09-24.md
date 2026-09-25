# c10d communicator failure-reason source audit — 2026-09-24

Decision: **a bounded observability question exists; no new bug or PR is
established.** Do not add a reason enum, alter #197232, or claim that a new
Flight Recorder field would fix the #196968 missing-rank mechanism.

## Review boundary

- Source: local PyTorch checkout at
  `10909dd81964c2497f1f2657c1132dbc1991e582`, the #197232 candidate
  head. This is a source audit, not an executed fault campaign.
- Subject: legacy `ProcessGroupNCCL` and `NCCLComm` reason flow. The separate
  `nccl2` backend, runtime logs, and the contents of optional NCCLX/ROCm
  communicator dumps were not exhaustively audited.
- Duplicate check: a bounded GitHub issue/PR search on 2026-09-24 for
  `commFailureReason`, communicator failure reason, and Flight Recorder found
  adjacent abort/shutdown proposals, but no confirmed exact duplicate. Search
  results are not a complete duplicate audit and cannot authorize a PR.

## Source-derived flow

| Step | Pinned source anchor | Observation | Limit |
| --- | --- | --- | --- |
| Storage | `NCCLUtils.hpp:316,370` | `abort()` accepts `std::optional<std::string>`; `commFailureReason_` is optional free text. | No closed reason vocabulary is present in this wrapper. |
| Assignment | `NCCLUtils.cpp:357-421` | `abort()` sets the field from its argument at 393 and logs the reason or “No abort reason provided” before calling `ncclCommAbort`. | The log is not itself a durable Flight Recorder field; failure while aborting may prevent a later record. |
| Nonempty caller found | `ProcessGroupNCCL.cpp:2918-2942` | A completion-hook exception constructs `errorStr` and calls `abortComms(errorStr)`, which forwards the reason through `abortCommsFromMap()` at 1443-1495. | This is a specific hook-exception path, not proof that all watchdog or timeout aborts carry a reason. |
| Empty callers found | `ProcessGroupNCCL.cpp:1537,2543,3149`; `:910-918` | The inspected PG abort, work abort, and communicator-destruction paths call without a reason. | Adding an enum beside the current argument would leave these paths unclassified unless their producers change too. |
| Read precedence | `ProcessGroupNCCL.cpp:2959-2982`; `NCCLUtils.cpp:731-739` | A present PG reason is returned before `checkForNcclError()`; the detail formatter also prefers a supplied reason. | “Prioritized” applies only when the optional field is present. It does not mean NCCL's code is always hidden. |
| Later error display | `NCCLUtils.cpp:135-148,211-214,415-450` | A later use of an aborted communicator can surface the stored reason; an abort failure can include it in an error detail. | An exception or log may be observed locally but may not survive process death or be captured by a peer. |
| Existing reason-propagation test | `test/distributed/test_c10d_nccl.py:2517-2555`; `ProcessGroupNCCL.cpp:1033-1037,716-753,857-892,910-918` | The test sets blocking wait and disables async error handling. Blocking wait prevents watchdog start. The timeout is recorded as a Work exception and logged; `wait()` then calls `WorkNCCL::abort()`, which calls `ncclComm_->abort()` **without a reason**. The later assertion looks only for “aborted.” | The timeout message may still reach the caller as the original exception/log. The narrow omission is the communicator's stored reason on later reuse, not total loss of the timeout fact. This test cannot demonstrate an FR reason-field benefit. |

## Flight Recorder countercheck

An exact search of the legacy `FlightRecorder.hpp` and
`FlightRecorderDetail.hpp` entry and dump shapes found no
`commFailureReason_` field. The dump does retain collective entries and
per-PG last-enqueued/started/completed sequence numbers
(`FlightRecorderDetail.hpp:512-536,637-672`). It also has an optional
`nccl_comm_state` map. In this checkout, that map is populated by
`getNCCLCommDumpMap()` only under
`(IS_NCCLX || USE_ROCM) && NCCL_COMM_DUMP`
(`ProcessGroupNCCL.cpp:402-430,437-454`). This audit did not establish the
contents of those backend-specific maps. Therefore the supported claim is
**no explicit PG failure-reason field in the inspected legacy FR shape**, not
“no communicator state exists in any Flight Recorder build.”

The underlying rank-1 dump may also be absent, as in #196968. Adding a field
inside that absent artifact would not repair producer availability. These
are separate failure modes and must not be sold as one fix.

## Competing explanations and decision

1. **The reason was never supplied.** This is source-reachable through the
   no-argument callers above; a durable enum cannot be inferred from an empty
   optional value.
2. **The reason was supplied but not retained in the artifact available to an
   external investigator.** The completion-hook path can set it, while the
   inspected legacy FR shape has no explicit corresponding field. Source alone
   does not establish an incident in which this omission changed diagnosis.
3. **The relevant fact is already available through a log, exception, or an
   optional backend dump.** This remains a live counterexample until a fixed
   environment and post-fault retained-artifact test exclude it.

Result: **the examined blocking-wait path has a producer-side gap, not a
storage-only gap**. The timeout is made into a Work exception at
`checkTimeout()` and logged again in `wait()`, but the subsequent communicator
abort receives `std::nullopt`. A closed enum added only beside
`commFailureReason_` would therefore store `unknown` on precisely this
tested path. The test title/comment should not be taken as proof that a
nonempty reason was propagated into the communicator. Whether the existing
exception/log already gives investigators enough information remains an
unanswered diagnostic-utility question, not a demonstrated new bug.

The original proposal “add a closed enum beside the string” is premature: it
would need a producer-side policy for reasonless callers, a versioned artifact
contract, a retained-artifact negative control, and a demonstrated distinction
that existing logs/FR/NCCL RAS cannot make. Producer changes are broader than
adding one storage field. No new upstream item is justified by this audit.

## Review questions / next gate

- For one named failure scenario, which two diagnoses remain indistinguishable
  after retaining the existing exception, log, Flight Recorder, and NCCL RAS
  output?
- For the five inspected call paths, this is answered: four pass no reason;
  only the completion-hook path supplies `errorStr`. For any *uninspected*
  caller or later source revision proposed as a motivating case, does it
  supply a nonempty reason, or would a new field merely formalize `unknown`?
- Can a negative control prove that an existing artifact answers the question
  without an added field? If yes, record **no probe needed**.
- If not, what is the smallest closed reason emitted by the state owner, and
  can it be tested without depending on a dump that may itself be missing?

Hold any new proposal in this seam until #197232 receives substantive review
and a separate duplicate check is repeated against that later source head.
