# Block 4 evidence-to-claim contract

Status: review candidate. This document does not extend the v0.2 verdict set or
publish a new upstream result.

## Purpose

This contract states what the three strongest lab cases permit a reviewer to
infer, what they explicitly do not permit, and which lower-level facts would
remove a remaining ambiguity. It is the admission boundary for Block 5 native
state work: a new producer or probe is useful only if it supplies a named fact
that changes one of these discrimination tables.

Every case uses the same chain:

```text
observation
  -> allowed inference
  -> forbidden inference
  -> required corroboration
  -> contradiction
  -> verdict or insufficient_evidence
```

The three terms below are intentionally different:

- **Primary verdict evidence** selects the case claim.
- **Attribution evidence** explains where the selected failure was observed but
  cannot manufacture the primary verdict.
- **Provenance evidence** binds subject, source, producer, interval, or content
  identity and may invalidate the record without selecting another verdict.

## Cross-case rules

1. Absence is a claim about an expected producer only after that producer and
   its expected output have been identified.
2. Process, endpoint, request, rank, and diagnostic producer are separate
   subjects until an explicit relation binds them to one incident.
3. Wall-clock proximity is not causal order. Same-host monotonic order, logical
   sequence, or an explicit hand-off marker is required.
4. A stack is a bounded observation of one thread at one capture interval. It
   can support attribution; it cannot by itself prove service-wide no-progress.
5. Version or frame-shape mismatch yields `unknown`, never the nearest native
   classification.
6. A proposed fix arm establishes only the behavior observed on that build and
   environment. It is not an upstream outcome.

## Case A — PyTorch #196968 / proposed PR #197232

### Claim under review

```text
participant present + peer dump request exists + dump responder stop requested
+ expected artifact absent
  -> producer_missing
  != participant missing
```

### Subject and interval

| Element | Frozen meaning |
| --- | --- |
| Incident subject | one two-rank ProcessGroupNCCL run |
| Participant subject | rank 1 process, bound by PID/process identity and declared rank |
| Diagnostic producer | rank 1 legacy Flight Recorder peer dump responder |
| Evaluation interval | after rank 1 enters shutdown and before the external wall bound |
| Peer event | rank 0 broadcasts the cross-rank dump request and writes a decodable local dump |

### Evidence-to-claim table

| Observation | Role | Allowed inference | Forbidden inference | Required corroboration | Contradiction | Result if requirement is missing |
| --- | --- | --- | --- | --- | --- | --- |
| rank 1 process identity remains present at capture | primary for participant presence | the original rank process was alive at the observed instant | the rank was making collective progress; the rank participated in the pending collective | stable process identity, declared rank binding, bounded capture time | process exited, PID identity changed, or rank binding is absent | participant presence is `insufficient_evidence` |
| external stack places rank 1 in `destroy_process_group()` | attribution | the sampled thread was executing teardown | the exact NCCL internal call was blocked; teardown began before every peer event | subject identity and a successful bounded capture | stack belongs to another PID, capture is outside the interval, or frame shape is unsupported | native attribution is `unknown` |
| shutdown flags show responder stop requested, communicator destruction started, and destruction did not complete | primary for producer lifecycle | responder termination was requested while teardown remained incomplete | communicator destruction caused the original application failure; `stop()` proves monitor thread exit; every backend/version has the same lifecycle | source/version pin, ordered stage flags, rank identity | responder observed the later request, destruction completed before the request, or flag order is invalid | producer-loss mechanism is `insufficient_evidence` |
| rank 0 broadcasts `exception_dump` and writes a decodable trace | primary for request existence | a peer dump request existed and the requesting producer functioned | rank 1 received the request; rank 1 participated in rank 0's pending collective | rank-0 identity, successful broadcast marker, artifact integrity | no broadcast marker or invalid local dump | cross-rank request path is `insufficient_evidence` |
| no complete rank-1 dump exists in the closed expected file set | primary for artifact absence | the expected artifact is absent inside the bounded result | rank 1 was absent; rank 1 did not participate; no dump was attempted | closed file manifest and expected producer set | a complete rank-1 dump exists or the file set is open-ended | only `artifact_not_observed`; no producer claim |
| unpatched 3/3 lacks rank-1 trace; proposed fix 3/3 produces complete traces | fix validation | released behavior exhibits the gap and the proposed build closes it in its tested environment | same-environment performance comparison; merge acceptance; default `nccl2` behavior | exact build identities, test collection count, decodable per-rank traces | affected base produces the required trace or proposed fix fails the contract | fix effect is `insufficient_evidence` |

### Bounded verdict

The published evidence supports `producer_missing` while the participant
remained present. It does not support `member_missing`, the exact NCCL blocking
function, default-`nccl2` behavior, or an accepted upstream fix.

### Remaining lower-level ambiguity

The stage flags establish lifecycle order but are experiment-specific. A reusable
native source would need to expose a closed transition for responder availability
and communicator destruction. A generic stack alone cannot prove that relation.

## Case B — vLLM #53859 / proposed PR #53883

### Claim under review

```text
stable EngineCore process + repeated health 2xx + admitted demand
+ flat token progress over the evaluation window
  -> alive_health_ok_no_progress
```

Stack and drop-counter evidence then attribute the stall and describe the fix
trade-off; neither is allowed to create the no-progress verdict.

### Subject and interval

| Element | Frozen meaning |
| --- | --- |
| Incident subject | one single-GPU, single-EngineCore serving cell |
| Process subject | EngineCore PID and start-time identity |
| Endpoint subject | API server health endpoint, related by the campaign harness |
| Progress subject | the admitted streaming request and its token progress events |
| Evaluation interval | deterministic consumer pause until release or request completion |

### Evidence-to-claim table

| Observation | Role | Allowed inference | Forbidden inference | Required corroboration | Contradiction | Result if requirement is missing |
| --- | --- | --- | --- | --- | --- | --- |
| EngineCore PID/start identity remains stable | primary | the observed EngineCore process remained alive | the engine was doing useful work | repeated identity checks covering the evaluation interval | process loss or identity change | `process_missing` or `undetermined` by precedence |
| `/health` repeatedly returns 2xx | primary | the endpoint remained responsive during the interval | model serving was healthy; the endpoint and PID are the same process | fresh samples and explicit incident relation | health loss after prior success | `health_lost` by precedence |
| an admitted request remains incomplete | primary demand evidence | useful work was expected during the interval | all service traffic was stalled | request start before the interval and incomplete state throughout it | request completed before the interval or no admitted demand | `undetermined` |
| token progress is flat across the required window | primary progress evidence | no progress was observed in the stated request/service scope | the cause is queue backpressure; every request is stalled | fresh samples or content events, stable producer identity, minimum span | any valid progress event in the interval | `progress_observed` |
| external stack matches publisher -> `Queue.put` -> wait | attribution | the sampled EngineCore thread was blocked in the publisher queue path | stack alone proves service no-progress; the lab wrapper is upstream code | process identity, supported frame rule, capture inside the flat window | capture belongs to another process/window or rule version mismatches | attribution `unknown`; primary verdict unchanged |
| proposed fix completes 64 tokens and records one accepted plus four dropped batches | fix/trade-off attribution | the tested change preserves request liveness under the trigger by dropping batches | reliable event delivery; a production drop rate; downstream loss detection | same trigger, source identity, progress completion, bounded counters | request stalls, source identity differs unexpectedly, or counters violate the closed cell contract | fix/trade-off claim is `insufficient_evidence` |

### Bounded verdict

The evidence supports health-green no-progress for one real EngineCore under a
deterministic queue-backpressure trigger, plus liveness restored with four
dropped event batches on the proposed fix arm. It does not establish DP-wide
impact, an organic production rate, reliable event delivery, or a merged fix.

### Remaining lower-level ambiguity

No lower-level fact is required for the primary verdict. A native observation
could distinguish Python queue wait from a deeper native/runtime wait in a new
incident, but it would improve attribution only. It must not become a second
progress signal.

## Case C — PyTorch #196996

### Claim under review

```text
mixed bf16/fp32 gradient inputs at FSDP2 uniformity check
+ exact local assertion on one GPU
  -> local correctness mechanism can present as a distributed hang
```

### Subject and interval

| Element | Frozen meaning |
| --- | --- |
| Mechanism subject | one FSDP2 parameter group entering `foreach_reduce` |
| Local evaluation point | before reduce-dtype conversion, at the uniform-gradient-dtype check |
| Distributed manifestation | rank 1 local assertion while rank 0 does not complete before the external bound |
| Control | uniform-bf16 unused-parameter path completes |

### Evidence-to-claim table

| Observation | Role | Allowed inference | Forbidden inference | Required corroboration | Contradiction | Result if requirement is missing |
| --- | --- | --- | --- | --- | --- | --- |
| single-GPU trigger records bf16 + fp32 inputs and the exact assertion | primary mechanism evidence | the correctness failure is locally reproducible without a collective mismatch | the original organic run created fp32 gradients by the same path | exact source/build identity, probe position, input dtype set, assertion identity | uniform inputs or a different failure | local mechanism is `insufficient_evidence` |
| unused-parameter control records bf16-only slots and completes 3/3 | negative control | unused parameters alone do not force the assertion in the tested configuration | all unused-parameter configurations are safe | same build/configuration apart from the frozen trigger | control asserts, has mixed inputs, or fails to complete | discrimination gate fails closed |
| forced mixed-gradient positive control asserts 3/3 | classifier control | the probe and classifier detect the named mixed-input condition | the organic source of fp32 is known | exact input-set and assertion match | mixed input completes or another assertion fires | classifier is not validated |
| two-rank divergent case records rank-1 assertion and rank-0 non-completion | manifestation evidence | one local correctness failure can leave a peer waiting so the job appears hung | rank 0 was blocked in a specific collective unless separately observed; the failure requires multiple GPUs | per-rank identity, exact rank-1 assertion, external bound, rank-0 non-completion | both ranks complete or neither exact assertion is observed | distributed manifestation is `insufficient_evidence` |
| both one-GPU minimal triggers reach the same assertion | minimization evidence | the bug class does not require torchtitan, pipeline parallelism, or multiple GPUs | a proposed fix is complete or merged | frozen reproducer and build identity | trigger cannot reproduce on the claimed build | minimization claim is withdrawn |

### Bounded verdict

The evidence supports a single-GPU FSDP2 mixed-gradient-dtype correctness bug
that can create a distributed hang symptom when ranks diverge. It does not
identify the source of fp32 gradients in the original organic workload, prove a
collective defect, or establish that a maintainer fix is merged and complete.

### Remaining lower-level ambiguity

The mechanism is already decided at the local FSDP2 contract boundary. Native
stack capture, NCCL RAS, and hardware telemetry are not required to prove it.
A future fix validation needs the same structured dtype observation and exact
assertion/completion outcomes, not a generic C++ probe.

## Cross-case capability-gap result

| Capability | #196968 | #53859/#53883 | #196996 | Block 4 decision |
| --- | --- | --- | --- | --- |
| stable process/rank identity | required | required | required for multi-rank manifestation | already available |
| progress/demand/health window | not the primary claim | required | not required | already available for serving |
| bounded external stack | attribution | attribution | optional | evaluate mature producers; do not rewrite |
| Python/native/GIL distinction | may narrow teardown attribution | may distinguish queue wait from native wait | not needed | Block 5 experiment only |
| Flight Recorder artifact identity | required | not needed | not needed | already available case evidence |
| responder lifecycle stage | required for reusable producer-loss mechanism | not needed | not needed | only candidate for a minimal probe |
| communicator logical state | corroborating | not needed | only if a competing collective explanation remains | prefer Flight Recorder/NCCL RAS |
| GPU telemetry | not decisive | not decisive | not decisive | no new collector |

Block 4 therefore authorizes no implementation by itself. It names one possible
irreducible fact—shutdown responder lifecycle—and requires Block 5 to show that
mature producers cannot supply it before a C++ probe is considered.

## Source records

- #196968 mechanism:
  [`../experiments/pytorch-unused-grad-dtype/GATE1G_RESULT_2026-09-13.md`](../experiments/pytorch-unused-grad-dtype/GATE1G_RESULT_2026-09-13.md)
  and
  [`../experiments/pytorch-c10d-shutdown-dump/UPSTREAM_FIX_VALIDATION_2026-09-16.md`](../experiments/pytorch-c10d-shutdown-dump/UPSTREAM_FIX_VALIDATION_2026-09-16.md).
- #53859/#53883:
  [`../experiments/vllm-zmq-event-backpressure/STAGE1_R3_RESULT_2026-09-16.md`](../experiments/vllm-zmq-event-backpressure/STAGE1_R3_RESULT_2026-09-16.md).
- #196996 mechanism controls:
  [`../experiments/pytorch-unused-grad-dtype/GATE0_RESULT_2026-09-12.md`](../experiments/pytorch-unused-grad-dtype/GATE0_RESULT_2026-09-12.md)
  and the upstream-minimal validation
  [`../experiments/pytorch-unused-grad-dtype/upstream_dtype_issue/VALIDATION_RESULT_2026-09-14.md`](../experiments/pytorch-unused-grad-dtype/upstream_dtype_issue/VALIDATION_RESULT_2026-09-14.md).

## Review acceptance checklist

- [ ] Every positive claim names its subject and evaluation interval.
- [ ] Every absence claim names the expected producer and closed output set.
- [ ] Attribution evidence cannot select a primary verdict.
- [ ] Each case contains an explicit contradiction and an insufficient-evidence path.
- [ ] #53859/#53883 is described as independent validation, not discovery.
- [ ] #52178 and #49869 are not imported into these discovery claims.
- [ ] #197232 remains a proposed fix until an explicit upstream outcome.
- [ ] No new producer is approved unless it discriminates a row in this contract.
