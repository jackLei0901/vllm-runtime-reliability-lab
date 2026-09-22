# Failures that never reach the supervisor

Inference services often fail across an ownership boundary. A worker, rank, or
engine loses the ability to do useful work, but the component responsible for
recovery receives a weaker signal—or no signal at all.

Here, *supervisor* means any decision-maker expected to detect failure and act:
a process parent, service manager, health checker, operator, or diagnostic
analyzer. It does not refer only to vLLM's `DPSupervisor` class.

Four existing investigations expose the same pattern:

| Case | What actually failed | What crossed the boundary | Unsafe conclusion |
| --- | --- | --- | --- |
| vLLM [#53859](https://github.com/vllm-project/vllm/issues/53859) / [#53883](https://github.com/vllm-project/vllm/pull/53883) | EngineCore stopped making token progress while publishing into a full event queue | The process stayed alive and `/health` stayed 2xx | The service is healthy |
| PyTorch [#196968](https://github.com/pytorch/pytorch/issues/196968) / [#197232](https://github.com/pytorch/pytorch/pull/197232) | A rank stopped answering dump requests during communicator destruction | Another rank produced a Flight Recorder dump; this rank did not | The missing rank did not participate |
| vLLM [#48966](https://github.com/vllm-project/vllm/issues/48966) / [#52178](https://github.com/vllm-project/vllm/pull/52178) | EngineCore died unexpectedly | The top-level serving process exited with status 0 | Shutdown was successful |
| vLLM DP supervisor Gate 0 | A managed child exited abnormally, or a post-start health probe failed | The parent returned status 0 | The supervised group stopped normally |

These are not four names for the same bug. They share one structural defect:
the state that matters does not survive the boundary in a form the next actor
can safely interpret.

## Case 1: alive and healthy did not mean progressing

The #53859 campaign forced deterministic KV-event queue backpressure in a real,
single-GPU EngineCore. In the affected arm:

- the EngineCore PID and start-time identity remained stable;
- `/health` continued returning 2xx;
- an admitted request stopped producing token progress; and
- an external stack placed the EngineCore thread in
  `ZmqEventPublisher.publish -> Queue.put`.

With the proposed #53883 change, the same request completed under the same
trigger. The liveness gain had a measured cost: four event batches were
dropped.

The lab did not discover this bug. It independently converted a reported
symptom and proposed fix into a bounded base/fix claim, including the trade-off
that a simple "fixed" label would hide.

```text
process alive + health 2xx != useful work is progressing
```

## Case 2: missing evidence did not mean a missing participant

In #196968, Flight Recorder produced a dump for rank 0 but not rank 1. Treating
the file set as a membership list would have removed the most important rank
from the diagnosis.

External process state and both-rank stacks showed that rank 1 was still alive.
Privacy-bounded shutdown-stage flags showed a narrower sequence: rank 1 had
stopped its dump responder, entered communicator destruction, and remained
blocked there before rank 0 requested the dump.

The missing artifact described the diagnostic producer, not the participant:

```text
producer missing != member missing
```

The lab originated #196968 and validated the proposed C++ change in #197232
with three affected-base failures and three proposed-fix passes. The PR remains
an open proposal as of 2026-09-21. This article does not claim an accepted or
merged upstream fix; the full case study remains gated on an explicit upstream
outcome.

## Case 3: an internal fatal failure became exit status 0

For the lifecycle defect reported in #48966, the system-level contract was
simple: an unexpected EngineCore death must make the top-level process exit
non-zero, while intentional SIGTERM must remain a clean exit.

The lab's paired validation observed:

| Path | Injection | Top-level status |
| --- | --- | ---: |
| Baseline | `SIGKILL` EngineCore | 0 |
| Proposed #52178 change | `SIGKILL` EngineCore | 1 |
| Proposed #52178 change | `SIGTERM` API server | 0 |

TP=2 process-level trials also produced status 1 for worker loss and EngineCore
loss, status 0 for intentional SIGTERM, and no orphaned processes.

This bug was independently found; the lab supplied system-level validation. Its
lesson is that a clean parent status is a claim about lifecycle propagation,
not proof that the child execution was healthy.

## Case 4: a supervisor observed failure and still returned success

The DP supervisor Gate 0 matrix exercised the exact upstream
`dp_supervisor.py` lifecycle boundary with four cells:

| Cell | Managed child status | Parent status |
| --- | ---: | ---: |
| Intentional SIGTERM after ready | -15 | 0 |
| Abnormal child exit before ready | 17 | 0 |
| Abnormal child exit after ready | 17 | 0 |
| Health probe failure after ready | -15 during cleanup | 0 |

The real monitoring path observed both abnormal child exits, and the real probe
failure path initiated shutdown, but `run_dp_supervisor()` did not translate
either failure into its returned process status. A service manager supervising
that parent could not distinguish them from intentional shutdown.

This is a lab-confirmed thin-boundary result, not yet an upstream issue or fix.
The experiment deliberately remains separate from #52178 because the
multi-port DP supervisor has a different parent-process lifecycle.

## What the four cases change

Each misleading signal is locally true:

- the process really was alive;
- the health handler really returned 2xx;
- the dump file really was absent; and
- the parent process really returned 0.

The error begins when a local observation is promoted into a stronger system
claim. The remedy is not "collect everything." It is to define the relation
that must hold across the boundary, preserve the identities and time window
needed to test it, and refuse the conclusion when those facts are missing.

The lab therefore separates three layers:

1. **Observation:** process state, health response, progress counter, stack,
   stage flag, child status.
2. **Relationship:** which subject and interval the observations describe, and
   whether they corroborate or contradict one another.
3. **Claim:** a closed verdict that can be recomputed—or `undetermined` when it
   cannot.

Stacks, GPU utilization, and diagnostic artifacts can strengthen attribution.
They must not silently substitute for progress, identity, or demand. Missing
producers remain explicit rather than becoming negative evidence.

## Replay one complete loop without a GPU

The v0.2.0 release packages the #53859/#53883 result as an offline replay:

```bash
git clone --branch v0.2.0 https://github.com/jackLei0901/vllm-runtime-reliability-lab.git
cd vllm-runtime-reliability-lab
python -m pip install .
vllm-dfx replay results/vllm-zmq-backpressure-stage1-r3-20260916
```

The command verifies the archived file set, identities, four-cell base/fix
contract, health and progress observations, stack claim, and event-loss
trade-off. It does not rerun the GPU experiment or claim production readiness.

The useful challenge is not whether the output looks plausible. It is whether
the evidence permits the verdict, whether a contradiction was ignored, or
whether the verifier should have returned `undetermined`.

## Boundaries

- The #53859/#53883 result is single-GPU and single-EngineCore.
- The #52178 results validate lifecycle behavior; they do not make the lab the
  discoverer of that bug.
- DP supervisor Gate 0 exercises a thin lifecycle boundary with exact upstream
  source and real process/server primitives, not full model serving.
- #197232 has no explicit upstream outcome yet, so this article does not present
  it as an upstream success.
- None of these results establishes automatic root-cause classification or
  production remediation.

The common result is narrower and more useful: a process, endpoint, artifact,
or exit code can be truthful while the conclusion drawn from it is wrong. The
reliability problem lives in that gap.
