# Stage B v2 preregistration — reviewed environment baseline

Status: contract candidate; no GPU execution is authorized until every
`TBD-before-execution` item is replaced, reviewed, and frozen.

## Purpose

Stage B v2 asks one bounded question on the already known #53859 mechanism:

> Can two mature attach-based stack producers satisfy the same admitted
> queue-wait rule on a stable, deliberately held vLLM EngineCore process while
> the healthy control does not satisfy that fault-only rule?

This is not a new issue, a rerun of the original Stage 1 identity, a benchmark,
or a general native classifier. It creates a new reviewed environment baseline
because the original dependency-pool identity is unavailable.

## Claims permitted by this run

If every gate passes, the run may claim only that:

1. the same-PID `py-spy A -> PyStack B -> py-spy A2` sequence produced bounded
   captures during the reviewed window;
2. A and A2 establish stability at the rule-predicate and attribution level;
3. both producer kinds satisfy the same admitted rule predicates and emit the
   same `blocked_in` value, with coverage reported separately;
4. the healthy control does not satisfy the fault-only rule.

It may not claim exact reproduction of Stage 1 R3, general producer
interchangeability, root-cause completeness, a v0.2 verdict change, or a need
for a C++ probe.

## Immutable environment manifest

Before renting or starting a GPU, record and review all fields below.

| Identity | Required value |
| --- | --- |
| preregistration commit | `TBD-before-execution` |
| base source commit and tree | `TBD-before-execution` |
| fix source commit and tree | `TBD-before-execution` |
| base/fix allowed diff | only the reviewed #53883 mechanism patch |
| Python version and executable digest | `TBD-before-execution` |
| OS image identifier and kernel | `TBD-before-execution` |
| GPU model, compute capability, driver | `TBD-before-execution` |
| CUDA and PyTorch builds | `TBD-before-execution` |
| vLLM input wheel filename and SHA-256 | `TBD-before-execution` |
| complete input wheelhouse manifest | `TBD-before-execution` |
| installer name, version, and exact command | `TBD-before-execution` |
| canonical environment extraction path | `TBD-before-execution` |
| installed distribution/RECORD manifest | `TBD-before-execution` |
| dependency-pool archive SHA-256 | `TBD-before-execution` |
| model repository, immutable revision, file manifest | `TBD-before-execution` |
| `py-spy` executable/version/SHA-256 | `TBD-before-execution` |
| PyStack executable/version/SHA-256 | `TBD-before-execution` |
| ptrace policy and target-scoped authorization | `TBD-before-execution` |

Package names and versions alone do not satisfy this table. Mutable model
branches, unpinned container tags, network installation during a cell, and a
dependency pool that cannot be restored at the canonical path are preflight
failures.

The environment must be archived before the first cell. A clean temporary
environment restored from that archive must regenerate the same build record
before GPU execution is authorized.

## Cells and invariant

| Cell | Target | Injection | Capture sequence | Expected distinction |
| --- | --- | --- | --- | --- |
| C0 | base EngineCore | none | A/B/A2 | target healthy; fault-only queue-wait rule absent |
| F1 | same base identity | existing deterministic publisher pause | A/B/A2 while held | queue-wait predicates present and attribution stable |

The run reuses the existing queue-backpressure mechanism. It does not add a new
fault injector. The target PID and `/proc/<pid>/stat` start ticks must remain
identical before and after every capture and across A/B/A2 within one cell.

## Acquisition bounds

- producer A/A2: `py-spy`, exact invocation frozen before execution;
- producer B: PyStack, exact invocation frozen before execution;
- per-producer timeout: 5 seconds;
- output limit: 1 MiB per capture;
- cleanup allowance: separately bounded and recorded;
- raw output: private excluded directory, mode `0600` where supported;
- public output: closed capture record, normalized attribution, coverage, raw
  digest, and comparison only;
- occupancy: one attach producer at a time;
- sequence: no unrelated command between A, B, and A2;
- hold invariant: the injected held marker must remain asserted through A2.

For F1, `stage_b_pair_adapter.py` occupies the frozen Stage 1 runner's existing
`py-spy dump --pid PID` slot. The adapter runs the full A/B/A2 sequence before
returning the first py-spy output to the unchanged runner, so the runner cannot
release the held marker between producers. The adapter output directory and
runtime identity fields are supplied through its closed required environment.
Its file digest and exact environment values are part of the run manifest.

For C0, the same runner uses the Stage-B-only `--capture-control-stack` opt-in
to invoke that adapter after health and EngineCore identity are bound but before
the control request starts. This is required under Yama scope 1: the attach
producer must be a descendant of the EngineCore-authorized campaign process.
An attach launched independently by a login-shell watcher is not an admissible
substitute, even when it targets the same PID and start ticks.

A timed-out or unavailable producer, identity change, released hold, output
budget exhaustion, or failed cleanup makes the cell `not_scorable`. It is not
retried inside the same result directory.

## Rule admission

Only the existing queue-wait hypothesis may be considered:

```text
publisher -> queue put -> condition wait
```

The exact normalized predicates and version allowlists must be derived from a
private exploratory capture, reviewed, and committed before the scored C0/F1
run. A rule must require at least one observed predicate, remain within the
`stack_snapshot` producer family, and use exact reviewed versions. Unmatched
captures remain `unknown`; nearest-match attribution is forbidden.

The healthy and fault observations are both required. A frame or predicate
seen in C0 cannot be the sole discriminator for F1.

## Execution gates

Run in this order and stop at the first failure:

1. verify the preregistration commit and closed manifest;
2. restore the archived environment at the canonical path;
3. regenerate and byte-compare both build records;
4. verify model revision and every model-file digest;
5. verify producer binaries and ptrace authorization;
6. run C0 once and inspect only the closed result;
7. run F1 once if C0 passes;
8. regenerate both build records after cleanup;
9. publish only reviewed closed evidence.

No model download, server start, or capture is permitted before its preceding
gate passes.

## Stop and continuation rules

- One successful C0/F1 pair is sufficient for review; repetition requires a
  separate reason and directories.
- Stage B v2 failure does not authorize a new fault area or a source-patched
  probe.
- Stage C remains blocked until Stage B validates the acquisition and
  attribution path and the multi-producer join contract is separately reviewed.
- An upstream-facing C++ probe remains blocked until #197232 receives an
  explicit maintainer outcome.

## 中文审阅入口

Stage B v2 是新的、经过评审的环境基线，不是原 Stage 1 R3 的精确复现。GPU 启动前
必须固定 wheelhouse、安装命令、canonical path、完整 dependency-pool 归档、model
immutable revision、producer binary 和 installed RECORD manifest。实验只复用
#53859 已知 queue-backpressure 机制，执行 C0 healthy 与 F1 held fault 两个 cell，
每个 cell 固定使用 `py-spy A -> PyStack B -> py-spy A2`。任一 producer 不可用、
PID identity 改变、hold 提前释放或环境前后 identity 不一致，都以 `not_scorable`
结束。该实验不改变 v0.2 verdict，也不授权 Stage C 或 C++ probe。
