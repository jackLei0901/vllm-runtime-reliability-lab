# Stage 0 protocol: deterministic publisher backpressure

Status: frozen for execution

Date: 2026-09-15

## Purpose

Establish a CPU-only oracle for vLLM #53859 before running a full EngineCore.
The experiment tests `ZmqEventPublisher.publish()` with its consumer paused by a
declared test subclass.

## Source arms

Both arms start from:

`22258a26bc090bccf5473cf681bbe9bac41bd035`

- **base:** no source change;
- **fix:** fetch #53883 head `1a2b8530`, apply that commit without committing,
  and record the resulting tree using `git write-tree`.

The fix arm must contain only the files changed by #53883. Abort if the patch
does not apply cleanly or if additional files differ.

Suggested preparation in two disposable worktrees:

```bash
git fetch upstream pull/53883/head:refs/remotes/upstream/pr-53883
git worktree add ../vllm-53859-base 22258a26bc090bccf5473cf681bbe9bac41bd035
git worktree add ../vllm-53859-fix 22258a26bc090bccf5473cf681bbe9bac41bd035
cd ../vllm-53859-fix
git cherry-pick --no-commit 1a2b8530
git diff --cached --name-only
git write-tree
```

Expected changed files in the fix arm:

- `vllm/distributed/kv_events.py`
- `tests/distributed/test_events.py`

## Controlled hook

`stage0_backpressure.py` subclasses `ZmqEventPublisher` and overrides only its
publisher thread. The thread announces that it is paused and waits on a release
event before running the production consumer loop. The real constructor and a
unique `inproc://` socket endpoint are used.

When Yama is present, the subject calls `PR_SET_PTRACER` for the declared
observer parent in `DFX_OBSERVER_PID`; an observer launched by that parent can
sample it without broadening the container's ptrace policy. Because this is an
isolated process and it owns the only ZMQ sockets, it explicitly terminates the
shared context after production `shutdown()` so the probe itself exits cleanly.

This is not a production reproduction. It is a deterministic component
experiment that removes consumer timing as a confound.

## Frozen inputs

- queue size: 1;
- observation bound: 1.0 second after the second publisher thread starts;
- base hold for external sampling: 20 seconds by default;
- first and second batches contain no user data;
- one run per arm is sufficient for this deterministic boundary check.

## Frozen predictions

| Observation | Base | #53883 fix |
| --- | --- | --- |
| Consumer is paused before first publish | true | true |
| Queue contains the first batch | true | true |
| Second `publish()` returns inside 1 s | false | true |
| Queue still contains the first batch before release | true | true |
| New batch was dropped | false | true |
| Base call returns after release | true | n/a |

For the base arm, an external stack sample of the printed subject PID must show
the blocked publisher caller in this allow-listed chain:

`threading.Condition.wait` <- `queue.Queue.put` <-
`ZmqEventPublisher.publish`

Line numbers are not part of the contract. Additional stdlib/thread bootstrap
frames are allowed. Sample the subject process, not a parent shell.

## Commands

Run from each vLLM checkout using its configured Python environment:

```bash
DFX_OBSERVER_PID=$$ python /path/to/stage0_backpressure.py --expect blocking \
  --hold-seconds 20 > stage0-base.json

DFX_OBSERVER_PID=$$ python /path/to/stage0_backpressure.py --expect nonblocking \
  --hold-seconds 0 > stage0-fix.json
```

During the base hold window, capture the printed `subject_pid` from a terminal
run and sample it from a second shell. Keep only allow-listed frames in the
public result.

## Decision

- PASS only if every prediction for both arms matches and the base stack chain
  is observed externally.
- A base timeout alone is not a PASS.
- A fix-arm return is not evidence of loss policy correctness.
- After PASS, proceed to Stage 1 design; do not infer EngineCore or DP behaviour
  from Stage 0.
