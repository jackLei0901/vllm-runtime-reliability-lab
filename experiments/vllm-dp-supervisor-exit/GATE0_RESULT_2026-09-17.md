# DP supervisor exit propagation — Gate 0 result

Date: 2026-09-17

Verdict: **CONFIRMED at the thin lifecycle boundary.**

## Result

The exact `dp_supervisor.py` blob from vLLM `main` at
`0bfc7a15d095fe83ecc82b50561a93c177fece2d` produced these outcomes:

| Cell | Server started | Child exit | Entry returned | Subject status |
| --- | --- | ---: | --- | ---: |
| real SIGTERM after ready | yes | -15 | yes | 0 |
| abnormal child exit before ready | no | 17 | yes | 0 |
| abnormal child exit after ready | yes | 17 | yes | 0 |
| health probe failure after ready | yes | -15 | yes | 0 |

Both abnormal child statuses are observed by the real
`DPSupervisor._monitor_children()` path but are not propagated by
`run_dp_supervisor()`. The probe cell executes the real
`_probe_all_children()` failure branch, sets the shutdown event after the
supervisor becomes ready, and also returns 0. A service manager supervising
that parent process cannot distinguish either failure from intentional
shutdown using process status.

The `-15` values in the intentional and probe-failure cells are produced by
the supervisor forwarding SIGTERM to its still-live child during
`_shutdown_children()`; the test does not kill those children directly. The
intentional control sends a real process signal with `os.kill()`, which reaches
the handler registered by `run()`. The retained subject record reports
`handled_signals=["SIGTERM"]`; all three failure cells report an empty list.
This distinguishes intentional signal-driven shutdown from failure-driven
shutdown directly rather than relying on the supervisor's default signal value.

The column is deliberately named subject status. Equivalence to `vllm serve`
comes from source reading rather than this measurement: `ServeSubcommand.cmd()`
calls `run_dp_supervisor(args)` and returns, and CLI `main()` dispatches that
method without translating its result.

The inconsistency is local to this launcher. The sibling multi-API launcher
uses `wait_for_completion_or_failure()` in `vllm/v1/utils.py`, which raises when
a managed process exits non-zero or when EngineCore dies unexpectedly. The
multi-port DP supervisor observes analogous failures but returns normally.

All four cells were run against the head of vllm-project/vllm#54963,
`61950f9589938ffc73e4f4eaf4181c39f66afbaf`, with
`enable_fault_tolerance=False` explicitly present in the subject arguments.
They produced the same outcomes: real SIGTERM after ready `-15 -> 0`, child
exit before ready `17 -> 0`, child exit after ready `17 -> 0`, and post-ready
probe failure with forwarded child SIGTERM `-15 -> 0`. That PR does not
duplicate this candidate fix.

## Identity and verification

| Subject | Source SHA-256 | Summary SHA-256 |
| --- | --- | --- |
| vLLM main | `d10e5991...a4b9a97` | `eaeff41b...f9690c4` |
| #54963 head | `7c80f64b...1e74588b` | `7a257ec2...ffc0c45` |

Both retained summaries pass `verify_gate0.py` under normal Python and
`python -O`. A deliberately altered summary with fault tolerance enabled is
rejected under `python -O`, confirming that optimized execution cannot remove
the checks. Removing the intentional cell's handler-call record is also
rejected under `python -O`. The verifier pins the two Git blob identities,
requires fault tolerance to be explicitly disabled, and distinguishes
signal-driven shutdown from all three failure-driven shutdowns.

Public closed-shape records:

- `results/vllm-dp-supervisor-exit-gate0-20260917/main-summary.json`
- `results/vllm-dp-supervisor-exit-gate0-20260917/pr54963-summary.json`

## Execution history

Three earlier attempts are retained as runner/environment corrections, not
mechanism evidence:

1. the original 15-second bound expired during full-package import;
2. the parser did not account for vLLM's process-name log prefix;
3. the 2-GiB container externally killed full-package imports.

The first scored run then covered two cells. Revision 4 added the two
post-ready failure cells and replaced the synthetic signal call with a real
`os.kill()` control after review. The source-derived predictions were recorded
before those new cells ran. The superseded two-cell summary hashes are retained
in `GATE0_TWO_CELL_SUPERSEDED_2026-09-17.md`.

The final runner loads the exact upstream source file. It stubs these vLLM
imports outside the exercised lifecycle path:

- `vllm.envs`;
- `vllm.logger`;
- `vllm.utils.system_utils`;
- `vllm.v1.engine.utils`;
- `vllm.entrypoints.launchers.utils.server_utils`.

It supplies a minimal `NoSignalServer` with the same signal-capture override as
upstream. Asyncio, subprocess handling, uvloop, aiohttp, FastAPI, uvicorn, and
psutil are real packages, and the post-ready cells require the uvicorn server
to report started before the fault is triggered. The retained summaries now
also record the 90-second bound and Linux platform identity.

## Duplicate-work search

GitHub issue and PR searches were run for `DPSupervisor exit status`,
`multi-port exit 0`, `run_dp_supervisor`, `exited DP Servers`, and
`DP supervisor non-zero`.

No existing issue or open PR was found that propagates non-FT multi-port
supervisor child/probe failure into the parent process status. The closest
items are:

- #52178 makes the single API-server path exit non-zero on EngineCore death;
  the multi-port supervisor currently masks that status;
- #54963, fault-tolerant survivor behavior, independently controlled above;
- #46534, a closed draft adding a startup timeout rather than non-zero status;
- #44915 and #47076, closed frontend/supervisor implementation changes.

The search is time-bound to 2026-09-17 and must be repeated before filing.

Upstream timing is intentionally held: the lab and regression-test design can
continue now, but no issue or PR should be opened until a human reviewer engages
on #52178. At that point the related multi-port gap can be raised in one line
before implementation work begins.

## Boundary and next gate

This result does not show that a real model-serving child fails this way, and
it does not validate a proposed fix. A probe coroutine that raises rather than
returning a failed health result is also not covered by the four scored cells.
Before opening an issue or PR:

1. repeat the duplicate search immediately before filing;
2. add the smallest upstream regression test around the existing
   `test_handles_child_exit` or lifecycle fixture;
3. run one complete-package process-level control with a real lightweight child
   server in an environment without the 2-GiB constraint;
4. design the fix so intentional SIGINT/SIGTERM remains status 0 while an
   abnormal child or crashed probe task becomes non-zero.
