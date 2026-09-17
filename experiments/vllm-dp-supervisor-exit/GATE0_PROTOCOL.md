# DP supervisor exit propagation — Gate 0 protocol

Status: revisioned protocol; Revision 4 predictions were recorded before the
new cells in that revision were executed.

Revision 4 (after the 2-cell scored run, in response to review): added
post-ready child exit, post-ready probe failure, and a real `os.kill` SIGTERM
control; predictions derived from source before these cells were executed.
The superseded 2-cell summaries are retained by hash in
`GATE0_TWO_CELL_SUPERSEDED_2026-09-17.md`.

Revision note: an initial infrastructure attempt used a 15-second outer bound.
On the 0.5-core host, both cells timed out during vLLM import without emitting
a subject record. No mechanism outcome was observed. The scored revision uses
a 90-second per-cell outer bound; predictions, subject behavior, and all other
acceptance criteria are unchanged.

A second unscored attempt completed both subject processes, but vLLM log
decoration prefixed the marker line with the process name. The exact-prefix
parser therefore retained no subject record and the verifier failed closed.
The scored revision locates one unique marker within a decorated line and adds
stdout hashing; predictions and subject behavior remain unchanged.

The full-package import was subsequently killed by the 2-GiB container before
either subject record was emitted. The scored runner therefore loads the exact
`dp_supervisor.py` blob from current `main` and supplies minimal stubs only for
unreached vLLM model, logging, and HTTP-launcher imports. Third-party asyncio,
process, HTTP, and event-loop libraries remain real. This is a thin lifecycle
test of the exact upstream class and entry point, not an end-to-end serving
test.

## Question

On current vLLM `main`, does the non-fault-tolerant multi-port
`DPSupervisor` return a non-zero process status when one managed API-server
child exits abnormally?

This is a boundary check for the follow-up explicitly excluded from
vllm-project/vllm#52178. It does not test model serving, fault-tolerant rank
recovery, or GPU behavior.

## Subject and changed variables

The runner loads the exact upstream source blob, executes the real
`DPSupervisor`, and invokes the public `run_dp_supervisor(args)` entry point in
a fresh process. Network probing and model startup are replaced with
deterministic process-lifecycle adapters; the real `run()`,
`_monitor_children()`, `_handle_signal()`, and `_shutdown_children()` paths
remain in use.

Four cells distinguish startup from the serving state and child failure from
health failure:

1. `intentional-sigterm-after-ready`: the supervisor's real uvicorn server is
   running when the subject process receives `SIGTERM` through `os.kill()`.
2. `abnormal-child-exit-before-ready`: the child exits with status 17 before
   the supervisor becomes ready.
3. `abnormal-child-exit-after-ready`: the child waits until the supervisor's
   real uvicorn server has started, then exits with status 17.
4. `probe-failure-after-ready`: a live child remains, but the real
   `_probe_all_children()` path observes a failed health probe after the
   supervisor server starts.

Each fresh subject process has a 90-second outer bound. This includes importing
vLLM, which took about 32 seconds on the 0.5-core execution host during the
unscored diagnostic run.

## Frozen predictions

| Cell | Child observation | Subject process status |
| --- | --- | --- |
| intentional SIGTERM after ready | -15 during forwarded cleanup | 0 |
| abnormal child exit before ready | 17 | 0 |
| abnormal child exit after ready | 17 | 0 |
| probe failure after ready | -15 during forwarded cleanup | 0 |

The second prediction follows from source reading: `_monitor_children()` logs
and breaks when a child is no longer alive, `run()` then completes after
cleanup, and `run_dp_supervisor()` does not translate the child exit status
into an exception or process status.

Gate 0 confirms the candidate defect only if both abnormal child cells retain
status 17, the post-ready cells prove the uvicorn server started, the probe
cell proves its injected failure was observed, every real top-level entry
returns normally, and every containing subject process exits 0. Any runner
exception, timeout, missing record, or different child status fails closed.

`vllm serve` equivalence is a source-reading claim, not a measurement in this
gate: `ServeSubcommand.cmd()` calls `run_dp_supervisor(args)` and returns, and
the CLI `main()` calls that dispatch function without translating its result.

## Relationship to #54963

vllm-project/vllm#54963 changes the supervisor only when fault tolerance is
enabled and states that behavior without `--enable-fault-tolerance` is
unchanged. Gate 0 first scores current `main`; the PR is a duplicate-work
control, not part of the primary verdict. All four cells must be rerun with
`enable_fault_tolerance=False` explicitly present in the subject arguments.

## Stop rule

Do not design or propose a fix in this gate. A confirmed result advances to a
duplicate search, a minimal upstream regression test, and only then a fix
design. A non-confirming result stops this candidate.
