# Engine liveness: upstream preparation

Status: **internal review material; no upstream post**. This is the compact
exit from the [English contract outline](../ENGINE_LIVENESS_RFC_OUTLINE_2026-09-27.md)
and its [Chinese companion](../ENGINE_LIVENESS_RFC_OUTLINE_2026-09-27.zh-CN.md),
not a new RFC or a claim that any open PR must adopt this design.

## Refresh and provenance

Read-only refresh on 2026-09-28 UTC: `vllm-project/vllm` `main` was
`2407f405b51abd23adbf0203b98464f448c58edf`. The five files that carry
the C1 entry-path claim have the same Git blob IDs as in the earlier
`55de40a2fc` duplicate check:

| File | Blob ID | Relevant current-source lines |
| --- | --- | --- |
| `vllm/v1/utils.py` | `81f4eaf842c2d58e833501e70366b1f831693556` | [`_shutdown_subprocesses` keeps `max(timeout, 5)`](https://github.com/vllm-project/vllm/blob/2407f405b51abd23adbf0203b98464f448c58edf/vllm/v1/utils.py#L478-L485); [`shutdown(None)` gets 5 s, explicit zero does not](https://github.com/vllm-project/vllm/blob/2407f405b51abd23adbf0203b98464f448c58edf/vllm/v1/utils.py#L614-L665) |
| `vllm/entrypoints/cli/serve.py` | `6a8492f4c4b7b0ec51d31435d01fef9e9db8a1fe` | [headless signal path](https://github.com/vllm-project/vllm/blob/2407f405b51abd23adbf0203b98464f448c58edf/vllm/entrypoints/cli/serve.py#L253-L260); [multi-API remaining-budget path](https://github.com/vllm-project/vllm/blob/2407f405b51abd23adbf0203b98464f448c58edf/vllm/entrypoints/cli/serve.py#L398-L413) |
| `vllm/v1/engine/core_client.py` | `5d59bdeaea3a7f69542dee0bd137aaf5c8958111` | [background `MPClient` cleanup](https://github.com/vllm-project/vllm/blob/2407f405b51abd23adbf0203b98464f448c58edf/vllm/v1/engine/core_client.py#L488-L496) |
| `rust/src/managed-engine/src/process.rs` | `cf247c3b8627a4bc33aace08547c609da1309b0f` | [headless child](https://github.com/vllm-project/vllm/blob/2407f405b51abd23adbf0203b98464f448c58edf/rust/src/managed-engine/src/process.rs#L57-L75); [outer minimum](https://github.com/vllm-project/vllm/blob/2407f405b51abd23adbf0203b98464f448c58edf/rust/src/managed-engine/src/process.rs#L123-L151) |
| `rust/src/managed-engine/src/cli.rs` | `0a81ae598e9cf7b341e2e5593c4301f7d97bfd8b` | [`--shutdown-timeout` passed only if positive](https://github.com/vllm-project/vllm/blob/2407f405b51abd23adbf0203b98464f448c58edf/rust/src/managed-engine/src/cli.rs#L126-L132) |

At the same pin, [`CoreEngineProcManager.shutdown`](https://github.com/vllm-project/vllm/blob/2407f405b51abd23adbf0203b98464f448c58edf/vllm/v1/engine/utils.py#L244-L256)
calls [`get_engine_process_shutdown_timeout`](https://github.com/vllm-project/vllm/blob/2407f405b51abd23adbf0203b98464f448c58edf/vllm/v1/engine/utils.py#L46-L69).
That helper grants a separate 15 s process grace only when request and process
timeouts are both zero **and** the platform is ROCm. A common manager boundary
therefore exists; a CUDA change is a design question, not an instruction to
copy the ROCm constant. Whether Rust's SIGTERM reaches EngineCore in the same
process group remains unchecked.

GitHub state snapshot at this refresh: [#24885](https://github.com/vllm-project/vllm/issues/24885)
is closed as stale; [#52365](https://github.com/vllm-project/vllm/pull/52365),
[#54638](https://github.com/vllm-project/vllm/pull/54638),
[#36964](https://github.com/vllm-project/vllm/pull/36964), and
[#36258](https://github.com/vllm-project/vllm/pull/36258) are open;
[#52281](https://github.com/vllm-project/vllm/pull/52281) and
[#58779](https://github.com/vllm-project/vllm/pull/58779) are merged.
[#36964](https://github.com/vllm-project/vllm/pull/36964) has an unresolved
conflict notice and a stale notice, so “open” is not evidence of active review.
The user's [#55537](https://github.com/vllm-project/vllm/pull/55537),
[#52178](https://github.com/vllm-project/vllm/pull/52178), and
[#197232](https://github.com/pytorch/pytorch/pull/197232) remain open.
These are time-bound states, not a schedule for contacting anyone.

## Compact contract card for review

```text
starting ──> ready ──> draining ──> stopping ──> exited
                │
                ├── paused / sleeping ──> ready
                └── terminal failure

For each lifecycle state, separately record:
process reachable | loop responsive | admitted work progressing |
progress unobserved (missing/stale producer) | terminal failure confirmed
```

The diagram is a proposed vocabulary, not a claim that vLLM implements this
enum. In particular, paused/sleeping work and remote-KV waits must not be
turned into false progress alarms; a loop ping is not token progress.

| Budget | Owner and boundary | Question |
| --- | --- | --- |
| Request drain | API/EngineCore policy; `shutdown_timeout` may be zero | Does zero mean abort requests now, without deciding the process-kill grace? |
| EngineCore process cleanup | Parent process manager; explicit zero reaches immediate tree kill on non-ROCm | Should a bounded post-abort cleanup window exist at the common manager boundary? |
| Worker cleanup and escalation | Executor first waits `x`, then needs up to 4 s after SIGTERM | What outer budget invariant accommodates `x + 4` plus delivery margin? |

Two examples, with evidence grades kept separate:

1. **Shutdown (C1/C2).** [K1](ENGINE_LIVENESS_K1_K2_CPU_RUN_2026-09-27.md)
   observed a 100 ms SIGTERM handler cut off under explicit zero grace. K2
   tested inner `x=1,2,5` against an outer `x` and a longer control. The C1
   historical intent question is anchored in [#36666](https://github.com/vllm-project/vllm/pull/36666),
   the rejected [#40985](https://github.com/vllm-project/vllm/pull/40985),
   [#43016](https://github.com/vllm-project/vllm/pull/43016), and the ROCm-only
   [#52281](https://github.com/vllm-project/vllm/pull/52281). No CUDA leak or
   client result was measured; C2's class was already reported in
   [#55632](https://github.com/vllm-project/vllm/issues/55632).
2. **Stall (C3).** [K5](ENGINE_LIVENESS_K5_DUAL_GPU_RESULT_2026-09-27.md)
   scored TP=1 `/health` staying 200 for a 45 s hold with recovery. TP=2
   showed a terminal timeout at the experiment's 30 s setting, but was
   formally unscored because the predeclared method name was wrong. This
   motivates the boundary question in #52365, not a new watchdog.

The first design question is binary: should `shutdown_timeout=0` mean zero
request-drain time **and** zero process-cleanup grace, or zero drain with an
independent bounded cleanup window? The second is whether a responsive loop
alone is sufficient for `/health`, or progress health needs a separate,
opt-in surface after false-positive controls. Neither answer is selected here.

## Scoped C1 comment draft (not posted)

Potential venue: the merged ROCm grace PR [#52281](https://github.com/vllm-project/vllm/pull/52281)
has the closest design context; open [#36964](https://github.com/vllm-project/vllm/pull/36964)
touches abort-output delivery but currently shows stale/conflict notices.
The broader contract card is **not** a comment to paste into either thread.

> I was mapping the shutdown budget across the Python launcher, headless
> `vllm serve`, and the Rust-managed headless path. One question about the
> zero-timeout case: was removing the 5 s process-grace floor in #43016
> intended for CUDA too? #36666 introduced it, and #40985 was closed after
> the minimum was described as intentional. On current main, explicit
> `timeout=0` in `vllm.v1.utils.shutdown()` proceeds to force-kill without a
> cleanup wait, while `_shutdown_subprocesses()` still keeps a 5 s floor.
> Signal-initiated headless and multi-API shutdown also reach the former path;
> Rust's outer 5 s bound does not itself provide an inner EngineCore budget.
>
> #52281 separates immediate request abort from process cleanup on ROCm.
> Should the contract make that separation on CUDA too, with the actual
> duration chosen separately? I have a CPU-only test showing that an explicit
> zero interrupts a 100 ms SIGTERM cleanup handler, but have not measured a
> CUDA resource leak or a client-visible failure.

The comment is intentionally a question about prior intent, not a claim that
#43016 was accidental. If used, its source links can be taken from the pinned
table above. The CPU test report is local-only until the Lab evidence is
published; without it, the comment still rests on the source and PR history.

## Disposition

No new issue, RFC, PR, or GPU run is needed to review this packet. The Lab
outline is ready for design review. The C1 question is the only possible
near-term upstream post from this packet, but its venue and timing remain
open under the existing-PR-first decision. A final source/thread refresh is
needed at the actual posting time because upstream changes continuously.
