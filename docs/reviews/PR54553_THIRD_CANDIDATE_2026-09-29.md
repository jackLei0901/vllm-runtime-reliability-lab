# Third external candidate: vLLM #54553

Status: bounded source preflight, 2026-09-29; no test or upstream delivery yet. [中文](PR54553_THIRD_CANDIDATE_2026-09-29.zh-CN.md) is authoritative. The visible PR head is `41dddf7`; refresh its head, tests and discussion before using this record.

## Consumer and decision

The consumer is the author of [#54553](https://github.com/vllm-project/vllm/pull/54553) and its reviewer. The PR proposes to bound **fatal-path** EngineCore shutdown, so a stuck teardown does not indefinitely prevent process exit. The author explicitly lists a simulated stuck-shutdown check and a normal `SystemExit` control as outstanding in the [PR discussion](https://github.com/vllm-project/vllm/pull/54553). The decision is whether this exact patch preserves clean shutdown and produces a bounded, nonzero fatal exit when teardown cannot finish. Its motivating XPU driver wedge is not being reproduced by the Lab.

## Frozen comparison before implementation

| Item | Prediction and limit |
| --- | --- |
| Ordinary reproduction | A helper that sleeps and then exits, or a test of copied timeout logic, proves little about the PR's EngineCore exception/finally path. The author already knows the missing two controls. |
| Possible Lab addition | A small **subprocess-level CPU fixture against the real pinned `EngineCoreProc` path**, run unchanged against the base and patch: controlled fatal error plus a teardown hold; separately, a `SystemExit`/requested-shutdown negative control. Capture exit code, elapsed time and whether teardown entered. If this requires copying the new branch into a test helper, it is not Lab value. |
| Expected distinction | Base: fatal teardown stays alive past the bounded observation window; patch: exits nonzero near the configured deadline. Both versions: intentional `SystemExit` follows the pre-existing shutdown path, without a new forced-exit marker. A normally finishing fatal teardown must not be killed by the new deadline. These are **predictions**, not observations. |
| Refuting or stopping result | The actual process path cannot be exercised without replacing the decision logic; a current upstream test already establishes the same base/fix/control result; the patched process does not exit within the declared bound; or clean shutdown takes the forced-exit branch. Stop and report the exact result rather than adding apparatus. |

The patch [adds a default 60 s setting](https://github.com/vllm-project/vllm/pull/54553/files) and, in `EngineCoreProc`, runs fatal-path `engine_core.shutdown()` on a daemon thread, joins with that timeout, then calls `os._exit(1)` if it is still alive. The `SystemExit` branch calls shutdown directly. This is a source reading, not a verified runtime result. The user-visible status and supervisor restart after exit are **outside** this CPU fixture. The author has an XPU reproduction rig; a Lab simulation must not be presented as reproducing the underlying GPU wedge.

## Immediate gate and budget

1. Read the current PR head and [#58279](https://github.com/vllm-project/vllm/pull/58279) for overlap, existing tests and new author results. If the author has delivered the same process-level negative controls, close this as duplicate review; do not compete with them.
2. In at most **two hours of CPU preflight**, identify the minimal way to enter the real EngineCore process `try/except/finally` path with a controlled fatal event and blocked `shutdown()`. Before coding, write a failing base fixture and declare a short configured deadline and an independent parent-process watchdog. Do not use a 60 s default for a CPU litmus.
3. Proceed only if the fixture can run both source revisions without scenario-specific branches in the scoring rule. Bound any subsequent implementation and test to **four additional hours**. If it requires a GPU, XPU, model download, or broad installation rebuild, stop and ask for a new decision; no hardware is authorized by this record.

Scoring will be `supported` only for valid identity plus the base/patch fatal contrast and both clean and normally finishing fatal controls; `refuted` for a valid contrary process outcome; `unscored` for missing identity, missed entry/teardown witnesses, apparatus timeout or incomparable revisions. Freeze the exact fixture and these thresholds before the first scored run. Publish only a minimal runnable reproduction after user review; do not post, ping a maintainer, open a PR, or count a Lab delivery from this preflight alone. Track instrument time separately under R2.
