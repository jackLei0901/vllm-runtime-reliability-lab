# Engine liveness K1/K2: Linux CPU run

Run date: 2026-09-27 Pacific / 2026-09-28 UTC. Status: **source-equivalent CPU
result**, not a full EngineCore shutdown campaign or a pinned binary-wheel run.
The [inventory](../ENGINE_LIVENESS_CONTRACT_INVENTORY_2026-09-27.md) defines C1
and C2; the [runners](../../experiments/engine-liveness-contract/README.md)
exercise only the named Python functions in a fake process tree.

## Identity and execution boundary

- vLLM Python source came from `git archive` of
  `c8602c79062440074a018c1d5f875a5571eb6881`, in an isolated directory;
  archive SHA-256: `8a744d3685798e60988957cee919069ae8679be639a0da88ed5eb86c8192dcec`.
  The existing checkout was not modified.
- Imported `vllm.v1.utils.shutdown` and
  `MultiprocExecutor._ensure_worker_termination` from that archive. Their
  containing files had SHA-256
  `550156cc5256696d5615769c63c9912c6edbdb02e167dc961a7802f2c8da0036`
  and `ceb3477473bdb1de36e01687e0b2083ad239d42b1ddc2ec8cec16ba2c477efa6`.
- Python 3.12.3, Linux, no GPU, 2 GiB memory cgroup and half-CPU quota. The
  dependency environment was pre-existing and **not** rebuilt for this source
  commit. Because a Git archive has no generated `vllm._version`, the runners
  reported `vllm_version="dev"` and `pinned_version_match=false`. The commit
  and source-file hashes, not that version-string flag, establish the Python
  source identity. This does not pin the dependency build.
- K1 runner SHA-256:
  `368fe05b08897ab5d63872aae0967f97f76c1dfa3931f72b96d2a2f936c0a8db`.
  K2 runner SHA-256, with an explicit CPU-only `fork` option:
  `d2ec1ee74c63dabe60f830d5b246670a462c08bdd0d8c4384a676fd035c2b2d8`.
  K2's default remains `spawn`; the fork option changes fake-engine startup,
  not either vLLM function or the scoring rule. The fake workers already use
  fork. No CUDA context was initialized.

## K1: zero shutdown grace

All three cells returned a complete result; runner exit code was 0.

| `shutdown` timeout | Fake-engine result | Teardown marker | Helper duration |
| --- | --- | --- | ---: |
| `0` | SIGTERM handler entered, then SIGKILL (`-9`) | absent | 0.002 s |
| `None` | exit `0` | present | 0.102 s |
| `1.0` | exit `0` | present | 0.103 s |

In all cells the child was no longer alive after shutdown. The two controls
show that the 100 ms handler can finish when a grace exists. Verdict:
`c1_zero_grace_confirmed` **for this helper and fake process tree**. The runner
also read the config default as `0` and found the launcher source passing its
shutdown timeout to the engine client. No actual model, request drain, leaked
resource, or CUDA teardown was observed.

## K2: nested worker budget

Two attempts with the default `spawn` mode ended with shell status 137 before
any `K2_RESULT` line: the first attempted the default `x=1,2,5` matrix; the
second attempted only `x=1`. The 2 GiB memory limit is a plausible constraint,
but the cgroup `oom_kill` counter remained zero, so the exact kill source was
not established. Neither attempt is scored. Orphaned fake-engine/worker
processes were identified by PID and stopped before the next attempt.

With `--engine-start-method fork`, an `x=1` control/sample trial and then the
preregistered `x=1,2,5` matrix both completed with runner exit code 0. The
full matrix is the scored result:

| Cell | Outer grace | Outer return | Inner SIGTERM first observed | Inner schedule completed | Engine exit |
| --- | ---: | ---: | ---: | --- | --- |
| control, inner `x=1` | 11 s | 5.021 s | 1.004 s | yes, at 5.007 s | `0` |
| nested `x=1` | 1 s | 1.005 s | 1.003 s | no | SIGKILL (`-9`) |
| nested `x=2` | 2 s | 2.005 s | 2.002 s | no | SIGKILL (`-9`) |
| nested `x=5` | 5 s | 5.008 s | 5.004 s | no | SIGKILL (`-9`) |

Because the forked fake engine sets `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS`
after inheriting the parent's imported modules, a cached import-time default
would be an apparatus risk. The first inner SIGTERM followed the configured
`x` at 1.003, 2.002, and 5.004 s, respectively; the control completed at
`x + 4` (5.007 s). These observations show that the real inner function used
the per-cell setting despite the fork startup mode.

Every cell recorded worker-handler readiness, EngineCore SIGTERM reception,
and entry into the real inner termination function. `workers_left_alive` was
zero **before** the runner's own cleanup in every cell. In the nested cells,
the outer tree kill removed the slow workers; the runner's fallback kill had
nothing left to do. The control proves the apparatus can observe the full
inner sequence. Verdict:
`c2_tested_nested_budget_shortfall` **for these three `x` values and slow fake
workers**. The first inner SIGTERM can race with the outer kill, as it did
here; the inner four-second post-SIGTERM wait cannot fit inside the same outer
`x`. The general `x + 4 > x` statement remains a source-level budget argument,
not an extrapolation from three timings.

## Retained evidence and decision

Full stdout/stderr logs are retained privately, outside the public Lab repo.
Their SHA-256 digests are: K1 `ed2a5b51176b588faa0a5f5f89c1ad1a674617f650583562007eefa256970f92`;
failed default-spawn K2 attempts
`f840d14f5344f64bdfc153c2c45e8a4374722ed88b878f2102c5d7e9f0047ea7`
and `c90cb18f7dd58d66616e039ad54bdae5b844430ad2cd36d959038834d4bbce91`;
fork `x=1` trial
`d35af51cec7665e4baf44861a028438fd5ddc99cfbe39f011c67070e0b6191b6`;
fork full matrix
`b65b9e74fa0f5ff0665b222dcdcf2cc887d3868d277705a48564c9d7b5cac023`.

K1 and K2 now justify a concrete **contract review**: separate request-drain
time from process-kill grace, and reserve an outer budget longer than the
worker's inner escalation schedule. They do not yet demonstrate a user-visible
vLLM failure. K5 is the next behavioural gate for the executor-dependent
health claim C3. K4 is optional corroboration of CUDA shutdown consequences,
not a prerequisite for discussing the C1/C2 timeout design. Any upstream RFC
first requires a full duplicate check. No upstream issue or PR is implied by
this CPU run alone.
