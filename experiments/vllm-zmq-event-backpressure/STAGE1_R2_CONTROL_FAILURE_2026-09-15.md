# Stage 1 R2 control qualification failure

Date: 2026-09-15

Status: **R2 stopped; no backpressure outcome was scored**

The repaired dependency pool passed its build-identity checks and Stage 1a,
but the first `base/control` cell again stopped with `server_not_ready`. The
server process group was removed and no later cell ran. The retained summary
has SHA-256
`f735a941b2458e8435c3be2e70e52b43694b301e8f3c7a328f56fdef034e3449`.

The private server log showed another bounded setup failure:
`missing_runtime_dependency`, this time for `torchvision`. The pinned vLLM
CUDA requirements specify `torchvision==0.28.0`, while the shared pool was
derived from an older environment and did not contain it. The failure occurred
during EngineCore model warm-up, before hook readiness and before any request
or backpressure trigger.

R2 therefore says nothing about #53859 or #53883. It demonstrates that
content-verifying all visible packages does not prove that the runtime
dependency set is complete. Do not add missing packages one at a time and
create R3. First construct an environment from the pinned CUDA requirements
and require the exact formal server command to reach `/health` without the
fault hook. Only then may a new pre-execution freeze be proposed.
