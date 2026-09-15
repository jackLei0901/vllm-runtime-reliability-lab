# Stage 1 R1 control qualification failure

Date: 2026-09-15

Status: **R1 stopped; no backpressure outcome was scored**

The first `base/control` cell stopped with `server_not_ready`. The server process
group was removed and no later cell ran. The public summary is retained under
`results/vllm-zmq-backpressure-stage1-r1-control-failure-20260915/`.

The private server log had SHA-256
`bd6085127007837d9093e90f3f35b7912564508d317e659df568d288c3e119dc`.
Its bounded diagnosis was `missing_runtime_dependency`: FlashInfer imported
`pynvml`, but the cleaned dependency pool did not provide that module. The
failure happened during EngineCore initialization, before the hook became
ready and before any request or backpressure trigger.

R1 therefore says nothing about #53859 or #53883. Its freeze and failed summary
remain unchanged. R2 adds `nvidia-ml-py==13.610.43` with its complete metadata
and seven content-verified files. The experiment logic, source trees, plugin,
command, request, matrix, bounds and scoring contract are unchanged.

Before R2 is frozen, both build identities and both Stage 1a arms must pass
again with the repaired pool.
