# Stage 1 R2 runbook overlay

Status: **frozen before execution**

R2 follows `STAGE1_RUNBOOK_DRAFT.md` and changes only the dependency identity.
Use the same source trees, private command, request, plugin, URLs, bounds and
fixed cell order.

The build records are:

- base: `results/vllm-zmq-backpressure-stage1-build-r2-20260915/`
  `stage1-build-base.json`;
- fix: `results/vllm-zmq-backpressure-stage1-build-r2-20260915/`
  `stage1-build-fix.json`.

Before cell 1, regenerate both records into
`/root/stage1-runtime/build-checkpoint-r2-before` and require byte-identical
matches with those public files. After cell 4, repeat into
`/root/stage1-runtime/build-checkpoint-r2-after` and apply the same requirement.

Write scored summaries to a new result directory and use a fresh private
directory for every cell. Do not reuse or overwrite the R1 control directory.
Stop after any failed control.
