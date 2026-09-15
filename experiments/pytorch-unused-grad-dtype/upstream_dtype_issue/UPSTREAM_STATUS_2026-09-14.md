# Upstream status — PyTorch FSDP2 mixed-gradient dtype defect

Last checked: 2026-09-14

## Issue

- Upstream report:
  [pytorch/pytorch#196996](https://github.com/pytorch/pytorch/issues/196996)
- Title: `[FSDP2] Gradient accumulation with fresh or unused parameters raises a mixed-dtype assertion`
- State: open
- Routing: `module: fsdp`, `oncall: distributed`,
  `oncall: distributed parallelisms`, `triaged`

## Maintainer response

A PyTorch maintainer stated in the issue that a fix is already in progress.
This lab will therefore not open a competing implementation PR unless
maintainers later ask for one.

## Remaining action

1. Wait for the maintainer-owned fix to be published and linked.
2. Record the fixing PR and merge commit in this file.
3. If maintainers request verification, rerun the single-GPU `control`,
   `last-microbatch`, and `unused-placeholder` matrix on the fixed nightly.
4. After the fix lands, synchronize the status of #196996 and mark this record
   complete.

The issue is not considered resolved merely because a fix is being developed.
Completion requires an upstream merged change or an explicit maintainer
resolution.
