# Stage 1 review entry

Date: 2026-09-15

Status: **Stage 1 R3 passed the four-cell single-GPU serving contract; public
summaries and the reviewed result are ready for verification**

## Review order

1. `STAGE1_R3_RESULT_2026-09-16.md` and the four summaries under
   `results/vllm-zmq-backpressure-stage1-r3-20260916/`
2. `verify_stage1_results.py`
3. `STAGE1_PROTOCOL_DRAFT.md`
4. `stage1_plugin/src/dfx_stage1_backpressure/__init__.py`
5. `stage1_contract.py`
6. `stage1_campaign.py`
7. `tests/test_vllm_zmq_stage1.py`
8. `STAGE1A_CPU_PREFLIGHT.md` and `stage1a_cpu_preflight.py`
9. `STAGE1A_RESULT_2026-09-15.md` and `verify_stage1a.py`
10. `STAGE1_INPUT_REVIEW_2026-09-15.md`
11. `stage1_build_identity.py`, `verify_stage1_build_identity.py` and
   `STAGE1_BUILD_IDENTITY_RESULT_2026-09-15.md`

Stage 0 is already committed evidence and is not being revised by this work.

## What exists

- A test-only `vllm.general_plugins` package that activates only when both its
  entry-point name and `DFX_STAGE1_ENABLE=1` are supplied.
- An EngineCore-local publisher pause controlled by a file outside that
  process.
- EngineCore-local scoped `PR_SET_PTRACER` authorization.
- Counters attached once to the real queue's `put` path. They distinguish
  accepted non-null batches from non-blocking `queue.Full` failures without
  double-counting `put_nowait` delegation.
- A stdlib-only external campaign runner that binds to the PID produced by the
  hook, observes a streaming request, samples `/health`, captures a bounded
  stack and cleans up the server process group.
- Import binding that requires the loaded `kv_events.py` bytes to equal the
  selected source tree's blob.
- A closed per-arm dependency manifest and complete installed-wheel binary
  identity, plus EngineCore-local mapped-binary evidence for the formal run.
- A pure scoring contract with closed hook-file shapes and no response-content
  retention.
- CPU tests for the cross-process file release, drop counter, progress rules,
  stack ordering and four-cell classification.

## Deliberate non-claims and limits

- The PASS covers one GPU and one EngineCore; it does not test data parallelism
  or `shm_broadcast`.
- #53883 restores progress by dropping event batches. This is not a reliable
  delivery result, and subscribers may not observe a sequence gap.
- R3 used self-contained arm environments. Its private identity record covers
  source trees and mapped worktree binaries, but it is not the complete public
  dependency manifest prepared for R2.
- The test-only plugin adds one wrapper frame to the sampled publisher stack.

## Closed review questions

The wrapper is installed before scheduler construction. It instruments only
the queue's `put`; success requires at least one accepted batch. A stall needs
two prior chunks, and recovery needs a later chunk plus clean completion. Hook
failures have a bounded separate record. Cleanup verifies both the process
group and EngineCore PID/start-time identity are gone.

Stage 1a passed on both exact R3 source environments with PyTorch 2.13.0+cu130.
All four serving cells then passed in fixed order. See
`STAGE1_R3_RESULT_2026-09-16.md` for the result, exclusions and the post-run
verifier correction.
