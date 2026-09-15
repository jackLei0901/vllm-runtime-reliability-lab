# Stage 1 review entry

Date: 2026-09-15

Status: **Stage 1a passed in both formal arm environments; paired build and
dependency identity passed; formal Stage 1 is not frozen**

## Review order

1. `STAGE1_PROTOCOL_DRAFT.md`
2. `stage1_plugin/src/dfx_stage1_backpressure/__init__.py`
3. `stage1_contract.py`
4. `stage1_campaign.py`
5. `tests/test_vllm_zmq_stage1.py`
6. `STAGE1A_CPU_PREFLIGHT.md` and `stage1a_cpu_preflight.py`
7. `STAGE1A_RESULT_2026-09-15.md` and `verify_stage1a.py`
8. `STAGE1_INPUT_REVIEW_2026-09-15.md`
9. `stage1_build_identity.py`, `verify_stage1_build_identity.py` and
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

## Deliberate non-claims

- The plugin has not yet been loaded by a model-serving EngineCore.
- The exact model, launch command and synthetic request are drafted and pass
  the local contract, but are not frozen.
- The installed vLLM 0.20.1 wheel was rejected. Both arms now use the same
  exact-commit precompiled wheel and pinned PyTorch 2.13. The read-only pool is
  attached through one hashed `.pth` file, and all 188 visible distributions
  are recorded with version, `RECORD` hash and source.
- No GPU result or Stage 1 verdict exists.
- This does not test data parallelism or `shm_broadcast`.

## Closed review questions

The wrapper is installed before scheduler construction. It instruments only
the queue's `put`; success requires at least one accepted batch. A stall needs
two prior chunks, and recovery needs a later chunk plus clean completion. Hook
failures have a bounded separate record. Cleanup verifies both the process
group and EngineCore PID/start-time identity are gone.

Stage 1a passed on both source trees in the same PyTorch 2.13 arm environments
prepared for formal execution; see `STAGE1A_RESULT_2026-09-15.md`. The paired
build/dependency verifier also passed, covering 19 installed wheel binaries.
Review of this revised evidence is the remaining gate before freezing.
