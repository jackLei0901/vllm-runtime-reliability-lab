# TP=2 V2 in-flight serving gate — CPU-frozen protocol

## Boundary

This supersedes the unscored [V1-wrapper hold experiment](VLLM_TP_INFLIGHT_SERVING_RESULT_2026-09-27.md)
for a **new** capability run; it does not rewrite that run. The retained
server logs said `Using V2 Model Runner`. At vLLM
`c8602c79062440074a018c1d5f875a5571eb6881`, the
[GPU worker chooses V2](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/v1/worker/gpu_worker.py#L384-L415),
the [V2 serving path calls](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/v1/worker/gpu/model_runner.py#L1386-L1392)
`ModelCudaGraphManager.run_fullgraph`, and its
[base method replays](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/v1/worker/gpu/cudagraph_utils.py#L393-L406)
the cached graph. A hold immediately before that call tests the default V2
path; it does not test V1 by setting `VLLM_USE_V2_MODEL_RUNNER=0`.

The question remains whether a lab-local Inspector callback trace can expose
an in-flight, per-rank channel-start asymmetry during a real TP=2 serving
request. No result here is a transport-hang cause, progress verdict, or
admission of a new C++ acquisition probe. A host-side delayed replay is a
controlled mechanism, not an organic failure.

## Frozen apparatus and CPU gate

- The [V2 general plugin](../../experiments/vllm-tp-dfx/serving-v2-stall-plugin/llr_tp_v2_stall.py)
  patches `ModelCudaGraphManager.run_fullgraph` and its constructor in the
  experiment process only. It never patches vLLM or NCCL source. The
  entry-point metadata is colocated on `PYTHONPATH`, with
  `VLLM_PLUGINS=llr_tp_v2_stall`. Installation requires fresh private marker
  and witness directories, a 2–5 second bound, and
  `VLLM_USE_BREAKABLE_CUDAGRAPH=0`. vLLM may fork its workers after loading
  plugins in the frontend. An `after_in_child` handler resets the inherited
  counters, lock, hold state and witness path to the child's PID; a spawned
  worker installs directly. `process_origin` distinguishes these two paths.
  Temporary witness filenames include the writer PID, so two children cannot
  collide on one parent's temporary file.
- Rank 1 alone may hold, once, before a cached FULL replay. The precondition
  includes a unique V2 manager, TP size 2, no breakable graph, an operator arm
  file, and an unspent hold. The entered marker is written atomically before
  sleeping. The original replay follows the bounded wait. The runner creates
  a separate `observe` marker only after construction, capture and warmup;
  `observed_full_cached_calls` counts cached FULL replays in that request
  window. Lifetime `full_cached_calls` alone cannot pass the control, since
  warmup may saturate it before a request begins.
- Each process writes one private `witness.<pid>.json`. The closed shape records
  installation, unique manager construction, selected V2 configuration,
  observed graph-manager kind/mode, rank, breakable setting, and saturated counts
  (`0`, `1`, `2+`) for replay, armed replay, FULL cached replay and eligible
  replay, including the observed-window FULL count. It writes only on state
  transitions, never one raw line per token. `FULL_DECODE_ONLY` is accepted
  alongside `FULL` and `FULL_AND_PIECEWISE`: each permits cached FULL replays
  at the pinned revision.
  No descriptor values, pointers, model output or stack frames enter it.
  [The reader](../../experiments/vllm-tp-dfx/v2_activation_witness.py) rejects
  unknown fields, ambiguous managers, invalid identity or missing required
  facts. The [runner](../../experiments/vllm-tp-dfx/serving_v2_stall_gate.py)
  binds each witness filename to the live NCCL-log PID and `/proc` start ticks.
  `runner_v2` and the FULL-call mode remain consistency checks, not independent
  evidence of V2 selection or extra mutation coverage: this manager and call
  site already constrain them.
- CPU fake-manager and reader tests must pass the positive path and mutations
  of arm, rank, TP size, V2 selection, manager uniqueness, mode, cache and
  breakable state, observed-window gating, and two forked children writing
  distinct PID-keyed witnesses without collision. The fork test skips on
  platforms without `os.fork`. Entry-point discovery, bounded writes and
  malformed witness rejection are also required. A CPU fake cannot establish that the
  installed wheel imports the plugin or that CUDA replay executes it.

## One dual-GPU booking, only after the CPU gate

Pin the same two-RTX-4090 host class, Qwen3-4B-Instruct-2507-v2 model, vLLM
commit, PyTorch `2.13.0+cu130`, NCCL `v2.29.7-1`, and forced
`disable_custom_all_reduce=True` as in the earlier serving run. Keep
`VLLM_USE_V2_MODEL_RUNNER` **unset**, so the runner selection is observed,
not forced. Set `VLLM_USE_BREAKABLE_CUDAGRAPH=0` explicitly. Use a fresh
owner-only directory per cell, including `inspector/` and `witness/`; set
`NCCL_DEBUG=TRACE`, `NCCL_DEBUG_SUBSYS=INIT,PROFILE`, `NCCL_DEBUG_FILE` to
`<private>/nccl.%p.log`, and stock Inspector JSON to `<private>/inspector`.
Set `NCCL_INSPECTOR_ENABLE=1`,
`NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=500`, and
`NCCL_INSPECTOR_DUMP_VERBOSE=1`; without the enable switch, Inspector is
disabled by default and absence of callbacks is an apparatus failure.
The standalone Inspector patch/library are the separately pinned
[acquisition control](VLLM_TP_INFLIGHT_SERVING_GATE_2026-09-27.md), not a new
product probe. Use an external 180-second hard timeout and stop after one
healthy and one 2–5 second hold cell. Never broaden the run live to another
backend, model or graph setting.

1. **Build/load smoke:** verify exact source/build identities, plugin discovery,
   two rank-bound callback logs, and exactly one valid V2 manager witness per
   worker PID, keyed to that PID. A witness keyed to a frontend or other
   non-worker PID does not substitute. If any is absent, stop as `unscored`
   before injection.
2. **Healthy control:** generate 16 tokens without an arm file. Both ranks
   must finish with stable PID/start-tick identities and show at least one
   actual V2 FULL cached replay **after** the `observe` marker on both ranks;
   no entered marker may exist. A FULL capture startup line or warmup replay
   alone does not pass.
3. **Bounded hold:** construct/capture the engine before arming. Rank 1 must
   atomically write its entered marker and witness `hold_entered=true` on a
   cached FULL replay in the observation window. With this request shape,
   the prefill is expected to use the mixed/piecewise path and the first held
   FULL replay to be the first decode step; the closed witness, not this
   expectation, decides whether the cell is scorable. While the marker is
   younger than the declared hold and
   the request is pending, snapshot callback logs and stock Inspector JSON;
   finish the request, then require rank 1 to catch up. A missing marker is
   `unscored`, not a negative Inspector result.

The entered marker and runner use Linux `CLOCK_MONOTONIC`, a system-wide clock;
the cross-process hold-window comparison is not portable without rechecking
that clock contract.

The runner's public line contains only closed activation facts, bounded counts,
ordinal collective position, binary digest and typed outcome. Raw witness
filenames, PID/start ticks, full NCCL/Inspector logs, paths, and communicator
hashes remain private. A callback asymmetry alone proves acquisition
feasibility. The runner deliberately reports
`stock_same_key_comparison: not_scored`: a separate, same-run join to the
**same pending communicator and sequence** is required before claiming a
stock-Inspector export-policy gap. Aggregate JSON counts cannot fill that gap.

## Decision after the run

- `healthy_v2_full_replay_observed` then `start_asymmetry_observed`: proceed to
  the private same-key stock comparison, without yet proposing an upstream
  change.
- A valid V2 witness but no entry marker: name the failed predicate from the
  closed witness; mark the cell `unscored`, then revise the experiment on CPU.
- Plugin or manager witness absent/ambiguous or keyed to a non-worker PID,
  malformed callbacks, wrong
  version, timeout, or unstable identity: stop `unscored`; do not buy another
  GPU cell merely to hunt for a passing configuration.
