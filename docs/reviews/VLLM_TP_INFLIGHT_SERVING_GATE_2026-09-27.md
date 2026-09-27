# TP=2 in-flight serving export gate — local preregistration

## Question and limits

Can a bounded, private Inspector callback trace identify a rank-specific NCCL
kernel-channel start asymmetry **during a real vLLM serving request**, when the
stock Inspector completed-collective JSON cannot export that pending event?
This is an export-capability comparison, not a new fault category, hang detector,
or probe-admission claim. An experimentally delayed rank is not a naturally
occurring transport failure.

Pin vLLM `c8602c79062440074a018c1d5f875a5571eb6881`, NCCL `v2.29.7-1`
(`b91894bd5b190c874d98a017f93f5daa515b65d0`), the same Qwen3-4B model,
PyTorch/CUDA pair, TP=2, and `disable_custom_all_reduce=True` as in the
[serving baseline](VLLM_TP_SERVING_CALLBACK_RESULT_2026-09-27.md). No default
backend, custom-all-reduce, or cross-version claim follows from this cell.

## Stages before renting GPUs

1. Freeze the [one-shot serving hold plugin](../../experiments/vllm-tp-dfx/serving-stall-plugin/llr_tp_stall.py). Its general-plugin entry point runs in workers; only rank 1, TP size 2, an operator-armed, already-cached FULL graph replay can wait. It writes one private entered marker, waits 2–5 seconds, then always calls the original wrapper. No signal-stop or irreversible device fault is used. CPU fake-wrapper tests must show that removing any eligibility fact makes the control fail.
   The experiment package includes minimal `.dist-info` entry-point metadata, so
   placing that directory on `PYTHONPATH` makes the plugin discoverable without
   installing into or rebuilding vLLM. CPU entry-point discovery was checked.
2. Freeze the [Inspector experiment patch](../../experiments/vllm-tp-dfx/inspector-inflight-trace-v2.29.7-1.patch). It records communicator hash, sequence number, channel count, channel ID and channel stop through NCCL's existing debug channel. The raw trace remains private. The patch must pass `git apply --check` against the pinned NCCL source. A GPU-side build/load smoke is still required; local Windows/WSL has no CUDA compiler.
3. Freeze the [offline normalizer](../../experiments/vllm-tp-dfx/inflight_trace.py) and CPU vectors. It must reject ambiguous rank binding, malformed or truncated histories, missing hold/release controls, invalid channel IDs, and a rank that does not catch up. A raw channel-start total is never treated as a collective index.
4. Freeze the [serving gate runner](../../experiments/vllm-tp-dfx/serving_stall_gate.py) and its exact command before opening the host. It reads rank-bound, PID-start-time-stable NCCL logs from a fresh private directory and runs the healthy and hold cells separately. The test runner itself is not an operator-facing collector.

## GPU cells and capture boundary

The GPU booking starts with a **fail-fast build/load gate**, not the serving
fault cell: on the dual-GPU host, verify the exact vLLM/NCCL/PyTorch versions;
copy the frozen files into a fresh private checkout; apply the patch to the
NCCL tag; build the standalone Inspector with the host's CUDA toolkit; confirm
its load in a bounded two-rank healthy smoke and that both rank-bound debug
logs contain well-formed `LLR_TP_EVT` descriptors. If any check fails, stop
and report `unscored`; do not alter the frozen script or quietly substitute a
different library. The source-only `git apply --check` and CPU tests do not
establish binary loadability. Set `NCCL_DEBUG=TRACE` and
`NCCL_DEBUG_SUBSYS=INIT,PROFILE` (the INIT header binds the rank), point
`NCCL_DEBUG_FILE` to `<private-dir>/nccl.%p.log`,
and enable stock Inspector JSON in a separate private subdirectory. Keep an
external hard timeout and stop after the healthy and one bounded-hold cell.

- **Healthy control:** same 16-token TP=2 request, no arm marker. Both ranks
  complete with stable rank identities and no entered marker. Startup log must
  show only `PYNCCL` for TP all-reduce. This control alone does not witness a
  FULL graph cache hit; the hold cell's marker is the direct witness for that
  later cell, rather than an inference from a capture-size log.
- **Bounded hold:** construct the engine before arming. Start the request; rank
  1 writes its entered marker immediately before a cached FULL replay and sleeps
  at most five seconds. Snapshot callback logs and stock Inspector JSON before
  arming, during the marker-to-release interval, and after the request finishes.
  The in-window snapshot must be taken while the request remains pending and
  the marker's monotonic age is less than the declared hold. The external runner
  has a separate hard timeout; timeout is `unscored` and triggers process cleanup.
- **Negative controls:** remove the arm marker (no hold), use an uncached graph
  or non-FULL mode (no hold), and remove one rank's callback log in an offline
  vector (`unscored`, never `rank_missing`). No second GPU campaign is opened
  merely to exercise these already CPU-testable exclusions.

The private debug directory is created empty, owned by the runner and mode
0700. Full logs, PIDs, paths, raw communicator hashes and model output remain
private. Each debug file has a 32 MiB read budget and the experiment has a
hard wall-clock timeout. If that budget, rank/process identity, event history,
communicator join, sequence occurrence or channel cardinality is ambiguous,
the result is `unscored`; a missing callback is never alone evidence of a
missing rank. Publish only a run-scoped communicator/sequence ordinal, typed
producer outcome, bounded channel counts and private-archive digest. The
debug-channel patch is a **lab-local acquisition control**, not an acceptable
always-on export policy; its callback logging overhead is separately measured.

## Decision

`start_asymmetry_observed` requires the same collective key to have a channel-start on rank 0
but not rank 1 **during** the verified hold, a healthy equal prefix before it,
and rank 1 catching up after release. The stock Inspector comparison must be
made in the same run and for the same pending collective; an aggregate JSON
count alone is insufficient. The current runner reports that aggregate only
as context and marks `stock_same_key_comparison: not_scored`. A successful
callback run establishes acquisition feasibility, not an export-policy gap.
Matching the pending `(communicator, sequence)` key against a private stock
JSON snapshot is a separate gate before any export-gap claim.

- If an unmodified, privacy-bounded producer already exports that same in-flight
  fact, record `no_new_export_needed`.
- If only the lab-local callback trace exposes it and stock Inspector omits the
  pending record, record `export_policy_gap_observed_for_this_cell`. This may
  motivate an Inspector-side bounded exporter design, **not** a new NCCL
  acquisition probe or an upstream issue yet.
- Otherwise record `unscored` with the exact failed precondition. Do not infer
  a transport cause or widen the Lab's closed verdict vocabulary.

## Source anchors

The pinned [vLLM graph wrapper](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/compilation/cuda_graph.py) distinguishes capture from cached replay. Inspector's [collective descriptor](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/plugins/profiler/inspector/inspector_plugin.cc#L203-L232) retains `seqNumber` and `nChannels`; its [channel callback](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/plugins/profiler/inspector/inspector_plugin.cc#L259-L293) retains the parent collective and `channelId`. Stock JSON is gated on completed channels ([source](https://github.com/NVIDIA/nccl/blob/v2.29.7-1/plugins/profiler/inspector/inspector_plugin.cc#L400-L422)). These are source facts, not evidence that a real serving hold has already been run.
