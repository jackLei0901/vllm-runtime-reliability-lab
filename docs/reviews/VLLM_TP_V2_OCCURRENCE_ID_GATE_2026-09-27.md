# TP=2 V2 serving: occurrence-identity follow-up gate

## Claim and boundary

This is a **new**, preregistered capability cell, not a reinterpretation of
the [unscored prior run](VLLM_TP_V2_INFLIGHT_SERVING_RESULT_2026-09-27.md).
The question is whether a Lab-local Inspector trace can bind a rank's
`KernelChStart` to a unique collective callback during a bounded V2 cached-FULL
replay hold. Neither callback proves collective completion or token progress.
The host-side hold remains a controlled delay, not an organic NCCL fault.

The old `(communicator, sequence)` identity is invalid under replay. The
[v2 local patch](../../experiments/vllm-tp-dfx/inspector-inflight-occurrence-v2.29.7-1.patch)
adds a communicator-local, atomic occurrence counter to each Inspector
`CollInfo`. It logs that ID and `func` on CollStart and obtains the **same
ID through the parent `CollInfo`** for KernelChStart/Stop. No callback is
joined by nearest timestamp or by a reused NCCL sequence. The occurrence
is an exact *within-rank* identity; equal occurrence ordinals across ranks
are an explicit hypothesis to check, not guaranteed by the counter itself.

## CPU and build gates before booking

1. Pin the Inspector source tag and commit, apply the patch with
   `git apply --check`, then compile it. The new patch currently has only
   local diff-syntax and Python-contract checks. A source-level
   `git apply --check` against NCCL `v2.29.7-1` (`b91894bd5b19`) now passes;
   Linux compilation and load remain unverified. If the dual-GPU instance is
   the only CUDA build host, run those gates at the start of one booking and
   stop before engine startup on failure. Keep the old plugin binary and old
   run immutable.
2. The separate [v2 parser](../../experiments/vllm-tp-dfx/inflight_trace_v2.py)
   accepts only `LLR_TP_EVT_V2`, with `comm`, `occurrence`,
   `func`, NCCL `seq`, and channel index/count. It rejects duplicate or
   non-contiguous occurrence IDs per communicator, duplicate channel starts,
   a channel outside its exact parent descriptor, descriptor changes, and
   rank-disagreeing descriptors. Its CPU fixtures must show that two
   collectives with the **same NCCL sequence** remain distinct, and that the
   historical duplicate-key shape fails the healthy control.
3. The runner saves private, SHA-256-addressable `before`, `during` and
   `after` callback snapshots. Their bytes remain private, under the same
   bounded archive policy as raw NCCL logs; only digests may be published.
   A rejected triplet must be replayable offline without reconstructing
   snapshots from final logs.
4. A passing healthy control alone creates a private, one-use receipt bound
   to its `after` snapshot digest. Hold mode verifies that snapshot and
   rejects differences in model argument, Inspector patch/binary, loaded
   NCCL library, V2 stall plugin, witness reader, runner, parser, and the
   available vLLM/Torch build fingerprints. This enforces
   control-before-hold without treating the receipt as a health signal.
   Use separate fresh cell directories and pass the exact control receipt
   path to the hold invocation; a used receipt cannot arm another hold.

## One dual-GPU booking

Use the same pinned vLLM revision, model, TP=2 hardware class and forced
PyNccl route as the prior cell, with fresh owner-only directories. Before
engine construction, the runner records vLLM/Torch versions and relevant
module/native-entrypoint digests, plus the Inspector patch and binary
digests. **After** construction it resolves both worker PIDs and hashes the
one NCCL shared library actually mapped by both processes. This `/proc`
check, not `torch.cuda.nccl.version()`, binds the PyNccl `ctypes.CDLL` load.
Version text and these selected file digests are not a full wheel
attestation; retain the pinned environment manifest privately. If the
loaded library cannot be identified, stop unscored. Record the final
archive digest privately.

1. **Build/load smoke:** the new marker format must appear in exactly two
   rank-bound NCCL logs, with unique worker PID/start identities and V2
   manager witnesses. Otherwise stop; do not arm.
2. **Healthy control:** run the same 16-token request without a hold. Require
   observed-window cached-FULL replay on both ranks, no hold marker, complete
   request, stable identities, and `validate_control_history(before, after)`.
   In particular, every occurrence must have one descriptor and the rank
   occurrence sets/descriptors must agree. Any callback-validation failure
   is `unscored / callback_history_invalid` and is a **hard stop before the
   hold cell**. Do not select a favourable callback subset or modify the
   parser on the running host.
3. **Single bounded hold:** only after the healthy control passes, arm rank 1
   once for 2–5 seconds before a cached FULL replay. Require an entered
   marker, a pending request during the snapshot, stable identities, exact
   callback ancestry and a completed request after release. Score only an
   exact rank-asymmetric occurrence whose descriptor agrees across ranks
   after catch-up. Ambiguity, timeout or mismatch is unscored; stop after
   this one hold regardless of outcome.

An external 180-second timeout applies to each cell. There is no live
expansion to another backend, model, graph setting or second hold.

## Secondary, non-scoring prediction

The pinned [V2 warmup](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/v1/worker/gpu/warmup.py)
is expected to contribute one prefill and three decode steps. The previous
closed aggregate had four runs of 73 AllReduce-like records of lengths
`73, 219, 73, 1,095`, consistent with those four warmup steps followed by
one request prefill and 15 request decode steps. In the new request window,
predict 16 × 74 collective occurrences per rank: 73 `AllReduce` and one
`AllGather` per step. The held rank-0 occurrence should be `AllReduce` at
position 1 of decode step 1 (request step index 2), with channel starts
matching that step's collective channel count.

The parser publishes `func`, per-rank occurrence counts after `observe`, and
the step index/position **only when** both ranks' ordered request callbacks
actually satisfy the entire 16-step, 73-plus-1 shape. Otherwise position
and the predicted-match field are null. A prediction failure cannot change
`start_asymmetry_observed` into a pass or a failure; it is reported for
interpretation and possible source-map correction, not used to choose a
favourable callback or alter the verdict.

## Interpretation

If the healthy control and hold both pass, the result establishes that this
local profiler **acquires** an in-flight, rank-specific channel-start fact
for one controlled serving window. It does not establish that stock
Inspector exports the same incomplete collective: that requires a separate
same-occurrence export comparison. If the healthy control fails, the result
is a contract or apparatus finding and the GPU campaign ends there. The
v0.2 progress verdict remains independent of all native evidence.
