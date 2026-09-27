# TP=2 in-flight serving gate: bounded run result

## Decision

**Unscored for the export-policy question.** The lab-local Inspector callback
patch built and emitted rank-bound events during a healthy TP=2 serving run.
The healthy control completed. The preregistered hold cell did **not** enter
the one-shot wait, so it contained no verified in-flight delay window. Neither
stock Inspector's aggregate JSON counts nor the request's successful completion
can substitute for that missing window. This run does not establish an
Inspector export gap or admit a new NCCL acquisition probe.

## Frozen inputs and environment

- Protocol, patch and initial runner: `c6d0b7f`; rank-binding correction made
  before the scored control/hold cells: `b8f8f32`. The correction replaces a
  generic NCCL INFO prefix with the Inspector plugin's `nranks: 2 rank: N`
  initialization declaration. It does not change the injected wait condition.
- vLLM `c8602c79062440074a018c1d5f875a5571eb6881`, PyTorch
  `2.13.0+cu130`, NCCL Inspector source `v2.29.7-1` at
  `b91894bd5b190c874d98a017f93f5daa515b65d0`; two RTX 4090 GPUs.
  The runner forced `disable_custom_all_reduce=True` and did not test default
  backend selection.
- Patched standalone Inspector shared library SHA-256:
  `e2f9acc8d985ab936503c58b89ac3965fe8ea2bb254a8ab76522538a0fd6c7ef`.
  The [patch](../../experiments/vllm-tp-dfx/inspector-inflight-trace-v2.29.7-1.patch)
  SHA-256 was
  `ca9f8502d95c953dbe42f956748dd953e8427f8c6c299e68acc739068459bd71`.
  The source-only application check and the host build both passed.
- A private archive of transferred source, the built library, the two later
  cells and the discarded initial smoke is retained off-host. Its SHA-256
  matched at the host and receiving machine:
  `1231c427a1c605377bf925e7bdd484cf8ad8eff3b0fcc8dcb2b59ff7aa97a64a`.
  Raw logs, communicator hashes, PIDs, paths and model output are not published.

## Cell results

| Cell | Observed | Scoring limit |
| --- | --- | --- |
| Initial build/load smoke | Plugin compiled. Both private NCCL logs contained 2,661 patched callback markers. The first runner rejected rank binding because one process's generic NCCL INFO lines also mentioned another rank. | `unscored`; parser assumption, not a missing-rank finding. |
| Healthy control after binding fix | Exit 0; 16 output tokens; rank identities stable; no hold marker. Stock Inspector JSON was available and the patched callback logs were rank-bound. | `healthy_control_complete`; this control alone does not prove a FULL replay cache hit. |
| One bounded hold | Exit 2; 16 output tokens; rank identities stable; arm marker existed, but the entry marker did not. | `unscored / window_or_release_unverified`. No rank-asymmetric in-flight result can be inferred. |

The plugin entry point was discoverable in the host environment and the
serving configuration reported FULL and PIECEWISE graph capture. More
importantly, the private logs for **both** later cells contain the pinned
worker's `Using V2 Model Runner` message. At this revision, the V2 worker
constructs its [V2 model runner](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/v1/worker/gpu_worker.py#L384-L415),
whose FULL serving path calls
[`ModelCudaGraphManager.run_fullgraph`](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/v1/worker/gpu/model_runner.py#L1392).
That manager replays its cached graph through
[`CudaGraphManager.run_fullgraph`](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/v1/worker/gpu/cudagraph_utils.py#L393-L406).
The experiment plugin instead patched the V1 full-graph
`CUDAGraphWrapper.__call__`. This is a source-supported explanation for the
missing entry marker, now corroborated by the actual runner-selection log.
It is not a controlled demonstration that no other predicate could also fail:
the run retained no per-rank plugin-install or per-call activation witness.
That uncertainty belongs to this experiment, not to stock Inspector.

The runner's `window_or_release_unverified` reason starts with the absent
entry marker: the wait was never entered, so **no hold window existed** to
verify. It does not suggest a release-timing failure after a successful hold.

## Next gate

Before another GPU booking, preregister the intended runner. For a
default-serving claim at this build, hook V2's `ModelCudaGraphManager.run_fullgraph`
immediately before its delegated replay, not the V1 wrapper. For a deliberately
V1-only test, set `VLLM_USE_V2_MODEL_RUNNER=0` and label that configuration.
Either way, add a bounded, privacy-safe witness of **which runner and graph
manager installed**, plus a closed count of the selected replay predicates
per rank, without descriptor contents or per-token logging. Mutation-test that
witness on CPU. `VLLM_USE_BREAKABLE_CUDAGRAPH=0` remains pinned; V2 has a
separate breakable path, so its effective value must be recorded. Re-run only
when a missing entry marker can be attributed to a named failed precondition.
A later export-policy claim still requires a verified hold window and a
same-collective stock JSON comparison; aggregate counts remain non-decisional.
