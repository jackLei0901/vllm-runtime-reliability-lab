# vLLM TP PyNccl / NCCL RAS capability result — 2026-09-26

Follow-up: the [A/B and Inspector run](VLLM_TP_RAS_INSPECTOR_AB_RESULT_2026-09-26.md)
adds a validated B condition, private raw-evidence retention, and a matching
NCCL Inspector build. Statements below that call these gates untested describe
only this earlier run; the follow-up supersedes them.

Status: **bounded producer-capability result; no new C++ probe admitted**.
Companion source survey:
[`VLLM_TP_DFX_PRODUCER_SURVEY_2026-09-26.md`](VLLM_TP_DFX_PRODUCER_SURVEY_2026-09-26.md).

## Question and boundary

Can the installed NCCL RAS per-rank `AllReduce` counters distinguish healthy
PyNccl collective calls from healthy CUDA-graph replays, and do they expose a
controlled peer-before-launch hold? This does **not** test a vLLM serving TP
hang, the all-entered-but-stuck condition, or a custom all-reduce route.

The GPU host imported vLLM source at
`c8602c79062440074a018c1d5f875a5571eb6881`, **not** the surveyed
`6af44e2887e3dbfb1c59807d1315af88f232df11`. The cells call
`PyNcclCommunicator.all_reduce` directly, so they establish that selected
route without inferring it from a serving startup log. They cannot be used to
assert the full `CudaCommunicator` dispatch behavior of either revision.

## Runtime and artifact identity

- Linux, two NVIDIA GeForce RTX 4090 GPUs (24,564 MiB each); GPU-to-GPU P2P
  read test reported `CNS` in both directions. No custom all-reduce cell was
  admitted on this host.
- vLLM source: `c8602c79062440074a018c1d5f875a5571eb6881`, imported from
  the source checkout through its existing environment; torch
  `2.13.0+cu130`; installed NCCL `(2, 29, 7)`.
- The RAS endpoint on `127.0.0.1:28028` returned JSON while the PyNccl
  communicator existed. The script discarded hostnames, IP addresses, PIDs,
  and raw communicator hashes before printing results; one run-scoped
  communicator ordinal with ranks 0 and 1 was observed.
- Healthy baseline script:
  [`ras_graph_baseline.py`](../../experiments/vllm-tp-dfx/ras_graph_baseline.py),
  run-time SHA-256 `f5f41b5e83b7374ff33ef75be47192084c3c223c52c15d183d74ce273133e79e`.
- Peer-hold script:
  [`ras_peer_hold.py`](../../experiments/vllm-tp-dfx/ras_peer_hold.py),
  run-time SHA-256 `f99008d915d387d95e939c2be79d96360e2a79436bcf5890c812135fa9b69399`.
- These two exact script versions were overwritten in place and are not
  retained in the checkout. A read-only search of the original host found
  only the current versions; the private evidence archive has no `.py`
  files. The links point to later versions, **not** to blobs matching the
  run-time digests. Treat the earlier hashes as transcript identifiers;
  this run's exact scripts cannot be independently inspected from the
  currently retained artifacts.
- Existing upstream `test_pynccl_with_cudagraph` passed once in the same
  environment. The scripts below are separate bounded extensions, not a
  modification of that upstream test.

Both scripts ran through `torchrun --standalone --nproc-per-node=2` with
`NCCL_RAS_ENABLE=1`, `PYTHONDONTWRITEBYTECODE=1`, and a 120-second process
timeout. The local environment needed `multiprocess==0.70.16` to collect the
existing upstream test. It was installed into the existing venv with
`uv pip install --python .venv/bin/python 'multiprocess==0.70.16'`; this is a
post-install environment, not a pristine copy of the original venv.
No source checkout files were modified. Four pre-existing staged FP8 config
files on the host were left untouched.

## Observed cells

The final baseline script changes the input before **every** call/replay and
checks that the output equals twice the current input, after CUDA
synchronization. This prevents a passing output left over from graph capture
from masquerading as an executed replay.

| Cell | Per-rank RAS AllReduce before → after | Program result | Interpretation |
| --- | --- | --- | --- |
| Eager, 20 calls | rank 0: 2 → 22; rank 1: 2 → 22 | Completed with `result: pass`; all 20 changing-input outputs correct | RAS counter advanced by exactly 20 per rank. |
| Graph, 20 replays | rank 0: 1 → 1; rank 1: 1 → 1 | Exit 0; all 20 changing-input outputs correct | Healthy replays executed without advancing this RAS counter. |

An earlier version of the baseline script, which checked only a constant
final output, also showed eager +20/+40 and graph +0/+0 for 20/40 iterations.
Those runs are **supporting observations only**; the table above uses the
stronger, changing-input assertion and its separately pinned script digest.

For the safe A condition, rank 1 delayed its **host call** to eager
`all_reduce` by three seconds after both ranks completed warmup. Rank 0
issued its call, and rank 1 queried RAS before releasing the hold. Two
independent runs gave the same sequence:

| Snapshot interval | Rank 0 count | Rank 1 count |
| --- | --- | --- |
| Before → during hold | 2 → 3 | 2 → 2 |
| During hold → after release | 3 → 3 | 2 → 3 |

Both runs printed `result: pass` after checking the released collective; the
first run's process exit code was explicitly checked as 0.
This is evidence that, on this eager PyNccl path, RAS exposes an issued-count
asymmetry while the peer is held before its call. It is **not** evidence that
RAS proves which rank has completed a collective or can distinguish this
condition from an all-entered hang.

The script has no pre-snapshot assertion that rank 0 reached its call before
rank 1's query; the three-second delay creates a margin, and the observed
rank-0 count increment is the after-the-fact call witness. Rank 0 then
executes `torch.cuda.synchronize()` before the post-release barrier. RAS
returned the during-hold snapshot while rank 1 had not issued its call, so
RAS was responsive in this bounded blocked-peer cell. No external stack was
captured at the query instant.

The test process emitted a PyTorch warning that its default process group was
not explicitly destroyed before exit. This did not turn either run into a
failure, but the scripts should add orderly teardown before reuse as general
test fixtures.

## Decision and remaining gates

1. **RAS per-rank counts are usable for this eager PyNccl issue-count
   observation** and cannot be treated as a device-side progress signal for
   this NCCL 2.29.7 CUDA-graph replay cell. The latter is an observed
   version/path limitation, not a claim about every NCCL release or profiler.
   The hold snapshot shows the counter can advance before that collective
   completes. The graph starting count is consistent with counting capture
   once; replay produced correct changing-input outputs without further
   counter increments.
2. **Conditional vLLM serving implication, not an observed serving result.**
   The pinned [`CompilationConfig` source](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/config/compilation.py)
   describes `FULL_AND_PIECEWISE` as the V1 default and full CUDA graphs for
   decode batches; the pinned
   [`CudaCommunicator` source](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/distributed/device_communicators/cuda_communicator.py)
   has a PyNccl route. If a particular serving collective is captured and
   replayed on this NCCL build, equal, frozen RAS counts would **not** prove
   that its ranks are progressing or that none is waiting for a peer.
   Eagerly issued collectives may still show an issued-count asymmetry. The
   effective graph mode can be downgraded by model/backend/configuration, and
   this run did not pin which actual serving collectives were captured.
   Verify that per-call placement before recommending RAS counts as an
   operator diagnostic for a specific vLLM deployment.
3. **A-versus-B fault discrimination remains unscored.** There is no
   validated B condition in which both ranks are shown to have issued while
   completion remains delayed. A candidate is a calibrated, bounded
   `torch.cuda._sleep(cycles)` on rank 1's current stream immediately before
   `comm.all_reduce`; the pinned
   [`PyNcclCommunicator` source](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/distributed/device_communicators/pynccl.py)
   passes that current stream to NCCL. Before admitting this private-API
   injection, assert that both host calls were issued, that the device-side
   delay overlaps the RAS snapshot, that completion follows release without
   a timeout, and that the same workload without sleep completes normally.
   The eager A/B prediction is unequal versus equal issued counts; the graph
   prediction is frozen counts in both conditions. Neither prediction has
   been run. Do not substitute a mismatched-count collective or uncontrolled
   process stop.
4. **No Lab C++ probe is justified yet.** An existing NCCL profiler/Inspector
   path or Nsight trace may cover graph replay even though these RAS counters
   do not. The profiler/plugin capability was not run here; `nsys` was not on
   `PATH`, and no profiler binary was identified in a shallow host search.
   NVIDIA's [Inspector source and build instructions](https://github.com/NVIDIA/nccl/blob/master/plugins/profiler/inspector/README.md)
   make it a concrete next producer, but the `master` documentation is not
   evidence that a plugin built against the host's NCCL 2.29.7 headers will
   load or report replay events. Pin a compatible source/header version,
   verify plugin loading, then check per-replay coverage and dropped records.
   This is an untested producer, not a demonstrated instrumentation gap.
5. Any future proposal must first identify the exact attribution claim that
   existing producers cannot make, demonstrate the A/B separation safely,
   and satisfy the Lab's minimal-probe gate in
   [`LOW_LEVEL_CAPABILITY_REQUIREMENTS.md`](../LOW_LEVEL_CAPABILITY_REQUIREMENTS.md).

The normalized outputs above were transcribed from the interactive execution
record. Raw RAS JSON and raw terminal logs were not retained as publishable
artifacts, so this is not a closed offline-verifiable bundle or a basis for an
upstream bug report. The scripts and source/runtime pins make the capability
test rerunnable, with that evidence limitation explicit.
On a future GPU booking, retain raw RAS responses **privately**, with digests
and a predeclared sanitization step; publish only normalized rank/ordinal
facts. The previous host was shut down after this campaign, so this retention
gap cannot be retroactively closed from the transcribed table.
