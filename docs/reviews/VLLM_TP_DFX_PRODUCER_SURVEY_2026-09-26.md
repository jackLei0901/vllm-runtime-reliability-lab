# vLLM TP collective DFX producer survey — 2026-09-26

Status: source/documentation survey only. No GPU run, observed TP hang,
new fault category, probe design, or probe-admission claim is made here.
CPU/source preflight result:
[`VLLM_TP_DFX_CPU_PREFLIGHT_2026-09-26.md`](VLLM_TP_DFX_CPU_PREFLIGHT_2026-09-26.md).
Subsequent bounded dual-GPU capability result:
[`VLLM_TP_RAS_CAPABILITY_RESULT_2026-09-26.md`](VLLM_TP_RAS_CAPABILITY_RESULT_2026-09-26.md).

## Question and source boundary

For a tensor-parallel collective that stops making progress, which existing
producer can identify the collective path, communicator, rank/process, and
whether work is still progressing? The question is not whether the Lab can
capture more data; it is whether a named distinction remains impossible after
the existing producers are used.

- vLLM source pinned for routing: `6af44e288`,
  `vllm/distributed/device_communicators/cuda_communicator.py` and
  `pynccl_wrapper.py`. This is a source baseline, not a claim that a later
  installed wheel follows the same route.
- Lab contract: `docs/LOW_LEVEL_CAPABILITY_REQUIREMENTS.md` §5. Its current
  sole probe candidate is the #196968 dump-responder/shutdown-stage boundary.
  This TP survey does not silently add another admitted probe candidate.
- NCCL features below are from vendor documentation for recent releases.
  Availability and event shape must be checked against the exact installed
  NCCL library before a capability result is scored.

## Source-derived route map

At the pinned vLLM revision, `CudaCommunicator.all_reduce` checks several
backends before the PyNccl path and the `torch.distributed.all_reduce`
fallback. Backend selection depends on input, configuration, hardware, and
whether a backend is disabled. Thus no single producer covers *all* vLLM TP
all-reduces by virtue of being installed. The route is selected **per call**:
the startup log enumerates enabled backends, not the backend taken by any
particular collective. For example, `CustomAllreduce.should_custom_ar`
requires a supported dtype/layout and an input smaller than `max_size`; a
small decode call and a larger prefill call in one run can take different
routes. Custom all-reduce may also disable itself when GPU P2P is unavailable.

| Selected route | PyTorch ProcessGroupNCCL Flight Recorder | NCCL RAS / profiler | Current conclusion |
| --- | --- | --- | --- |
| `torch.distributed.all_reduce` fallback | In scope if this is a ProcessGroupNCCL collective and FR is enabled | NCCL behavior still version/config dependent | Source route exists; runtime selection unmeasured. |
| `pynccl_comm.all_reduce` | Not a ProcessGroupNCCL call, so PG-NCCL FR does not directly record this invocation | Candidate coverage because it calls NCCL | Test RAS and profiler on the exact NCCL build, including graph replay. |
| vLLM custom/quick all-reduce implementation | Not a ProcessGroupNCCL call | Do not assume an NCCL producer observes a non-NCCL kernel | Pin the selected implementation and test its existing trace/stack coverage. |
| FlashInfer, AITER, symmetric-memory and other paths | Not automatically covered by PG-NCCL FR | Do not infer NCCL/non-NCCL solely from backend name | Inspect the selected implementation before assigning producer coverage. |

The vLLM PyNccl wrapper loads a selected NCCL shared library through
`ctypes.CDLL(so_file)` and obtains named functions from that handle. A generic
`LD_PRELOAD` interposer must therefore **not** be assumed to intercept these
calls. `VLLM_NCCL_SO_PATH` selects a library; it is not a documented probe
interface. An interception claim requires a minimal runtime test.

## Existing producers that may already close the gap

| Producer | Documented capability relevant here | What remains unverified |
| --- | --- | --- |
| PG-NCCL Flight Recorder | Records ProcessGroupNCCL operations when configured. | Whether the actual vLLM operation selected the c10d fallback; a missing FR entry alone is not a missing collective. |
| NCCL `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,DESTROY` | NCCL documents `DESTROY` for communicator destroy/abort; source contains per-communicator lifecycle logs. | Exact installed version's entry/exit markers, retention, and privacy-safe joining. |
| NCCL RAS | Recent documentation shows communicator hash, rank, PID, status, per-rank collective-operation counts and count mismatch/missing-rank reporting; JSON output is available from NCCL 2.28.7. | Exact target version, enabled state, responsiveness during the fault, whether counts advance on healthy graph replay, and whether the controlled fault pair is distinguished. Equal counts alone do not prove that ranks are blocked in the same collective. |
| NCCL profiler plugin / Inspector | Provides per-communicator/per-collective events; vendor tooling can trace NCCL operations through CUDA graph launches. | Which event types the target plugin/build emits on the selected PyNccl path in eager and graph mode, and their overhead. |
| External native stack / process identity | CPU thread state, process liveness and PID start-time binding. | A CPU wait is not by itself a device-side collective sequence or progress fact. |

Two corrections to the proposed probe shortcut follow. First, RAS may already
provide rank-to-PID and communicator identity for NCCL participants, though
that does not automatically bind a vLLM metrics `engine` label to a process or
prove a dump responder is active. Second, graph replay reduces ordinary host
API-call visibility, but does not establish that NCCL profiler-based tracing
is blind to replay. Neither a lifecycle shim nor a device marker is justified
by source reading alone.

## Capability-run protocol — not yet executed

1. **CPU/source preflight:** record exact vLLM, PyTorch, NCCL, CUDA, plugin
   and Lab versions; check that the profiler plugin and RAS query tooling are
   present. Record RAS bind address/port (default localhost:28028) and test
   reachability from the target container. Unreachable, disabled, unsupported
   and execution failure are producer outcomes, not fault observations.
   Record the dual-GPU host's P2P capability before admitting a custom-AR
   cell. Do not infer interposition from `readelf` alone.
2. **Single-GPU smoke only:** verify producer loading, NCCL version, RAS
   response and a one-rank NCCL event if the chosen workload exercises one.
   This cannot validate a vLLM TP collective or cross-rank attribution.
3. **TP=2 healthy baseline, one route per cell:** on a dual-GPU host, first
   constrain an operation to PyNccl by disabling custom all-reduce and ruling
   out the earlier symmetric-memory, quick, FlashInfer and other dispatch
   branches in the pinned build. The startup enabled-backend list is a useful
   cross-check, **not** a per-call route witness. Require a bounded test-only
   witness that the chosen eager/capture call entered PyNccl; graph replay
   retains the captured route but does not repeat that host call. A separate
   custom-AR cell is conditional on P2P capability and a positive per-call
   witness. Retain private PID/start identity, FR configuration, RAS snapshots
   and profiler output for eager and graph capture/replay. Before any fault,
   explicitly score: **do RAS per-rank operation counts advance during
   healthy graph replay?** If they do not, distinguish capture-only counting,
   unsupported producer, and failed collection; do not presume the cause.
4. **Preregistered discrimination pair, PyNccl route first:**
   A = one rank is held in a bounded host gate before it launches the chosen
   collective, while its peer reaches the call. B = every rank launches the
   chosen collective but completion is held. B is **not yet an admitted
   injection**: first demonstrate a bounded, reversible mechanism and the
   all-entered premise without relying on a mismatched-count operation that
   may error or corrupt data. If B cannot be established safely, stop after
   the healthy baseline and A; report the comparison unscored rather than
   claiming RAS can or cannot distinguish the pair. The question, once B is
   valid, is whether RAS per-rank counts and status separate A from B in both
   eager and graph mode. A non-c10d custom path is a later, separate producer
   coverage question; do not pool it with this pair.
5. **GO/NO-GO:** a Lab-local C++ probe may be designed only if two named fault
   modes remain indistinguishable, the missing fact is a closed transition,
   a healthy/fault pair can validate it, and deleting that fact weakens the
   claim, per the existing §5 gate. Otherwise publish a no-probe-needed
   diagnostic method or an explicit unscored result.

No TP=1 result may be counted as TP=2 evidence. No graph-mode result may be
generalized to all backends. Do not publish raw communicator pointers, host
names, IP addresses, PIDs, log paths, or full native traces; publish bounded
normalized facts (including run-scoped communicator hashes and per-rank
counts when safe) and private-artifact digests under the Lab's existing
privacy rules.

## Review questions

1. Can the pinned build establish a positive per-call PyNccl route witness
   with earlier dispatch branches excluded, including during graph capture?
2. Does the installed NCCL RAS already distinguish the proposed fault pair?
3. Does the installed profiler emit useful events during graph replay, and at
   what overhead? If not, is that a plugin/config/version limit or an actual
   missing observation point?
4. What additional fact, if any, would a C++ probe provide that changes a
   closed Lab attribution rule rather than merely adding another trace?

## Primary references

- vLLM pinned source: <https://github.com/vllm-project/vllm/blob/6af44e288/vllm/distributed/device_communicators/cuda_communicator.py>;
  <https://github.com/vllm-project/vllm/blob/6af44e288/vllm/distributed/device_communicators/pynccl_wrapper.py>;
  <https://github.com/vllm-project/vllm/blob/6af44e288/vllm/distributed/device_communicators/custom_all_reduce.py>.
- PyTorch Flight Recorder: <https://docs.pytorch.org/tutorials/unstable/flight_recorder_tutorial.html>.
- NCCL RAS: <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/ras.html>.
- NCCL debugging subsystems and profiler plugin configuration:
  <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html>.
- NCCL communicator lifecycle logging in current source (not a substitute
  for inspecting the installed version):
  <https://github.com/NVIDIA/nccl/blob/master/src/init.cc>.
- NVIDIA NCCL Inspector: <https://github.com/NVIDIA/nccl/blob/master/plugins/profiler/inspector/README.md>.
- Nsight Systems advanced NCCL tracing: <https://docs.nvidia.com/nsight-systems/UserGuide/>.
