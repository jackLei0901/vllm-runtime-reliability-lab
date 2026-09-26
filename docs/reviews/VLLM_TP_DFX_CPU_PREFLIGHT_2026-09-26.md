# vLLM TP DFX producer capability — CPU/source preflight, 2026-09-26

Result: **SOURCE READY / RUNTIME NOT SCORED**. This is not a TP=2 run, a RAS
capability verdict, a fault reproduction, or admission of a C++ probe.

Companion protocol: [`VLLM_TP_DFX_PRODUCER_SURVEY_2026-09-26.md`](VLLM_TP_DFX_PRODUCER_SURVEY_2026-09-26.md).

## Immutable source identity

- vLLM commit: `6af44e2887e3dbfb1c59807d1315af88f232df11`.
- `cuda_communicator.py` blob: `fb0c281b4a86404d3c191ae3089f4832713e3938`.
- `custom_all_reduce.py` blob: `e18f9194f54edbc8709817fc8ccfad21a8dacd59`.
- `tests/distributed/test_pynccl.py` blob:
  `91dccd5798693eabb8599cea91f597717ccf9e32`.

These identify reviewed source, not the wheel on a future GPU host. That
host's imported vLLM source, wheel and NCCL library must be pinned separately.

## Source checks completed

1. `CudaCommunicator.all_reduce` decides the backend on each call. Its
   startup log lists enabled backends only. `CustomAllreduce.should_custom_ar`
   additionally checks dtype, weak contiguity, size alignment, world size,
   and `inp_size < max_size`. A run can route different tensor shapes to
   different backends. The custom implementation disables itself when its
   GPU P2P check fails.
2. On the pinned NVIDIA CUDA path, `--disable-custom-all-reduce` disables the
   vLLM custom implementation, but **does not alone force PyNccl**. The
   earlier dispatch branches must also be excluded. The candidate isolated
   PyNccl cell sets:

   ```text
   --disable-custom-all-reduce
   VLLM_ALLREDUCE_USE_FLASHINFER=0
   VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=0
   VLLM_ALLREDUCE_USE_SYMM_MEM=0
   VLLM_USE_NCCL_SYMM_MEM=0
   ```

   Quick all-reduce and AITER custom all-reduce are ROCm paths in this
   constructor, not NVIDIA CUDA alternatives. The cell is still **unscored**
   until the target process confirms these settings, shows PyNccl enabled,
   and supplies a positive per-call route witness. The startup log alone
   cannot supply that last fact. A fallback to `torch.distributed.all_reduce`
   must be detected rather than silently counted as PyNccl.
3. The pinned tree already has `tests/distributed/test_pynccl.py`, including
   `test_pynccl_with_cudagraph`. Its worker explicitly captures a PyNccl
   all-reduce and replays the graph. This is a useful **healthy producer
   baseline**, not a vLLM serving hang or repeated RAS-sampling test. A
   bounded extension can replay multiple times with RAS queries before,
   during and after the replay window; exact modifications need review before
   running. It does not establish that RAS counts advance on replay.
4. The PyNccl wrapper obtains symbols from a `ctypes.CDLL(so_file)` handle.
   `LD_PRELOAD` interception remains untested and is not required by this
   producer-capability run.

## Local environment checked

- Windows-visible GPU: one NVIDIA GeForce GTX 1660 Ti, 6 GiB. This cannot
  run the required two-GPU TP comparison.
- Local WSL: Linux and `readelf` available; system Python has no `torch`,
  `ncclras` is not on `PATH`, and `ldconfig` returned no `libnccl` entry.
- No local TCP listener was found at `127.0.0.1:2222` at preflight time.
  No remote GPU instance or NCCL installation was tested.

`ncclras` missing locally says nothing about RAS availability on a future
GPU host. RAS reachability, plugin loading and graph behavior remain runtime
questions with typed unavailable/unsupported outcomes.

## Next gate requiring a dual-GPU environment

Before booking: identify the host's vLLM/PyTorch/NCCL/CUDA versions, two
visible GPU identities, P2P test result, free disk space, and actual plugin
path. Freeze a single booking budget and stop time. No GPU expense or
destructive cleanup is authorized by this document.

Run in order:

1. Existing `test_pynccl` eager and graph tests as baseline, with per-rank
   process identity and configuration retained privately.
2. Verify RAS is enabled and queryable from the container; record its bind
   address/port and query errors separately from empty observations.
3. Repeat healthy graph replay under bounded RAS sampling. Score whether
   per-rank operation counts advance; do not infer the answer from docs.
4. Only if healthy producers work, build a per-call route-witnessed PyNccl
   cell for the bounded A condition (one rank held before launch).
5. Do not run B (all ranks entered but completion held) until a safe,
   reversible injection is independently established. Without B, the
   A-versus-B discrimination question remains **unscored**.

If the imported vLLM build differs from the pinned source, repeat the route
audit before interpreting any output. If no safe B exists, stop rather than
replace it with a count mismatch or an uncontrolled SIGSTOP.
