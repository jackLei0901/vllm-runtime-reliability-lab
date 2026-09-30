# SM90 block-FP8 performance preparation

Status: CPU-reviewed preparation; not publicly frozen, built or GPU validated.

The [Chinese charter](../../../docs/kernel/SM90_BLOCK_FP8_PERF_CHARTER.zh-CN.md)
is authoritative; an [English translation](../../../docs/kernel/SM90_BLOCK_FP8_PERF_CHARTER.md)
is available. Start with the [runbook](RUNBOOK.zh-CN.md), not a GPU command.
Runbook section 7 contains the 2026-09-30 host-preflight summary and unresolved
booking gates. Only a basic Torch CUDA smoke check ran; no vLLM performance
experiment ran, and the packet still has no public freeze. The inspected host
was shut down at the user's request; provider billing confirmation is separate.

`protocol.py` defines the proposed paired measurements and decision rules.
`build_perf.py` is a successor to the unchanged historical R2 repair builder.
It registers base and variant under separate, validated private namespaces and
uses the same compiler/linker flags for both. Mock CPU tests do not establish
successful compilation or runtime dispatch isolation.

Run the CPU preflight tests from the Lab root:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_sm90_perf*.py' -v
.venv/Scripts/python.exe experiments/kernel-operand-contracts/sm90-block-fp8-perf/timing.py --session A --plan
.venv/Scripts/python.exe experiments/kernel-operand-contracts/sm90-block-fp8-perf/correctness.py --plan
```

The builder requires Linux, the pinned source/CUTLASS, the real pinned CMake
compile reference and patched Torch headers. Use `--help` for required paths
and `--plan` for source validation without compilation. Builds require new
output directories. Never overwrite historical R2 artifacts or receipts.

The only proposed C++ difference is the small-M dispatch predicate; these
performance arms do not contain the B1/B2 guard prototype. The separate
namespaces and identical `-Wl,-Bsymbolic` flags are intended to prevent ELF
symbol interposition; successful dual loading and dispatch must still be
checked at runtime before interpreting any timing.

The draft tools now include:

| Tool | Purpose |
| --- | --- |
| `model_preflight.py` | Bind pinned config and local weight files |
| `wheel_preflight.py` | Retain/inspect the parent archive without installing or importing Torch |
| `serving_preflight.py` | Bind an existing install, dependencies, source and actual parent-wheel extensions without device operations |
| `serve_trace.py` | Native serving; separate Nsight and Torch shape-profile passes |
| `witness_review.py` | Validate a reviewed capture/replay mapping, not automatic attribution |
| `identity_probe.py`, `trace_tools.py` | Verify actual dual-library dispatch from marked launches |
| `correctness.py` | Fixed 176 arm cases, graph replay and FP16 |
| `timing.py`, `runtime.py` | Paired CUDA-event timing, raw clocks, fail-closed scoring |
| `summarize.py` | Public allowlist; raw material stays private |
| `freeze_packet.py` | Byte manifest before the user commits and pushes |

CPU checks do not establish compilation, runtime isolation, serving or profiler
schema compatibility. Serving needs pinned Python plus recorded binary provenance;
Stage 0 admits an explicitly verified parent cu130 wheel, not an unknown wheel
or a claimed full-pin build. The archive is prepared; runtime compatibility remains unconfirmed.
No full source build is silently funded. Forced Nsight then forced shapes take
priority; default Nsight is optional, with no default shapes. Missing graph-node
or shape evidence cannot be replaced with concurrency assumptions.

Generate the manifest after review; a local manifest is not a public freeze.
GPU tools require `--freeze-commit` and compare the public manifest with local
bytes before execution. No optimization or end-to-end speedup is established.
The v0.7 plan adds preparation P (60 minutes) before A (100) and conditional
B (70), on one rental. Public freeze verification precedes P. Runbook section
8 supersedes the historical no-GPU-only preparation order in section 7.
