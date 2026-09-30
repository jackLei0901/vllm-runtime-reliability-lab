# SM90 block-FP8 GEMM decode performance: charter

Status: 2026-09-30, v0.7 same-card preparation and freeze materials; not publicly frozen, no performance measurements.
The [Chinese version](SM90_BLOCK_FP8_PERF_CHARTER.zh-CN.md) is authoritative.
This is the second owned low-level project, after the
[contract repair](SM90_BLOCK_FP8_PROJECT_REQUIREMENTS.md). It is a measurement
project first; one candidate change is attempted only if a precommitted gap gate
passes. Failure of that candidate closes this experiment, not the performance
workstream.

## 1. Questions

On one SM90 GPU, for the linear layers of one block-FP8 model served at decode
sizes (M = 1–64):

1. **Q0 workload:** which kernel executes each block-FP8 linear layer, and what
   fraction of decode GPU time do those kernels take?
2. **Q1 baseline:** per-call latency of the CUTLASS SM90 blockwise GEMM, timed
   alone, under a stated cache condition, with copy bandwidth as a reference.
3. **Q2 dispatch:** is latency non-monotonic at the dispatch boundary
   M = 63 → 64, and what do the M = 15/16/17 controls show about configuration
   versus tile count?

## 2. Source basis (pinned `7b054aca96cea8be1369d651c3434ad140580b92`)

### 2.1 Dispatch

[`scaled_mm_blockwise_sm90_fp8_dispatch.cuh` L204–225](https://github.com/vllm-project/vllm/blob/7b054aca96cea8be1369d651c3434ad140580b92/csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm90_fp8_dispatch.cuh#L204-L225):

| Condition | MMA tile | Cluster | Kernel schedule |
| --- | --- | --- | --- |
| `M % 4 == 0` | 128×128×128 (M×N×K) | 1×2×1 | TMA warp-specialized cooperative |
| `M % 4 != 0` | swapped problem; tile 128 along N, 16 along M | 1×1×1 | TMA warp-specialized ping-pong |

The kernel is `GemmUniversal<Shape<int,int,int,int>, Mainloop, Epilogue>` with no
explicit tile scheduler (L125–126), and
[`cutlass_gemm_caller`](https://github.com/vllm-project/vllm/blob/7b054aca96cea8be1369d651c3434ad140580b92/csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/cutlass_gemm_caller.cuh#L34-L63)
passes a default-constructed `KernelHardwareInfo`. CUTLASS v4.7.1 is pinned to
`cb4247394dd82148787aed73e5dc7cef33cbf862`.
[`tile_scheduler.hpp` L107–128](https://github.com/NVIDIA/cutlass/blob/cb4247394dd82148787aed73e5dc7cef33cbf862/include/cutlass/gemm/kernel/tile_scheduler.hpp#L107-L128)
maps the default tag to `PersistentTileSchedulerSm90`. Both
[cooperative L226–246](https://github.com/NVIDIA/cutlass/blob/cb4247394dd82148787aed73e5dc7cef33cbf862/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp#L226-L246)
and [ping-pong L229–249](https://github.com/NVIDIA/cutlass/blob/cb4247394dd82148787aed73e5dc7cef33cbf862/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp#L229-L249)
query the device SM count when the supplied count is nonpositive. Ping-pong
advances consumer work by the number of MMA warp groups. Do not equate logical
tiles with launched CTAs or assume identical work assignment in both schedules.
Exact grid sizing and observed geometry remain runtime witnesses: Nsight
Systems records grid and block dimensions without hardware counters.

### 2.2 Three quantities, kept separate

- **Logical output tiles:** non-swapped `ceil(M/128)·ceil(N/128)`; swapped
  `ceil(N/128)·ceil(M/16)`.
- **Clusters:** non-swapped tiles grouped in pairs along N; swapped clusters are
  single tiles.
- **Launched CTAs:** scheduler-dependent; recorded from the trace, not derived.

Logical tiles for N = 4096 (32 tiles along N):

| M | Path | Logical tiles |
| --- | --- | ---: |
| 15 | swapped | 32 |
| 16 | non-swapped | 32 |
| 17 | swapped | 64 |
| 63 | swapped | 128 |
| 64 | non-swapped | 32 |

**M = 63 → 64 is the primary boundary comparison:** one more row, a different
configuration and four times fewer logical tiles. **M = 15/16/17 is a separate
control:** 15 and 16 have equal tile counts but different configurations; 15
and 17 share a configuration but differ in tile count. Tile counts describe the
geometry; they do not establish the cause of any latency difference.

### 2.3 Kernel selection

The CUDA block-FP8 list ([`kernels/linear/__init__.py` L461–469](https://github.com/vllm-project/vllm/blob/7b054aca96cea8be1369d651c3434ad140580b92/vllm/model_executor/kernels/linear/__init__.py#L461-L469))
is FlashInfer/DeepGEMM dynamic, then DeepGEMM, then CUTLASS. `VLLM_USE_DEEP_GEMM`
defaults to 1, and `has_deep_gemm()` accepts the copy vendored in the vLLM wheel
([`import_utils.py` L500–507](https://github.com/vllm-project/vllm/blob/7b054aca96cea8be1369d651c3434ad140580b92/vllm/utils/import_utils.py#L500-L507)).
The dynamic kernel sends M < 32 to FlashInfer and M ≥ 32 to DeepGEMM
([`flashinfer.py` L149–170](https://github.com/vllm-project/vllm/blob/7b054aca96cea8be1369d651c3434ad140580b92/vllm/model_executor/kernels/linear/scaled_mm/flashinfer.py#L149-L170)).
**Expected from source, to be verified in Stage 0: on a default Hopper install,
CUTLASS is a fallback path, not the default.**

### 2.4 Prior dispatch work and prediction limits

[#44572](https://github.com/vllm-project/vllm/pull/44572), authored by
yewentao256 and merged 2026-06-13, introduced this swap path, following
[#43706](https://github.com/vllm-project/vllm/pull/43706) (merged 2026-06-01).
The former reports roughly 17–18 versus 50 microseconds at several small,
non-divisible-by-four M values for (N,K)=(4096,7168). Its old arm includes
padding/allocation/copies and its timing/cache conditions differ from ours.
It motivates P1/P4 but does not isolate a pure-GEMM ratio, establish a padding
cost here, or measure M63/64 on our four shapes. This study confirms and extends
prior work to divisible-by-four capture points, not discovery of swap_ab.

[#52775](https://github.com/vllm-project/vllm/pull/52775) (SM120, merged
2026-08-19) restricts a small-M swap path after prefill regressions;
[#40170](https://github.com/vllm-project/vllm/pull/40170) (SM120, closed
unmerged) explores finer M-tile selection. These are tuning precedents,
not SM90 measurements. Large-M changes are out of scope.

## 3. Workload and target policy (decide before freezing)

**Model:** `Qwen/Qwen3-8B-FP8`, revision
`220b46e3b2180893580a4454f21f22d3ebb187d3`, TP = 1, BF16 activations.
The [pinned config](https://huggingface.co/Qwen/Qwen3-8B-FP8/blob/220b46e3b2180893580a4454f21f22d3ebb187d3/config.json)
confirms hidden 4096, intermediate 12288, 32 query heads, 8 KV heads, head
dimension 128, 36 decoder layers, dynamic e4m3 FP8 and weight blocks [128,128].
The download preflight must hash actual config bytes, compare these fields,
and retain the revision and weight-file identities before running.
The four shapes below are derived from this config. Each projection executes
once per decoder layer; decode measurement excludes startup and prefill.

| vLLM layer | Derivation | Expected (N, K) |
| --- | --- | --- |
| `qkv_proj` (merged) | N = (q_heads + 2·kv_heads)·head_dim, K = hidden | (6144, 4096) |
| `o_proj` | N = hidden, K = q_heads·head_dim | (4096, 4096) |
| `gate_up_proj` (merged) | N = 2·intermediate, K = hidden | (24576, 4096) |
| `down_proj` | N = hidden, K = intermediate | (4096, 12288) |

The unquantized `lm_head` is excluded. These replace the DeepSeek-V3 benchmark
shapes, which come from a model that does not fit one H800.

**Target policy (accepted for this bounded study):** when capability, shape and
configuration checks pass, Hopper is expected to select FlashInfer/DeepGEMM. This charter studies the **forced CUTLASS fallback
configuration**. The rationale is that CUTLASS is what serves block-FP8 linear
layers on SM90 when DeepGEMM is disabled or unavailable, and vLLM maintains a
DeepGEMM-disabled benchmark for this path. All conclusions are limited to that
configuration and make no claim about default serving. Selection settings:
`VLLM_DISABLED_KERNELS=FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel`,
with `VLLM_USE_DEEP_GEMM=0` also recorded. Sufficiency is verified from the
selection log plus the trace; neither alone counts. If this policy is rejected,
stop here and write a charter for the default path instead.

**Graph relevance is conditional:** the pinned balanced capture grid includes
1,2,4 and multiples of 8 below 256
([config L2413–2445](https://github.com/vllm-project/vllm/blob/7b054aca96cea8be1369d651c3434ad140580b92/vllm/config/vllm.py#L2413-L2445)).
Interactivity mode, explicit capture sizes and appended token/decode maxima
can add other values. Stage 0 fixes balanced mode, max_num_seqs=64, no
speculation, no custom capture list, and a token budget above the capture
ceiling. Expect a uniform logical M63 decode step to execute at graph M64;
verify the resolved capture list and execution, not concurrency alone.
For this grid, captured M4 and M8..64 use the non-swapped base path.
M63/64 is a microbenchmark proxy, not two distinct served graph sizes.
Record prefill M/path passively as a later lead, without widening timing gates.

## 4. Stages

**Stage 0: workload witness.** Serve the model in both configurations (default
and forced CUTLASS) at fixed decode concurrencies of 1, 16, 63 and 64, under Nsight
Systems. For each configuration, record:
- the `Selected <kernel>` log lines (a selection witness only);
- the traced kernel names and grid/block dimensions for each linear layer (the
  execution witness);
- actual scheduled-token counts, logical M and graph-padded execution M;
- the fraction of decode-step GPU time spent in block-FP8 linear kernels,
  using summed non-overlapping kernel durations within the same step window.

Use 128 input tokens and 64 generated tokens per request, fixed token-ID inputs,
ignore EOS, no speculative decoding, TP=1 and prefix caching disabled. Record
all compilation/graph settings. A concurrency value is not proof of execution
M. Separate prefill/warm-up; retain at least ten steady decode steps per load.
Layer/shape attribution requires NVTX or an equivalent op-metadata witness,
not a kernel-name guess. Record the resolved capture list; the expected M63
logical to M64 execution mapping is assessed only for observed uniform
63-token decode steps. If trace, source/binary identity or shape attribution
is unavailable, Q0 is unscored; do not optimize as if the path were witnessed.

Stop if CUTLASS kernels are not observed under the forced configuration. Record
the default configuration only as context.

**Stage 1: harness (CPU side, before booking).** Time the GEMM alone with
pre-quantized inputs (no activation quantization), with the R2 minimal extension
built from the pinned #55537 head, without the additional B1/B2 repair guards.
Inherited #55537 operand checks remain in both performance arms.

- **Rotating weights:** `B = max(3, ceil(2·L2_bytes / per-set bytes) + 1)`
  distinct weight/scale sets, where `L2_bytes` is read from device properties at
  runtime. They are visited round-robin in a fixed order.
- **Graph construction:** one CUDA graph per (shape, M, arm), holding `R`
  consecutive calls that cycle through the sets. Activations and outputs are
  preallocated. Nothing is allocated or synchronized inside the timed region
  except by the op itself, identically in both arms.
- **Warm-up and timing:** `W` graph replays, then CUDA-event timing of each
  replay; per-call time is replay time divided by `R`.
- **Cache condition:** rotation reduces cache reuse but does not guarantee that
  every call reads all weights from HBM, and it does not reproduce serving
  exactly. With hardware counters, measure DRAM read bytes and L2 hit rate at up
  to three points. Otherwise, state the cache condition as an assumption.
- **Hot contrast:** the same GEMM, same graph structure and same R, but with a
  single weight set.
- **References:** device-to-device copy bandwidth measured in the same session.
  This is a reference for interpretation, not a ceiling the GEMM must reach.

**Stage 2: baseline (H800).** For the four shapes in §3, M ∈ {1, 2, 3, 4, 5,
7, 8, 15, 16, 17, 24, 31, 32, 33, 40, 47, 48, 56, 63, 64}, with BF16 output, measured under
the §6 procedure. Optional reference: DeepGEMM at the same points, if it is
already importable.

**Stage 3: one candidate change (only if the §5 gap gate passes).** A dispatch
variant that uses the swapped small-tile configuration for every M ≤ 64. Among
the sweep points, it changes M ∈ {4, 8, 16, 24, 32, 40, 48, 56, 64}. Correctness (§7) must
pass before any variant timing is interpreted. Split-K, stream-K and new tile
shapes are out of scope.

## 5. Predictions and the gap gate (freeze before booking)

P1/P4 are prior-informed predictions from §2.4, not measurements on this
configuration. Direction is motivated; effect sizes and all-cell applicability
remain unknown.

- **P1 boundary:** for at least one relevant shape, t(64) − t(63) exceeds the
  §6 decision bound. The expected direction is t(64) > t(63).
- **P2 control (descriptive, no pass/fail):** report t(15), t(16) and t(17) for
  each shape, with no causal claim from tile counts.
- **P3 cache:** for each shape at M ∈ {1, 16, 64}, the hot contrast is faster
  than the rotating run by more than the decision bound. Activation
  quantization is excluded from both.
- **P4 variant (Stage 3 only):** at the changed points of shapes that pass the
  gap gate, test whether the variant is faster than base by more than the bound.
  Check regressions at every changed point of all four shapes, including shapes
  that did not pass P1. No changed point may regress beyond the bound; all
  unchanged controls must stay within the bound. Report improvement, equivalence,
  regression and insufficient evidence separately; equivalence is not a speedup.

**Gap gate for Stage 3:** P1 holds on a relevant shape, reproduced in at least
`q` of `b` paired blocks, and exceeds a minimum effect of `e_rel` relative and
`e_abs` absolute. The values of `q`, `b`, `e_rel` and `e_abs` are fixed at
freeze. Roofline or copy-bandwidth ratios support interpretation only; they
cannot pass or fail the gate.

**No gap:** if the gate fails, record the baseline and stop this experiment.
That is a completed result.

## 6. Measurement procedure (freeze before booking)

- **Fixed counts:** warm-up `W`, calls per graph `R`, replays per block `n`,
  blocks `b`.
- **Paired, interleaved arms:** base and variant are built with distinct
  registration namespaces and loaded in one process. Within each block they are
  interleaved in ABBA order; the pair order alternates across blocks.
- **A-A calibration:** base against base, in the same interleaved positions,
  before any cross-point or cross-arm comparison.
- **Precommitted decision bound:** `max(k · s_AA, e_abs, e_rel · t_base)`, where
  `s_AA` is a robust spread of the A-A block differences (a scaled median
  absolute deviation) and `k` is fixed at freeze. Computing it after
  calibration is allowed; changing the formula is not.
- **Clock and thermal rejection:** record SM and memory clocks, temperature,
  power and throttle reasons, including hot-state samples during each block.
  Apply the fixed-power exception and rejection rules in §11.2. Reject a block
  if its hot-state SM clock deviates from the session median by more than the
  frozen percentage. Rejected blocks count against `q`; they are not replaced.
- **Repeats:** all planned repeats run. No selective reruns after inspecting
  results, no threshold changes and no added points.
- **Equal arms:** diff the base and variant `build.ninja` and compile commands.
  Only the registration namespace and the dispatch source may differ. Both arms
  share harness code, allocation pattern and synchronization. The R2 builder
  has a separate performance successor with `--namespace` and CPU tests;
  historical R2 is untouched. GPU compilation and two-library loading are still
  unverified. Both arms use `-Wl,-Bsymbolic` to bind internal ELF symbols locally;
  verify dispatch identity for each loaded library before timing, since distinct
  registration names alone do not prove different kernel implementations ran.
- **Nsight Compute:** probe counter permission at session start. Without it,
  timing and trace results stand and counter items are unscored.

**Dual-library witness before timing:** at identical valid inputs with M64,
trace the base cooperative/non-swapped kernel and variant ping-pong/swapped
kernel. If symbols cannot be attributed to each namespace or both arms launch
the same path, stop as unscored; different registration names alone are not
evidence of isolation.

## 7. Correctness coverage for the variant

The historical 30-case suite supplies reference cases, not a reusable fix-arm
verdict: performance arms do not include the B1/B2 repair guards, so their old
`raises` expectations cannot be applied unchanged. Preserve the original suite
and add a separate performance-variant suite, for all four §3 shapes:
- the changed dispatch points M ∈ {4, 8, 16, 24, 32, 40, 48, 56, 64};
- adjacent unchanged swapped controls M ∈ {3, 5, 15, 17, 31, 33, 63};
- numerical comparison against a float dequantized reference, with the frozen
  relative-error bound (0.005);
- CUDA-graph capture and replay at M ∈ {4, 64};
- FP16 output at (N,K) = (4096,4096), for all 16 correctness M values.

Both arms must pass all new numerical and capture checks. The mandatory matrix
is 64 BF16 eager + 8 BF16 graph + 16 FP16 eager cases per arm (88 per arm,
176 arm-cases), with independent output references. No variant performance is
interpreted if any mandatory case fails or is unscored.

## 8. Budget and sessions

Total **18 working hours**, including preparation and builds. One H800 rental
contains three bounded stages; completing all three is conditional:

- **Before booking:** finish local tools, source variant, wheel archive,
  installation/download plan, CPU checks and public freeze SHA verification.
  Do not require releasing a scarce H800 just for preparation. The model and
  remote install may be prepared in same-card stage P. A no-GPU environment
  remains optional for downloads, not CUTLASS compilation with 2 GB RAM.
- **Preparation P (at most 60 minutes, including failure sealing):** check
  network/storage, download the pinned model, install the explicit parent
  wheel into an isolated environment and verify dependencies/source/extensions.
  Retain model and `serving_preflight.py` receipts. On failure or timeout, do
  not enter A; seal and shut down. This is not performance measurement; do
  not change pins, model, candidate, parameters or effect thresholds to pass.
- **Session A (at most 100 minutes):** identity and Nsight probe 10, base build
  20, Stage 0 witness 25, marked dispatch probe/export/check 5,
  Stage 2 baseline with A-A calibration 30, seal and
  power-off 10.
- **Session B (at most 70 minutes, only if the gate passes):** identity 10,
  variant build 20, correctness 10, paired timing 20, seal 10.

The single-rental cap is P60+A100+B70=230 minutes. If P or A stops, seal and
shut down immediately rather than consuming the remaining allowance. Continue
B on the same card only after the complete A gap gate passes, without another
booking; a card change still obeys the identity conditions below. Earlier
sessions put two builds plus tests at about 42 minutes, not a guarantee for
these performance builds. On a stage overrun, remaining points are unscored.

Stage 0 priority is forced-CUTLASS Nsight, forced-CUTLASS capture shapes,
then optional default Nsight for context. No default shape-profile run.
The three launches share a 25-minute deadline; drop default context first,
never a mandatory forced witness. The 36x4-per-step attribution applies only
to forced CUTLASS, not to default FlashInfer/DeepGEMM launch patterns.

Session B may use a different physical UUID, recorded explicitly, but must
match GPU model, SM count, L2, NVIDIA driver, CUDA driver API, nvcc and Torch.
Its dispatch/correctness/timing witnesses must use the same B card. All B
comparisons and A-A calibration remain within B; the A gap gate may come from
a different card, which is a transfer limitation, not a cross-card speedup.

## 9. Conditions for starting validation

1. The duplicate-work check is complete: open vLLM PRs and issues on
   blockwise-FP8 small-M dispatch, `swap_ab` and SM90 tile configuration, plus
   CUTLASS upstream.
2. The workload and the target policy (§3) are fixed.
3. The corrected charter, harness, namespace-enabled builder, correctness tests
   and predictions are reviewed, committed and pushed.
4. The build and measurement plan fits the session budget.

Run P first, then A only if preparation passes, then B only if the complete
gap gate passes. Public freeze verification precedes P, not installation or
measurement followed by a retrospective commit.

## 10. Constraints carried over

Raw logs, traces, hostnames, paths and profiler reports stay private. Publish
only configuration, counts, summary tables and digests. There is no new upstream
issue and no fourth vLLM PR under the current decisions. Any later PR follows
vLLM AGENTS.md. Formal adoption goes to the Oct 26–Nov 1 review.

## 11. Pre-freeze completion record (2026-09-30; no GPU data)

### 11.1 Proposed constants, fixed by the next public freeze

| Parameter | Value | Meaning |
| --- | ---: | --- |
| W | 5 | Warm-up graph replays per timed position |
| R | 8B | Eight complete weight-set cycles; hot contrast uses the same R |
| n | 9 | Timed replays per position, each divided by R |
| b | 7 | Planned paired blocks, never replaced |
| q | 6 | Minimum valid blocks exceeding the bound for a positive claim |
| k | 3 | Noise multiplier on scaled MAD (1.4826) |
| e_rel | 0.03 | Minimum relative effect |
| e_abs | 0.5 microseconds | Minimum absolute effect per GEMM |
| Clock deviation | 10% | Hot-state SM clock deviation from session median |

These are conservative decision constants, not measured sensitivity or a
statistical confidence level. They remain reviewable until the public freeze.
If calibration or measurement is incomplete, record insufficient evidence,
not "no gap". Complete all seven planned blocks even after six match.

### 11.2 Session A pair definitions and calibration

For every (shape,M,cache condition), run A-A calibration using two position
labels for the same base graph. Each block has four positions: ABBA in even
blocks and BAAB in odd blocks. Each position has W warm-ups then n timed
replays. Each side contributes 2n values; its block value is their median.

Then compare M64 versus M63 for each shape in the same order, with
`difference = t64 - t63` and reference `t63`. Calibrate both cells separately;
use the larger of their two decision bounds. Hot/rotating comparisons likewise
use paired positions, `difference = t_rotating - t_hot`, with rotating as the
reference. A-A calibration covers the 80 rotating cells and 12 hot cells.
The broad sweep baseline comes from their calibration measurements, not an
unpaired comparison against a separately collected timing sweep.

For each cell, compute s_AA = 1.4826·MAD of valid A-A block differences about
their median. Require at least q valid calibration blocks. If absolute median
A-A bias exceeds max(e_abs,e_rel·t_reference), calibration fails for that cell.
Otherwise the bound is max(k·s_AA,e_abs,e_rel·t_reference). A positive comparison
requires at least q of the original b blocks individually above the bound and
a median valid difference above it. Preserve every raw replay value privately.

For Session B use the same definition with `difference = t_base - t_variant`
and reference base. Perform a fresh A-A calibration in that session; never
subtract Session A timings from Session B timings. Check all 36 changed cells
and all 44 unchanged cells. Reject an improvement claim if a changed cell has
a median regression beyond its bound, or if an unchanged cell's median absolute
difference exceeds the bound. At least q valid blocks are needed for each
mandatory regression check; otherwise the outcome is insufficient evidence.

Monitor clocks during hot timed blocks, not idle before/after values alone.
Record both raw endpoints and hot-state samples. Fixed-power software capping
is allowed if identical throughout and clocks pass the deviation check;
thermal throttling, hardware slowdown, power-limit changes, or missing clock
samples invalidate the block. Do not replace rejected blocks.

### 11.3 Budget arithmetic and preparation status

A replay includes R kernel calls. For each comparison/calibration cell the
fixed workload is b·4·(W+n) graph replays. Session A has 92 A-A cells plus
4 boundary pairs and 12 cache pairs: 108·7·4·14 = 42,336 replays.
Session B has 80 rotating A-A cells plus 80 base/variant pairs:
160·7·4·14 = 62,720 replays, plus the 176 correctness arm-cases.
This is a workload count, not a measured runtime estimate. GPU durations,
graph capture, model startup and builds have not been timed for this packet.
Using #44572's 17–60 μs at different shapes as a rough scale, R=8B can put a
replay around 0.5–1.5 ms: about 21–64 seconds of kernel time for A and 31–94
seconds for B. This is not a host/session forecast; builds, serving, capture,
synchronization and tracing dominate the risk.
Use the first three fixed-order cells only to project remaining time; do not
change counts or thresholds. An overrun leaves remaining cells unscored.
The session caps remain A=100 minutes and B=70 minutes within 18 work hours.

CPU preparation added an isolated performance
[builder](../../experiments/kernel-operand-contracts/sm90-block-fp8-perf/build_perf.py),
[protocol definitions](../../experiments/kernel-operand-contracts/sm90-block-fp8-perf/protocol.py)
and tests. No historical R2, repair patch, receipt or frozen manifest changed.
The namespace tests mock compilation: they prove argument/registration
separation, not successful nvcc compilation, linking or different GPU execution.

Draft CUDA timing, native serving collection, reviewed shape/graph witnesses
and the 176-arm-case suite are now implemented and CPU-tested. See the
[runbook](../../experiments/kernel-operand-contracts/sm90-block-fp8-perf/RUNBOOK.zh-CN.md).
This does not establish GPU or profiler compatibility.

**Still blocks freeze/booking:** review these executable tools;
verify compiled registration/dispatch separation during runtime preflight;
download and bind the model config/weights; review the work-count budget and
complete the public manifest. No GPU run is authorized by this document edit.

The serving Python must match the pin. An explicitly recorded cu130 precompiled
wheel for parent `4f1451679088e5832bce0965a254295c854aaa09` is admitted only
for Stage 0: the pin's only C++ change is the reviewed CPU-side operand checks
in `scaled_mm_entry.cu`; these checks are absent in that serving binary.
This mixed Python/binary identity is not a full pin build or a K5 validation.
The minimal timing arms still build the pinned C++ and their inherited checks.
Before booking, retain the wheel URL, full commit, archive SHA-256, dependency
metadata, installed-extension hashes and install log. Require Torch
2.13.0+cu130; the preflight verifies every installed extension against the archive.
Explicitly select the retained wheel, never fall back to latest. The user's
cu130 index response advertises the parent x86-64 cp38/abi3 wheel, with a root
commit URL resolved from its relative path and a null variant. Bind the original
index/digest and resolve its entry rather than require `/cu130/` in the final
URL. The retained archive is now inspected: 315,907,592 bytes, SHA-256
`f50bf6c6c63785be641d8512139593ebeb20f7c1e9bf290dc7864de414909128`,
torch==2.13.0, Python >=3.10,<3.15, stable CUDA extension present. Its internal
tag is generic `cp38-abi3-linux_x86_64`, not the advertised manylinux tag;
do not infer a glibc floor. Two over-strict tool checks (exact manylinux tag
and legacy HIP-only `_C`) were corrected; both failure receipts are retained.
CUDA/ELF loading and installation remain unverified. If incompatible, stop before booking;
do not silently add a full source build. See the runbook for the exact index.

Read-only ELF inspection of the two main stable extensions found direct
`libcudart.so.13`/`libcuda.so.1` dependencies, with required GLIBC up to 2.14
and GLIBCXX up to 3.4.21. This is not all-wheel or transitive-dependency coverage.
Agent access to local WSL fails with E_ACCESSDENIED. The user prepared a Python
3.10 venv and confirmed the required Torch version is indexed, but a local full
install is no longer pursued. A subsequent target-host preflight found H800
SM90 with 114 SMs, 120 GiB of cgroup memory, Torch 2.13.0+cu130, nvcc 13.0.88
and Nsight Systems 2025.3.1.0. A basic Torch GPU addition succeeded; this is not
a vLLM extension or performance result. The existing serving dependencies fail
pip check, the pinned model is not downloaded, and direct Hugging Face access
timed out. Entering a paid host before confirming the public freeze was a
workflow error. No performance experiment ran; the host was shut down at the
user's request. See runbook section 7 for the remaining installation, storage,
model and public-freeze gates. No new GPU booking is approved.

Native Nsight and Torch shape profiling use separate fresh-server passes with
one shared 25-minute Stage 0 deadline. Capture-op to replay-node mapping is
reviewed evidence, not automatic cross-profiler proof. Missing input dimensions,
graph nodes or complete-step membership means insufficient evidence, not an
inferred M from concurrency. Establish this route before recommending booking.

Implementation constants: seed 730 and independent weight seed 10730 keep
B and B-scales identical across M; relative L2 correctness bound 0.005 with
TF32 disabled; graph replay checks zeroed A and restored A. NVML hot-window
sampling is every 5 ms; do not change power limits. Dispatch probes require
three marked launches per arm. Public byte-manifest verification precedes GPU
execution; raw traces and review mappings remain private. These details do not
change the fixed matrix, decision thresholds or session caps above.

### 11.4 Bounded duplicate search and historical follow-up

The initial open-only search missed #44572. This search-scope defect was
corrected before freeze by reading the merged/closed work in §2.4.
On 2026-09-30, searched GitHub issues/PRs with these exact queries:
`repo:vllm-project/vllm is:open "swap_ab"`,
`repo:vllm-project/vllm is:open "blockwise" "small"`,
`repo:NVIDIA/cutlass is:open "SM90" "small"`,
`repo:vllm-project/vllm is:open "SM90" "tile"`,
`repo:NVIDIA/cutlass is:open "blockwise" "FP8"`.
Inspected the current #56248 file diff: it honors operand strides but does not
change the small-M dispatch. #55537/#56480/#56659 cover layout checks/strides,
not this proposed variant. Related [#43214](https://github.com/vllm-project/vllm/pull/43214)
is a DO NOT MERGE low-latency blockwise FP8 CuTeDSL/PDL implementation with
serving measurements; it is related performance supply, not the same C++
dispatch change. CUTLASS [#2923](https://github.com/NVIDIA/cutlass/issues/2923)
reports block-FP8 latency concerns on B200, not a matching SM90 measurement.
CUTLASS [#3596](https://github.com/NVIDIA/cutlass/issues/3596) and its fix
[#3599](https://github.com/NVIDIA/cutlass/pull/3599) concern B-scale reuse
across successive M waves in a 256-row tile. At the pinned collective,
`NumSplitsM = TileM / 128` ([L208–209](https://github.com/NVIDIA/cutlass/blob/cb4247394dd82148787aed73e5dc7cef33cbf862/include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp#L208-L209));
both selected tiles have TileM=128, hence one M wave. This is source evidence
that this specific successive-wave defect does not match our two selected
tiles, not a general correctness guarantee or a GPU validation. The numerical
gate remains mandatory; do not apply the dependency fix to either arm.
Also queried `repo:vllm-project/vllm is:pr is:open author:yewentao256` and
checked all four returned file lists (#59084/#58845/#57443/#57053): none listed
the SM90 blockwise dispatcher. This does not exclude private work, renamed
authorship or subsequent pushes. No exact duplicate of this follow-up was
identified; the swap technique itself is prior work. Refresh thread heads before proposing an upstream change.
