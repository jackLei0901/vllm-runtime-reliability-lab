# SM90 block-FP8 operator contract repair: preliminary requirements

Status: 2026-09-29, v0.2, feasibility in progress. The [Chinese version](SM90_BLOCK_FP8_PROJECT_REQUIREMENTS.zh-CN.md) is authoritative; this is its translation. This proposes a low-level project whose implementation and validation the Lab owns. It is not a frozen execution protocol. Budget, model and measurement thresholds must be settled before their respective stages begin.

## 1. Problem to solve

vLLM's SM90 block-FP8 `cutlass_scaled_mm` path relies on input dtypes, the physical layout of scales and operand strides. The Lab has observed that some inputs inconsistent with the kernel's interpretation pass the entry checks and produce substantially incorrect or non-finite outputs. The standard branch of the same operator checks some of these inputs already.

The required boundary is executable: supported inputs produce correct results; other inputs receive explicit handling or an error. Supported inputs must be explained through callers, the operator entry and kernel constraints. Negative-control results alone cannot decide whether a layout should be rejected or supported.

The initial local policy is **reject**: add checks for SM90 block-FP8 operand dtypes (B2) and scale layouts (B1). Invalid inputs must be rejected, without implicit copying or conversion. This is a local contract to validate, not an upstream-approved policy. Supporting another scale layout requires a separate scope and cost decision.

The local branch is proposed to build on a pinned #55537 head (checked on 2026-09-29: `7b054aca96cea8be1369d651c3434ad140580b92`), inheriting its operand-stride rejection and alignment checks. B0 is inherited regression coverage only: both arms retain #55537; no new B0 fix is written. Parallel approaches are reject in [#55537](https://github.com/vllm-project/vllm/pull/55537), versus honoring strides in [#56248](https://github.com/vllm-project/vllm/pull/56248) (SM90), [#56480](https://github.com/vllm-project/vllm/pull/56480) (SM100), and [#56659](https://github.com/vllm-project/vllm/pull/56659) (SM120). Feasibility must refresh them and check entry-point conflicts. If #55537 cannot serve as the base, revise scope or exclude B0 before proceeding; do not silently switch policies.

## 2. Existing evidence and value limits

| Evidence | Established facts | Facts not established |
| --- | --- | --- |
| [B1: eight H800 cases](../../experiments/kernel-operand-contracts/B1_H800_RESULT_2026-09-29.md) | Required-layout controls were correct; scales in incorrect physical layouts were accepted and substantially changed output on the tested path; equivalent-layout controls were correct | Normal serving produces these layouts; a newly built current upstream binary behaves identically |
| [B2/B0: sixteen H800 cases](../../experiments/kernel-operand-contracts/B2_B0_H800_RESULT_2026-09-29.md) | The blockwise branch accepted e5m2/int8 and produced incorrect or non-finite output; the standard branch rejected the tested e5m2 input; padded views reproduced the known problem | Behavior for all dtypes/shapes; a rejection policy is the upstream API contract |
| [Source inventory](QUANT_GEMM_OPERAND_CONTRACT_INVENTORY_2026-09-29.md) | Constraints and source locations for inspected entry, helper and dispatcher paths | An inventory does not prove a repair, the executed caller path or external demand |

These runs used one H800 and the recorded extension. Relevant source-file identities were checked, but the original binary's complete build provenance was not independently established. Historical results retain those limits. New repair validation requires traceable base/fix builds. Inspected normal vLLM callers supply the required inputs, so this is currently a latent operator-boundary risk, not a demonstrated production failure.

Three kinds of value are assessed separately:

- **Technical value:** replace observed silently incorrect input behavior with explicit, regression-tested behavior, reducing incorrect-result risks when callers change or switch backends.
- **External value:** consumers remain unconfirmed, and the upstream route is currently blocked by existing constraints. The natural edit overlaps #55537, which cannot be widened without authorization; a fourth vLLM PR is also ruled out. The planned exit is a local branch and reproducible material, with at most a short #55534 comment subject to existing publication timing and authorization. This draft does not authorize posting. External use is expected to remain `not_observed` unless a usable submission route emerges after #55537 merges or a maintainer requests the fix; neither condition alone proves adoption.
- **Project and learning value:** complete the chain through callers, the PyTorch custom op, C++ dispatch, CUDA kernel, tests and normal serving validation. The user participates in source reasoning, builds and profiler operations. This capability outcome does not substitute for external-use evidence.

## 3. Deliverables and requirements

The Lab owns this chain: define the contract → write C++ → build base/fix → check the input matrix → validate valid-caller compatibility and eager host overhead → retain reproducible material. The main acceptance rationale is bounded project and learning value, not confirmed external demand. The patch may be about ten lines of checks; the skill gain is in builds, dispatch and validation, not performance optimization.

| ID | Requirement | Acceptance |
| --- | --- | --- |
| K1 Input contract | State dtypes, shapes, strides, devices and necessary alignment for A/B/output and activation/weight scales; identify whether constraints come from callers, the entry or the kernel | A source-pinned contract table covering ordinary and swapped dispatch; unresolved API choices have a reasoned local candidate policy |
| K2 C++ behavior | Reject invalid B1/B2 inputs without copying or conversion; errors identify the input and requirement | Each addition has a before-failing/after-passing test; choose the shared helper or SM90 dispatcher and declare architecture coverage |
| K3 Regression tests | Keep valid, wrong-dtype, wrong-scale-layout, equivalent-layout and swapped controls; B0 must be rejected in both arms | Freeze incremental B1/B2 base/fix predictions; reject invalid inputs in eager calls and graph capture, and pass valid capture/replay controls; do not claim replay repeats entry checks |
| K4 Build identity | Microtest base is pinned #55537; fix adds only B1/B2. Match toolchains/dependencies and record source, patch, commands and extension digests | First assess a minimal extension containing the real entry/dispatch/kernel; if infeasible, reassess full-build cost rather than testing copied checks |
| K5 Serving compatibility | Use only the full fix build with a real, natively configured block-FP8 model fitting H800 | Start and complete predefined requests/output-validity checks without rejecting valid callers; require runtime evidence of SM90 CUTLASS execution, otherwise unscored; no serving performance A/B |
| K6 Host overhead | For rejection, compare valid eager CPU submission cost and review the graph-check timing in source | Freeze method, repetitions and tolerance; avoid per-call synchronization conflating GPU time with check cost; report resolution limits, not zero overhead, when differences cannot be resolved |
| K7 Usable delivery | Provide patch, tests, build/run commands, summary results and limits; retain raw private material privately | A non-author can run or review it without requesting missing files; record delivery, independent use and maintainer acceptance separately |

Serving validation establishes that new checks do not reject valid callers; it does not prove serving previously triggered B1/B2. Disabling DeepGEMM (`VLLM_USE_DEEP_GEMM=0`) is only one routing condition, not evidence that CUTLASS executed. Select the backend and retain a runtime path witness.

K3's B0 rejection comes from #55537's entry checks, but its existing `test_cutlass_c3x_rejects_padded_operand` uses a single scale and covers only the standard path. Blockwise rejection is currently a source prediction; this project's B0 would directly test it for the first time. Identical arm predictions do not imply existing runtime coverage.

CPU entry checks execute during CUDA-graph capture; replay skips that Python/C++ dispatch. They cannot detect invalid input state arising only on replay. Captured tensor metadata is fixed; replacing dtype/stride/shape normally requires a new call or capture, rather than arbitrary new tensors passed into replay. This project adds no value-domain checks and does not detect subsequent buffer-content changes. See [PyTorch CUDA-graph documentation](https://docs.pytorch.org/docs/main/notes/cuda.html#cuda-graphs). Calls outside captured regions retain eager check cost.

The current SM90 dispatcher selects scale types/layout configuration through templates and constructs runtime layouts from shape, not scale strides; see the pinned [ScaleConfig source](https://github.com/vllm-project/vllm/blob/91d7324cb19d301c72d849e457221ee8dd645024/csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm90_fp8_dispatch.cuh). Another layout may require rearrangement/copying or a different kernel instance. Its design and compilation/binary costs require separate verification, not acceptance under a few-checks budget.

## 4. Relationship to the existing Lab

This draft proposes adding delivery of an owned low-level code change to the Lab's goals. Current [Lab requirements](../LAB_REQUIREMENTS.md) R1 and the external-delivery target restrict work to others' runtime/lifecycle threads. This project permits an owned kernel repair, so formal adoption must explicitly amend those scopes and targets. It does not count toward the old target.

Carry over source/build pins, positive and negative controls, explicit unscored outcomes, privacy rules and publication checks. Existing progress, termination-propagation and shutdown-budget checks remain available; any maintenance-mode decision is recorded separately, without changing historical results. Existing experiments and the source inventory can be reused. No new general framework, classifier or profiler is required.

Upstream activities continue under existing publication and contact constraints. Refresh related issues/PRs and duplicate-work status at review time. Maintainer silence does not prevent bounded local implementation, but a candidate API policy cannot be described as upstream-approved, and #55537 cannot be widened without authorization.

## 5. Gates, budget and outcomes

Proposed total budget: **20 working hours**, including design, source work, builds, tests, documentation and analysis. Proposed paid H800 time: **one session, at most two hours**. These budgets require confirmation and do not authorize booking. First spend at most **four hours** on source and implementation feasibility: confirm B1/B2 still apply, select a candidate contract, explain normal-caller compatibility, identify builds and existing tests, and assess overlap with current PRs. Only after this gate passes should implementation and the run protocol proceed.

Before booking H800, the patch and regression tests must be reviewable, microtest base/fix and full-fix builds/resource budgets feasible, the model and target backend selected, and experimental rules committed. Build time counts toward the total budget; an expensive build cannot silently consume the short GPU session. K6 requires no hardware performance counters and adds no profiler-training or counter-permission task. Define the K5 path-witness method beforehand.

Record acceptance items separately rather than using an ambiguous overall success label:

- **Technical delivery complete:** K1–K7 pass, producing a rebuildable C++ repair, regression tests and normal-serving compatibility results. Upstream merge status is recorded separately.
- **Partial completion:** a reproducible patch/test exists, but serving-path, performance-compatibility or build-identity gaps remain. List failed or unscored requirements; do not claim end-to-end completion.
- **Stop:** feasibility shows an existing fix already covers the work, a valid contract or build cannot be established within budget, or the candidate cannot meet numerical and compatibility requirements. Preserve the reason; do not extend the project by adding shapes, changing kernel families or designing another framework.

At the four-hour gate explicitly decide whether about ten lines of checks with no upstream route still justify up to sixteen remaining hours for learning. Record continue/stop reasons, not sunk cost. Review again before GPU booking. At the budget limit, report completed items and gaps; further investment needs a concrete additional requirement and budget. Independent use, maintainer adoption and reduced real failures are external outcomes. If absent, record `not_observed`; local test passes do not substitute for them.

## 6. Items required before freeze

1. Refresh the four PRs and source; check entry conflicts and choose B1/B2 placement. Shared `scaled_mm_helper.hpp` covers SM90/SM100/SM120: do not add unconditional SM90-specific layout restrictions. Changes to other architectures require broader validation and renewed budget approval; otherwise isolate to SM90.
2. Design patch/tests and two build plans: matched microtest base/fix, and a full fix build for K5. Earlier B1/B2 used an existing vLLM extension; the minimal-extension route is unverified. It must compile the real path, isolate registration names and correspond to the full patch. Build the full fix off H800 on a GPU-less or 4090 host after checking RAM, disk, CUDA toolchain and dependencies. Treat `TORCH_CUDA_ARCH_LIST=9.0a` as a candidate configuration and verify effective CMake/nvcc commands generate SM90a, not just the environment variable. Builds still count toward twenty hours.
3. Select model, requests and actual-backend witness; freeze K5 output-validity rules and K6 eager host measurements. Do not restore serving performance A/B. Performance analysis needs a separate scope, not an addition to this project.
4. Explicitly accept learning as the current primary motive, the blocked upstream route and amendments to the old Lab scope. Reassess the value of remaining validation after four hours.

The first action for this review draft is item 1, the source and contract feasibility check. Decide whether to freeze it as an execution project after that check completes.

### Initial feasibility record (2026-09-29; gate not yet passed)

- GitHub refresh: all four PRs are open; #55537 matches the pin above. Heads for #56248, #56480 and #56659 are respectively `107d22b8a85024435296396485de9b6dbd457ead`, `f668271bf39603d1eee6d25fe85f062f8c4a8b0e`, and `b89d84a94cf54984ff42df69f0ce6bd242a5af0f`. This is a dated observation, not a guarantee of future state.
- Initial placement: `cutlass_scaled_mm_blockwise_sm90_fp8` in `c3x/scaled_mm_blockwise_sm90_fp8.cu`, checking A/B e4m3 dtype and scale layouts before output-dtype dispatch. This SM90-only entry covers both output dtypes and swapped dispatch, without editing the shared helper or #55537 entry lines. Inspect each parallel PR diff before claiming absence of conflicts; different files alone are insufficient.
- B1 checks must allow physically equivalent size-one dimensions, not mechanically fix every singleton stride. Assess padded/TMA-aligned scales against the kernel's packed scale addressing case by case; activation-scale validation cannot simply use `is_contiguous()`.
- GitHub changed-file checks confirm that none of the three parallel PRs edits the planned SM90 blockwise `.cu`; #56248 edits its downstream dispatcher, while the others edit SM100/SM120 dispatchers. All four PRs edit the shared test file, so initially place new B1/B2 tests in a separate file. This establishes file-level absence of textual overlap, not merge/build-tested semantic compatibility. If #56248 rather than #55537 becomes the actual base, B0 should be honored and correct; the local rejection expectation cannot be reused.
- Initial caller audit: pinned `CutlassFp8BlockScaledMMKernel` sets `column_major_scales=True`, leaves default-false `tma_aligned_scales` disabled, and passes `Bs.T`. For nonempty 2-D scales, candidate A predicates are `(M<=1 || stride(0)==1)` and `(Kblocks<=1 || stride(1)==M)`; candidate B predicates are `(Kblocks<=1 || stride(0)==1)` and `(Nblocks<=1 || stride(1)==Kblocks)`: K-major with normal strides `(1,Kblocks)`. The earlier draft inverted B; comparison with original B1 and the caller corrected it before execution. These are not CUDA-tested; retain existing shape/dtype conditions. TMA padding exceeding M across multiple Kblocks does not match A's layout. Do not accept it merely because a quantization utility produced it, or claim a production trigger was found.
- No runnable CUDA build environment was found this round: Windows PATH lacks the tools, and WSL enumeration returns `E_ACCESSDENIED`. No old tunnel was accessed or instance started. Compilation remains incomplete; next confirm an existing Linux/CUDA build environment rather than booking H800 for this purpose.
- Saved the [#55537 base patch](../../experiments/kernel-operand-contracts/pr55537-base-7b054aca.patch), exported from the pinned head with `git format-patch -1 --stdout --no-signature`. File SHA-256: `1f3f7c463813ef9857c32db6df03b8c179acecbaabde3bc6ebe7bafd72074887`. Apply to parent `4f1451679088e5832bce0965a254295c854aaa09`. This preserves the change only; full source/parent and dependencies still need a reproducibility plan. A patch is not a source archive.
- Pinned CMake includes the SM90 blockwise `.cu` in the stable extension with target `9.0a`. No nvcc/cmake/ninja was found on the current Windows PATH. No build, remote access or booking was attempted; this does not establish the remote toolchain's availability.
- **Continue only build feasibility and contract refinement, not the remaining full validation.** Pending: parallel-diff conflicts, compilation of a real minimal extension, full-fix build resource estimates, valid-caller layout equivalence, and the K5 model. After resolving these or reaching four hours, make the explicit §5 continue/stop decision.

### H800 environment preflight supplement (no build or K3/K5 run)

The local candidate patch, minimal-build driver and independent tests are in the [preparation packet](../../experiments/kernel-operand-contracts/sm90-contract-repair/README.md). After the B-scale correction, eleven of twelve CPU checks pass; the actual CPU torch-tensor check skips because local torch is absent. Four existing index checks pass. Patch applicability and ruff are checked separately. None establishes C++ compilation or GPU validation; a skip is not a pass, and this document remains unfrozen.

After user confirmation of fingerprint and hostname, read-only checks through the authorized tunnel found one H800 PCIe with 81559 MiB VRAM, driver 595.71.05, capability `(9,0)`, and cgroup `memory.max=128849018880` (120 GiB). About 22 GiB remains on the system disk and 4.4 GiB on the 50 GiB data disk. Models occupy about 32 GiB; low space does not establish that they are disposable.

The image's torch is 2.12.1+cu130, but existing `vllm-fp8-venv` has 2.13.0+cu130, CMake 4.4.3, ninja 1.13.2 and stable Tensor headers. CUDA compiler `/usr/local/cuda-13.0/bin/nvcc` is version 13.0.88; GCC is 11.4. Default PATH lacks nvcc/ninja, and system CMake 3.22.1 is insufficient. A future build must select the matching tools explicitly, not trust the image label. The existing environment is a candidate without first reinstalling torch.

No environments, models or results were deleted. No minimal extension was compiled, and full-fix disk requirements remain unverified. Existing source checkouts are not the pinned #55537 and cannot generate this project's results. This round completed only environment preflight; builds and GPU validation remain incomplete. The user requested shutdown after recording. Issuing shutdown and confirming billing stopped are separate checks.


### Execution decision, 2026-09-30

The user selected the [H800 protocol v1](../../experiments/kernel-operand-contracts/sm90-contract-repair/SESSION_PROTOCOL.md): one 120-minute session, including preflight, two minimal builds, isolated tests and shutdown. AutoDL no-GPU and local WSL builds are withdrawn for this run. The user-provided WSL environment has 7.7 GiB RAM, 2 GiB swap and no nvcc/ninja/cmake on PATH or pip for python3; provisioning it is not selected. This supersedes earlier build-route proposals, not historical findings.

Only the preparation/build/regression stage is frozen by the associated local commit; public freeze requires a confirmed push before booking. K5/K6 and the four-hour continue/stop decision remain pending. CPU tests with missing dependencies must be completed without skips during preflight. No compilation, CUDA test or external adoption is claimed.
