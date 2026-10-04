# Pre GPU Decision for a Native CUDA Kernel Project

Status: 2026-10-03, America/Los_Angeles, v0.4 successor for user review and public freeze. The [Chinese version](NATIVE_KERNEL_PROFILE_DECISION_2026-10-03.zh-CN.md) is authoritative. The earlier freeze `c4fc7fca2fced24de6ef8e548d4f2cb04f4f6171` is preserved; its admission failed before any model startup or collection. The user requested repairs for a later run, not GPU activation now. This successor is not frozen until committed and pushed.

Current decision: installation and no-GPU checks passed. **Publish the freeze before booking**, then repeat host, disk, memory, GPU identity and idle checks. GPU runtime and serving remain untested. Incremental building is not a discovery prerequisite.

Execution approval addendum (2026-10-03): the user authorized one discovery session capped at 60 minutes and its preparation. Incremental-build proof moves to implementation admission. The supplied duplicate scan is non-exhaustive; incomplete targeted checks block implementation, not direction profiling. This exception does not approve another rental or all historical deadline amendments. Host identity was verified privately during preparation; reconnection and GPU admission checks remain necessary.

## 1 Problem and Intended Value

Select one handwritten CUDA kernel on a real model's default decode path and own its call-path analysis, root cause, native implementation, regression tests, operator measurements, and model validation. Lab retains reusable tests and evidence; the change is delivered in vLLM. No standalone engine, diagnostic recorder, or new framework is planned.

Career value comes from an explainable C++/CUDA implementation and engineering trade-offs, not experiment counts or a mandatory RFC. A real profile establishes reachability and cost, not automatically demand, fixable inefficiency, or maintainer adoption. A small change may remain an ordinary PR rather than carry the main portfolio project.

Complete the current #55537 update and planned review request first. Offline preparation need not wait for its merge, but new GPU work must not displace it. Close FP8 #59800 according to the user's merge confirmation. INT8 is excluded from this profile. The previous SM90 dispatch experiment remains [insufficient_evidence](SM90_BLOCK_FP8_PERF_RESULT_2026-10-01.md), without rescoring or rerunning.

## 2 Source Review Baseline

The GitHub main query returned `b0e21b308352587a6fd02f72722a2e815bfd62f0`, with a [commit](https://github.com/vllm-project/vllm/commit/b0e21b308352587a6fd02f72722a2e815bfd62f0) timestamp of 2026-10-03 06:28:02 UTC. This is a read-time snapshot, not a permanently current main or the identity of an installed wheel.

This review read config, QuantFP8, block-linear dispatch, FlashInfer/DeepGEMM branches, quantization helpers, FlashAttention cache updates, and CMake at that pin. No model or compilation ran. The older local checkout is not this pin and contains uncommitted changes; its working-tree state was not used as clean-source proof, and it was not modified.

## 3 Default Path Reachability and Model Selection

| Boundary | Source fact checked in this review | Selection implication and missing evidence |
| --- | --- | --- |
| Custom ops | With Inductor and compilation not NONE, none is appended if all/none is unspecified. Blocked weights append +quant_fp8 unless -quant_fp8 is present.[Config](https://github.com/vllm-project/vllm/blob/b0e21b308352587a6fd02f72722a2e815bfd62f0/vllm/config/vllm.py) | An option does not guarantee execution of a particular handwritten kernel; backend, fusion, and the actual graph still matter. |
| Block linear | CUDA candidates prioritize dynamic FlashInfer/DeepGEMM, then DeepGEMM, CUTLASS, and others. Availability and can_implement determine selection.[Dispatch](https://github.com/vllm-project/vllm/blob/b0e21b308352587a6fd02f72722a2e815bfd62f0/vllm/model_executor/kernels/linear/__init__.py) | Do not force CUTLASS, disable default fusion, or use eager mode to manufacture a target. |
| Dynamic backend | apply_input_quant=False. The dynamic function's small-M branch enters FlashInfer; the other branch explicitly calls per_token_group_quant_fp8 before DeepGEMM.[Dynamic kernel](https://github.com/vllm-project/vllm/blob/b0e21b308352587a6fd02f72722a2e815bfd62f0/vllm/model_executor/kernels/linear/scaled_mm/flashinfer.py) | Batch 1 and 32 may use different quantization producers. +quant_fp8 does not prove ordinary CUDA quantization runs at both. |
| Ordinary and packed quantization | QuantFP8 has ordinary and packed branches. The ordinary helper enters _C.per_token_group_fp8_quant on CUDA with contiguous input; the packed helper calls _C.per_token_group_fp8_quant_packed.[QuantFP8](https://github.com/vllm-project/vllm/blob/b0e21b308352587a6fd02f72722a2e815bfd62f0/vllm/model_executor/layers/quantization/input_quant_fp8.py), [helpers](https://github.com/vllm-project/vllm/blob/b0e21b308352587a6fd02f72722a2e815bfd62f0/vllm/model_executor/layers/quantization/utils/fp8_utils.py) | Actual M, layout, scale format, operator, and fusion determine the implementation. Require execution evidence and separate ordinary/packed duplicate checks. |
| KV writes | FlashAttention's do_kv_cache_update calls reshape_and_cache_flash; CMake includes native cache sources.[Backend](https://github.com/vllm-project/vllm/blob/b0e21b308352587a6fd02f72722a2e815bfd62f0/vllm/v1/attention/backends/flash_attn.py) | Conditional reachability only. Final attention backend, fusion, and symbol-to-source binding remain to be verified. |
| Norm and sampling | RMSNorm native enters IR; native sampler/topk files are present in CMake. | File presence does not establish execution. IR priorities, fusion, V2 sampling, and greedy settings have not been fully traced, so these are not witnessed candidates. |

The selected model is `Qwen/Qwen3-8B-FP8` @ `220b46e3b2180893580a4454f21f22d3ebb187d3`, TP=1, BF16 activations, dynamic activation quantization, blocks [128,128], hidden=4096, intermediate=12288, and 36 layers. The cached config was independently read and hashed: `79e454d6cc36c41ab0e9418c8f78a033afa9ac88e5a58d67c1471de315c8d29a`. Both weight shards passed header-length checks and SHA-256 comparison with their HF cache blob names; remote LFS metadata was not independently fetched. Memory fit remains a GPU admission check. Do not download a second model or expand into MoE.

The supplementary review predicts KV write at small M, with group quant added at M>=32. Independent source checking confirms the ordinary dynamic threshold `input.shape[0] < 32`, but true `VLLM_BATCH_INVARIANT` sends all sizes to the DeepGEMM branch. Record that setting and classify each step by actual padded M; concurrency 32 does not guarantee the M>=32 branch on every step. Priority and installed dependencies predict selection, not its witness on an installed wheel. The 36 KV writes and up to 144 group quant calls are structural expectations for a complete dense decode step, not measured counts in each window.

Do not yet call this an exhaustive two-kernel set: norm/IR and fusion exclusions remain incomplete. Read-only whole-trace aggregation of the older forced-CUTLASS SQLite contains handwritten CUDA group quant, KV write, RMSNorm+quant, and SiLU+quant symbols. This is neither default-backend execution evidence nor a concurrency-specific stable decode sample; it establishes that the old trace cannot directly validate the new exhaustive set.

The frozen prediction is KV write at small M, plus ordinary group quantization when the dynamic backend executes M>=32. Norm/activation fusion may add native kernels; that set is deliberately not claimed exhaustive. Exclude GEMM, attention, external-library and generated Triton kernels from candidates, but retain them in the source-distribution report. The FlashAttention cache-update call chain provides a potentially eligible native path; its execution and remaining headroom are questions for this discovery, not established findings.

## 4 Duplicate Work and Demand

Merged precedents are [#56478](https://github.com/vllm-project/vllm/pull/56478), [#55330](https://github.com/vllm-project/vllm/pull/55330), and [#58194](https://github.com/vllm-project/vllm/pull/58194). They establish prior work, not a remaining gap. [#47334](https://github.com/vllm-project/vllm/issues/47334) proposes CuTeDSL migration; it does not establish that all handwritten CUDA is retiring or that no one is rewriting a target.

The supplied review subsequently scanned changed-file paths for 674 unique open PRs using `gh pr list --state open --limit 200 --search "<term> in:title" --json number,title,files`, for quant/kernel/cache/fp8. Three queries reached the 200-result cap, so this is not exhaustive or freshly rechecked here. Relevant work: #51939 moves quantization outside the opaque DeepGEMM op; #42597 prototypes all-reduce/norm/quant fusion; #49928 retains quantization inside that boundary for TMA scales; #48653 touches the group-quant kernel; #59289 changes MLA cache writing, not reshape_and_cache_flash. The earlier title-only no-match statement is superseded. Complete targeted duplicate exclusion remains an implementation gate.

When access resumes, run `gh search prs --repo vllm-project/vllm --state open --limit 100 "<term>"` for quantization, norm, cache, and sampling, also search closed/merged work, and check truncation. For each prospective target, search paths, symbols, and related issues; read candidate diffs, not just titles. Check ordinary and packed quantization separately. Exclude covered candidates; silence on a PR does not release its work for ownership. Retain queries, dates, relevant heads, overlap, and unresolved exclusions.

Each candidate gets at most one page: workload and user scenario, symptom, default call path, existing report or its absence, change hypothesis, numerical contract, duplicates, and upstream route. CODEOWNER is routing, not a consenting consumer.

## 5 Binary Identity and Build Route

The official wheel for `b0e21b3` is installed in an isolated environment: version `0.30.1rc1.dev618+gb0e21b308`, SHA-256 `2282e931a96725cbe8a20f0f7ccc77e25406fff8eee5132f75f80ddbbc503780`, from the fixed-commit index at `https://wheels.vllm.ai/b0e21b308352587a6fd02f72722a2e815bfd62f0/`. All 5,305 installed wheel members match; all 2,585 compared in-tree Python files match the pinned source archive. Another 225 wheel Python files are generated or bundled third-party files absent from that archive, bound by the wheel hash rather than claimed source-identical. This does not independently prove native build provenance. Record loaded extension hashes during GPU admission.

The dependency check passes for 199 packages. Torch is `2.13.0+cu130`; FlashInfer Python and the complete cubin package are `0.7.0.post1`; Nsight Systems is `2025.3.1.0`. CPU Torch/vLLM imports pass without initializing CUDA. Failed download/install attempts are retained privately; final mirror downloads matched canonical package hashes. No GPU run or source build occurred.

Prefer compatible prebuilt external dependencies and an incremental target build, but this route has not been trial-built. [CMake at the pin](https://github.com/vllm-project/vllm/blob/b0e21b308352587a6fd02f72722a2e815bfd62f0/CMakeLists.txt) places most CUDA sources in _C_stable_libtorch, not a simple one-file _C rebuild, and includes multiple external builds. The previous minimal CUTLASS extension does not establish cheap builds for arbitrary native kernels.

Before implementation admission, use an adequate Linux toolchain to demonstrate a loadable baseline target and an incremental rebuild after a reversible identity change in the selected source. Record elapsed time, memory, disk, and proof that the new binary loads. Do not edit historical checkouts; an identity change is not an optimization or correctness result. Without adequate CPU/RAM, record an implementation build blocker rather than improvise a full build during profiling. If only paid GPU hosts are available, propose a separately budgeted build preflight; this document does not authorize it.

## 6 Bounded Discovery and Benefit Assessment

The user approved one 60-minute discovery session; the settings below await public freeze. Count documentation, downloads, queries and tool repairs rather than resetting historical costs. The 4-hour offline cap was not reconciled, and the extended preparation is disclosed as a budget-accounting deviation, not claimed compliant. This approval does not fund implementation.

Propose one discovery session capped at 60 minutes including identity, startup, compile caches, two-load profiling, sealing, and cleanup. No implementation session is included. Use retained startup records and offline preparation to set allocations and minimum remaining time for starting a collector; do not freeze before showing the work fits. At the deadline preserve evidence and clean up. Supporting-evidence failure does not permit unlimited collection or a model/backend switch.

Freeze concurrency 1 and 32, 128 input token IDs/64 output tokens, greedy with ignore_eos, TP=1, BF16, max model length 2048, max sequences 32, prefix caching disabled, and no speculation. Keep default backend and runner. Run two warmups and one unprofiled reference batch per load, then two profiled windows per load. Each window must contain at least 16 unambiguous complete decode-only steps after excluding startup/prefill. Missing or ambiguous step attribution means insufficient_evidence; retain partial data without replacement windows. Record token counts and infer padding only from resolved config plus execution evidence; concurrency is not executed M.

Use Nsight Systems for symbols, counts, step windows and CPU/GPU overlap. Export and attribute traces offline after shutdown; no additional Nsight Compute collection is authorized here. The unprofiled HTTP batch duration is only a workload reference, not unprofiled step time or TPOT. Do not use it as the denominator of the model-benefit formula below. If a valid step denominator is unavailable, report qualitative headroom and defer quantitative implementation admission.

Exclude GEMM, attention, external libraries, and generated Triton. Approximately 2% of summed GPU time is a ranking hint, not implementation admission. Inspect at most three candidates and select one. Sum relevant invocations per step, separate overlap and critical-path contribution, and bind symbols to csrc files.

Exclude external libraries from candidates, not reporting. Retain implementation-source buckets for vLLM csrc, Inductor-generated Triton, FlashInfer, DeepGEMM, FlashAttention, other attributed work, and unknown. A name prefix is not sufficient attribution: FlashInfer may invoke DeepGEMM. Shared symbols require a joint label or unknown, never double counting. Report summed-kernel shares separately from step wall time and overlap; source shares are not removable savings or end-to-end speedups.

Report optimistic performance headroom separately from savings supported by a specific change. Use bytes/BW only when a bandwidth model applies; consider HBM/L2 traffic, compute, launch, and dependency latency. Large-buffer copy bandwidth is a reference, not a small-kernel target.[NVIDIA triage guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/)

With no overlap and an unchanged critical path, `estimated model fraction saved = predicted savings across target invocations per step / unprofiled step wall time`. State the denominator; step time is not HTTP TPOT. With overlap this is optimistic only. The proposed 1.5% portfolio admission bar is an engineering choice, not statistical significance or an observed gain. A theoretical ceiling above it is insufficient: identify removable work in source. Operator and model A/B remain necessary for an effect claim; smaller changes may become ordinary PRs.

## 7 Decisions and Stop Rules

| Stage | Passing requirement | Otherwise |
| --- | --- | --- |
| GPU admission | Verified model/commit/config/files and matching installation; nonempty expected set; duplicate scan and truncation disclosed; usable tools; approved session budget | Missing install/resource evidence means defer; do not rent for an incremental-build proof. |
| Valid profile | Both loads provide attributable native execution and stable windows sufficient for cost and critical-path assessment | Apparatus or witness failure is insufficient_evidence, not no_gap; no automatic extra session. |
| Implementation admission | One candidate passing complete targeted duplicate checks, with source-level cause, numerical contract, substantive C++/CUDA change, credible benefit estimate, demonstrated incremental build, and separately approved budget | A valid profile with no qualifying candidate is completed_no_candidate. Retain source distribution for direction selection, without automatically starting another project. |

Propose an offline early-stop rule: if the candidate set is exhaustive and every candidate's conservatively estimated optimistic benefit ceiling is below the portfolio bar, using comparable configuration, stable decode windows, and a stated denominator, close discovery on this model without booking as an offline decision not to admit it. This is not measured no_gap or completed_no_candidate. Whole-trace forced-CUTLASS averages, sub-1-MB inputs, or a launch-dominated hypothesis alone cannot justify that stop; check layout, fusion, binary/pin differences, and critical-path relevance. Limit offline attribution to reusing old evidence, without a general trace framework. Insufficient evidence means defer.

Only discovery is proposed for funding. Changes, model A/B, and GPU rental are not automatically authorized. Propose implementation costs after selecting a candidate. Without one, maintain Lab and existing contributions rather than immediately searching another model.

## 8 Proposed Policy Amendment and Protected Work

[LAB_REQUIREMENTS](../LAB_REQUIREMENTS.md) still labels itself a local unapproved/unpublished proposal. Its 10-12/11-01/11-30 constraints informed previous conversations, but formal approval history was not verified this turn. This document does not silently supersede them.

Propose a bounded kernel-discovery entry: a representative default workload, source reachability, duplicate checks, and build feasibility replace a committed incident consumer only for **discovery** admission. Profile plus a source-level cause must pass before implementation admission. Record external demand separately; without a report, call this workload-driven optimization, not demonstrated customer demand.

Delivery means submitted, runnable patch/tests/evidence. Adoption means merge or an explicit decision to use the work. A rejection is a review outcome, not adoption; one's own PR cannot retroactively satisfy three artifacts in other authors' runtime threads.

Propose changing 10-12 to approval/rejection of one bounded kernel decision page and budget. Without a qualifying proposal, no new construction starts and the 11-01 maintenance condition remains. Propose that the 11-30 three-runtime-thread target no longer apply to new kernel work; do not replace it with mandatory merges or positive results. Changes require explicit dated user approval. Until then prior constraints remain, without automatic extensions from writing this page. The #55700 10-12 close-out rule is unchanged.

Protect planned #52178 and PyTorch #197232 follow-ups, #55537 update/review, the existing working tree, and SUBMIT.ps1 encoding housekeeping. Refresh their live states and dates separately; this task does not change, push, or message them.

## 9 Remaining Actions Before Freeze

1. User reviews the two decision files, collector, preflight/admission tools and CPU tests; commits these explicit paths and pushes. Before booking, run `preflight.py prepare` to compare all six public files and persist a private receipt. Successful per-file comparisons survive a later download failure; retries belong to unpaid preparation. Use a new receipt for a changed freeze, not an edited old one.
2. Reconnect to the prepared environment. Installation is complete; do not reinstall or download another model. Recheck disk headroom before booking: last preparation receipts report only about 3.1 GiB system and 2.2 GiB data space. If caches and reports cannot fit, resolve that before starting the paid run; do not delete models, environments or historical evidence without permission.
3. After GPU activation, start the 60-minute clock immediately. Within five minutes record H800/SM90 identity, driver, cgroup resources, free disk and idle GPU. `admit.py` reads the real wheel-comparison schema (`wheel_members_checked=5305`, `mismatch_count=0`, `mismatches=[]`) and binds admission to the public receipt, interpreter and boot ID. It checks the public receipt offline without network fallback; preparation's receipt alone is not admission. No fixed historical monotonic timestamp or reset clock is allowed.
4. Use the reviewed collector without backend changes or trial source builds. Startup has a 20-minute cap; collection stops at minute 55 and must not begin with less than ten minutes left. Reserve the final five minutes for stopping the server/profiler, sealing and transferring the small receipts. Retain reports remotely and verify their hashes; export and interpretation are offline. If a deadline or witness fails, preserve insufficient_evidence and do not automatically retry.
5. Shutdown after sealing; independently confirm the provider billing state. Source distribution, step sufficiency and candidate admission are reviewed from the sealed trace, not from a successful HTTP run alone.

Preparation added the [independent collector](../../experiments/native-kernel-discovery/collect.py) and CPU tests, without modifying prior frozen tools or unrelated working-tree edits. Linux syntax/help and CPU import checks passed; Linux/Nsight collection is untested. Commit and push remain user-operated. A collected report is review_pending, not evidence that a candidate exists.

## 10 Admission Repair and Unpaid Rehearsal

The earlier attempt passed public-file comparison but a private operational script treated an empty mismatch list as unequal to integer zero. After correction it needlessly fetched the public files again and encountered HTTP 503. Admission exceeded five minutes; no model or profiler started and the host was shut down. This is an apparatus failure, not a kernel result. Failed receipts remain private and unchanged. The old private `admit_and_run.py` is retired; do not invoke it.

The successor keeps model, workload, candidate criteria and the 60-minute cap unchanged. Only admission and freeze binding change. Before booking, prepare the public receipt on any network-capable CPU host and copy the complete six-file packet plus receipt to the prepared Linux host. First run the offline check and admission dry-run there; both must pass without a GPU, network, model startup or compilation. If that host is unavailable, record the Linux rehearsal as pending; do not call local mocked tests a real host pass.

```bash
# All paths below are operator-selected private paths; FREEZE is the successor SHA.
"$PY" "$P/experiments/native-kernel-discovery/preflight.py" prepare --root "$P" --freeze "$FREEZE" --receipt "$PUBLIC"
# On the prepared host, with networking unavailable:
"$PY" "$P/experiments/native-kernel-discovery/preflight.py" check --root "$P" --freeze "$FREEZE" --receipt "$PUBLIC"
"$PY" "$P/experiments/native-kernel-discovery/admit.py" --packet-root "$P" --freeze "$FREEZE" --public-receipt "$PUBLIC" --preparation "$PREP" --identity "$IDENTITY" --work "$DRY_WORK" --dry-run
# After activation, record START immediately from this host's time.monotonic().
"$PY" "$P/experiments/native-kernel-discovery/admit.py" --packet-root "$P" --freeze "$FREEZE" --public-receipt "$PUBLIC" --preparation "$PREP" --identity "$IDENTITY" --work "$WORK" --session-start-monotonic "$START"
```

Use a new work directory for each attempt. The dry-run does not initialize CUDA or produce an admission receipt. The paid admission probes capability in a short-lived child, rechecks idle GPU afterward, then launches the collector. Collector timeout stops its separate server process group. Neither local tests nor the dry-run establish serving compatibility, adequate compile-cache disk usage, or trace sufficiency; those remain runtime risks. Missing evidence is insufficient_evidence, with no automatic replacement run.
