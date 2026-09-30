# SM90 contract repair: build and test preparation

Status: 2026-09-30, execution packet v1, frozen by local commit, uncompiled and unexecuted. Public freeze requires pushing and confirming that same commit before booking. [Chinese](README.zh-CN.md) is authoritative. See the [requirements](../../../docs/kernel/SM90_BLOCK_FP8_PROJECT_REQUIREMENTS.md). Historical B1/B2/B0 scripts remain unchanged. File digests are in [FREEZE_MANIFEST.json](FREEZE_MANIFEST.json); later semantic changes require a dated amendment before execution.

## Purpose and limits

`candidate.patch` changes only the SM90 blockwise `.cu`, rejecting non-e4m3 A/B and non-packed scales while permitting physically equivalent singleton dimensions. Both arms use pinned #55537 `7b054aca96cea8be1369d651c3434ad140580b92`; B0 must be rejected in both. This local policy has no upstream authorization and opens no PR.

`build_minimal.py` compiles the real entry, common.cpp, SM90 dispatch and FP8/INT8/AZP/blockwise kernels. Only `binding.cpp` uses an isolated `lab_sm90_contract` namespace. No copied entry checks or kernels substitute for source. Do not import vLLM, reuse an old binary, or load both arms in one process. The standard path remains compiled for future controls; current tests focus on blockwise.

This is not the full vLLM extension or K5 serving validation. Stable registration, linkage and flags still require Linux compilation. Success means `built_not_validated`; failure/timeout means `unscored`. Builds use two jobs and a 45-minute bound, terminating the compiler process group on timeout. No automatic rerun or full-build fallback; inspect for residual processes after timeout.

## Local checks

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -p test_sm90_contract_build_plan.py
.venv/Scripts/python.exe experiments/kernel-operand-contracts/sm90-contract-repair/build_minimal.py --vllm-src ../vllm-55537-reject-only --arm base --plan
```

Review actual kernel element types, column-major A scales, K-major B scales (column-major for the transposed 2-D tensor), singleton equivalence and guard placement before output-dtype/swap dispatch. Valid B is contiguous `(Nblocks,Kblocks)` followed by `.T`, with strides `(1,Kblocks)`; invalid B is that view's `.contiguous()`, with strides `(Nblocks,1)`. CPU checks do not establish GPU correctness.

## Linux build preparation (not yet execute-ready)

Create fresh detached base/fix worktrees from the pinned repository, not an old remote HEAD. Apply only the candidate patch to fix, unstaged. CUTLASS must be a clean `v4.7.1` checkout; the driver records its commit/header digests and downloads nothing. Preserve reproducible source/parent availability separately from the saved patch.

```bash
git -C "$SOURCE_REPO" worktree add --detach "$BASE_SRC" 7b054aca96cea8be1369d651c3434ad140580b92
git -C "$SOURCE_REPO" worktree add --detach "$FIX_SRC" 7b054aca96cea8be1369d651c3434ad140580b92
git -C "$FIX_SRC" apply --check "$PACKET/candidate.patch"
git -C "$FIX_SRC" apply "$PACKET/candidate.patch"
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$VENV/bin:$PATH"
"$VENV/bin/python" "$PACKET/build_minimal.py" --vllm-src "$BASE_SRC" --cutlass-src "$CUTLASS_SRC" --arm base --out "$BASE_BUILD"
"$VENV/bin/python" "$PACKET/build_minimal.py" --vllm-src "$FIX_SRC" --cutlass-src "$CUTLASS_SRC" --arm fix --out "$FIX_BUILD"
```

Resolve these variables to inspected absolute paths first; commands with undefined variables are not ready to paste. The old H800 `vllm-fp8-venv` is a candidate, subject to renewed identity checks. The driver requires torch 2.13.0 and selects `TORCH_CUDA_ARCH_LIST=9.0a`. Review actual gencode/compile/link commands in build.ninja and the log. Never overwrite/install into existing vLLM. Output directories must be new; raw receipts/logs stay private.

Try only base compilation first. During the trial, extract the real blockwise `.cu` command from vLLM CMake configure at the same pin (`compile_commands.json` or build.ninja). Compare gencode, macros, language standard, includes, compiler and link options against the minimal extension; preserve reference commands/digests and differences. Do not guess configuration through successive link-error repairs. Stop on registration, linkage, space or time failure; review flags and patch before building fix. Approximately 22 GiB system-disk free space is candidate scratch, not proof that two builds fit. The data disk's 4.4 GiB cannot authorize a full build. No automatic deletion of models/source/results. Use observed artifact size/build time to estimate K4/K5 and make the four-hour continue/stop decision.

## GPU regression (only after freeze)

Use the [fixed H800 protocol](SESSION_PROTOCOL.md): 120 minutes from enabling the billable instance, including preflight, base/fix builds, tests and shutdown. Local WSL and AutoDL no-GPU builds are not execution alternatives. Collection is 30 per arm (base: 27 active plus 3 planned skips; fix: 30 active); each case runs once in a fresh process. Builds use `--timeout-seconds 2100`.

`test_repair_sm90.py` covers valid bf16/fp16, ordinary/swapped M, independent A/B wrong dtypes, independent A/B wrong scale layouts, A/B/output padding, singleton equivalence and valid graph capture/replay. Fix rejects dtype/A-layout/B-layout at capture and then runs a valid context-recovery control. Base skips those three invalid capture cases. Singleton controls make a scale contiguous only when its own dimensions include size one; the nonsingleton B layout in `(1,512,1024)` stays unchanged.

Frozen predictions: valid relative error below 0.005; base wrong dtype/layout non-finite or error above 0.05; fix raises the matching error; both arms reject B0. Public freeze and environment gates remain necessary before execution. Replay does not repeat guards.

```bash
"$VENV/bin/python" "$PACKET/run_isolated.py" --arm base --receipt "$BASE_BUILD/build_receipt.json" --out "$BASE_RESULTS" --budget-seconds 420
"$VENV/bin/python" "$PACKET/run_isolated.py" --arm fix --receipt "$FIX_BUILD/build_receipt.json" --out "$FIX_RESULTS" --budget-seconds 420
```

Verify source, CUTLASS, harness and binary receipt digests before running; save collection/stdout/stderr. `harness_sha256` includes the test and isolation runner. Tests verify their own digest, binary digest, torch version and arm before loading. Record digests at freeze too. A case with lost CUDA context is unscored; the next case uses a new process, never a rerun. JUnit records cases, not adoption. Missing controls cannot establish repair success.

## Outstanding gates

Current revised `test_repair_sm90.py` SHA-256: `3a3d23355696244a56f4287a98bcabf5275e31b9ff11a72b3b9a78417edd3d40`. Later edits require updated review and receipts. Relocated binaries resolve beside the receipt using `binary_file`, preserving original digests and build-host paths. pytest.ini is receipt-bound and selected with `-c`. Singleton controls require the exemption; mixed-int8 cases are new predictions.

The two CPU suites have 21 checks: 19 pass; real torch-tensor and pytest re-collection checks skip locally because dependencies are absent. cwd/rootdir/config wiring and relocated-binary digest tests pass but cannot substitute for real re-collection or binary loading. Installing pytest was blocked by network policy. Both skipped checks must pass in the prepared environment before proceeding. Minimal compilation/GPU behavior remain unverified; K5/K6 are unfrozen. No remote operations, booking or four-hour gate pass is claimed.
