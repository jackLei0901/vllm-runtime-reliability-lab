# H800 single-session preparation protocol

Status: 2026-09-30, execution protocol v1, user-approved fixed H800 route, unexecuted. A local commit freezes the files; public freeze requires pushing and confirming that same remote commit. No booking before public freeze. [Chinese](SESSION_PROTOCOL.zh-CN.md) is authoritative. Source, layout and numerical predictions follow the [packet](README.md).

## Value and scope

Test only whether the real SM90 minimal extension builds and whether B1/B2 guards reject known invalid inputs while preserving valid inputs, singleton equivalence and graph capture/replay. Both arms start at #55537 `7b054aca96cea8be1369d651c3434ad140580b92`, so both reject B0. No full vLLM build, K5 serving, K6 overhead or upstream publication. Success is local repair evidence, not production compatibility or external adoption.

## Before booking

- Review the corrected K-major B guard and valid `Bs.T` versus invalid `.contiguous()` fixtures. Publicly commit the packet, isolation runner, tests and protocol; record the commit and per-file SHA-256 before any results exist.
- Prepare reproducible pinned source/parent and clean CUTLASS `v4.7.1`. Old remote source or an existing extension cannot replace the two builds. Dependency acquisition and CMake configure remain untested.
- Recheck the new host, SSH fingerprint, SM90, cgroup memory and actual free space on both disks. Previous specifications are not current evidence.

## Fixed route: one H800 session, at most 120 minutes

Build and test on the H800 instance only; no route switching during execution. Withdraw AutoDL no-GPU and local WSL build plans. User-provided host facts: six logical processors, about 15.9 GiB RAM; Ubuntu 22.04 / WSL2 has 7.7 GiB RAM and 2 GiB swap, no nvcc/ninja/cmake on PATH and no pip for python3. Virtual-disk free space does not establish Windows physical capacity. These justify not provisioning a local environment for this validation, not claiming local compilation is impossible forever.

| Stage | Cap | Stop condition |
| --- | --- | --- |
| Environment/identity, CPU tests, compiler-command review | 20 minutes | Wrong torch 2.13.0+cu130, nvcc/ninja/CMake, SM90, RAM/space or source/dependency identity; skipped CPU tests; no same-pin CMake reference |
| Base minimal build | 35 minutes | Registration, linkage, space or timeout failure; no parameter-changing retry |
| Fix minimal build | 35 minutes | Same; do not enter without usable base and reviewed guards/flags |
| Per-case isolated tests | 7 minutes per arm | Unrun cases become unscored at deadline; no extension or rerun |
| Seal, download verification, power-off | 16 minutes | Reserve this time, not for extra tests |

The 120-minute cap starts when the user enables the billable instance, including transfer, dependency preparation and troubleshooting. Confirm an unknown start time; do not restart the clock at SSH connection. Do not borrow between stage budgets. Stop on failed preflight, without provisioning a large environment, changing flags/source or switching to a full build. Reuse an existing matched environment only after verification. Two jobs; `--timeout-seconds 2100` per build; terminate compiler process groups on timeout and inspect residual processes. If insufficient billed time remains for the next stage, seal and power off early. Apparatus failure is unscored, not a negative kernel conclusion.

Put verified `CUDA_HOME/bin` and `$VENV/bin` on PATH; check `command -v nvcc` and `command -v ninja`. Run both CPU suites in the existing torch environment; the real tensor test must pass, not skip. Extract the blockwise `.cu` command from same-pin CMake configure; preserve reference artifacts/digests and compare gencode, macros, language standard, includes, optimization and linkage. Explain differences from the minimal namespace/build mechanism rather than guessing through successive linker repairs.

Resolve scratch to an inspected absolute directory; both build and result directories must be new. Recheck space before each stage and reserve at least 3 GiB, an operational reserve rather than proof of sufficient capacity. If space is insufficient, list exact disposable caches/current-session artifacts for approval. Do not delete models, old source, environments or results. An unapproved cleanup plan cannot pass preflight.

Install pytest in the prepared environment and actually run `test_collected_id_resolves_from_unrelated_directory`: a collected ID must re-collect to exactly one case with fixed packet cwd/rootdir. Neither this nor the real CPU tensor check may skip. Missing-dependency skips locally are not passes.

Build both arms on the same verified instance; retain receipt bytes. Resolve `binary_file` beside the receipt and check its original digest; the original build path is provenance only. Receipts identify host/system/libc/Python; separately record execution host/SSH, GPU/driver, RAM/disk and torch/CUDA. Stop this run on host or environment changes rather than treating a clone, relocation or new instance as continuation.

## Test execution

Resolve and verify all absolute-path variables in the README first:

```bash
"$VENV/bin/python" "$PACKET/run_isolated.py" --arm base --receipt "$BASE_BUILD/build_receipt.json" --out "$BASE_RESULTS" --budget-seconds 420
"$VENV/bin/python" "$PACKET/run_isolated.py" --arm fix --receipt "$FIX_BUILD/build_receipt.json" --out "$FIX_RESULTS" --budget-seconds 420
```

Collect exactly 30 unique node IDs, then launch each once in a fresh process in collection order. Each case has at most 45 seconds within its arm deadline. No `-x`, reruns or hiding later cases after a failed prediction. Base has three planned invalid-capture skips and 27 active cases; fix has 30 active cases. Unexpected skips, missing/bad receipts, crashes, unusable CUDA context, setup errors and exhausted budgets are unscored. An assertion failure with usable CUDA context is prediction_missed. Only the exact three base capture skip reasons qualify as planned_skip.

The build receipt must bind isolation-runner and test digests. Preserve collection, per-case start/end, return codes, logs, JUnit and run_receipt.json. Matching all predictions gives only all_predictions_matched. Do not change thresholds, rescore or supplement failed runs. Guards execute at capture, not again on replay with changed data.

Four mixed-int8 cases are new predictions; historical B2 tested both int8 together. Seven minutes per arm is unmeasured. Record per-case and first-three durations; project one collection duration plus ten times the first-three runtime. Pin cwd/rootdir and `-c` to the packet's empty pytest.ini, also receipt-bound. Do not extend deadlines; late graph cases may remain unrun and unscored.

## Archive and exit

Privately preserve source/CUTLASS/tool identities, patch/harness/binary digests, CPU tests, CMake references/differences, build.ninja, build logs, case artifacts, timing and space. Rehash the downloaded archive; publish only necessary configuration, counts, conclusions and limits. Power off under the user's authorization; stopping billing still needs cloud-console confirmation.

The four-hour feasibility gate still requires a separate continue/stop decision using actual resources, outcomes and the blocked upstream route. Explicitly decide whether remaining K5/K6 learning work merits investment. This session does not complete the whole project.
