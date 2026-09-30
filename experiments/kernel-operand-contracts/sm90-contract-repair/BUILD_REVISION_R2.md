# Build-tool deviation R2 and reproduction recipe

The [Chinese version](BUILD_REVISION_R2.zh-CN.md) is authoritative.
This is a post-run publication supplement, not a replacement freeze.

## What changed

The user authorized R2 before compilation: match both minimal arms to actual
CMake commands generated from `7b054aca96cea8be1369d651c3434ad140580b92`.
C++17 became C++20; stable ABI/USE_CUDA/Py_LIMITED_API definitions, the generated
patched Torch header include, ENABLE_FP8 and associated CUDA options were added.
Four Torch half-conversion-disabling flags were removed as pinned CMake does.
Unrelated full-extension kernel macros were omitted because their sources are
not part of the minimal extension. Its isolated namespace and Torch loader/link
options still differ from the full extension. No patch/test/threshold changed.

The executed private builder SHA-256 was
`bb624fef3235d207f8c115fdffa313c84d086203d60dc9b208757e777da7d3b9`.
[build_minimal_r2.py](build_minimal_r2.py) is its publication successor:
`a16daa2bafc476a25aea35e51de439168a205516578ac9b1e24f0e2007ed2168`.
Its only change is using `Path(__file__).name` instead of the literal
`"build_minimal.py"` when binding its own receipt digest. It therefore binds the
correct public filename; compiler parameters are unchanged. A CPU test reverses
this one substitution and reproduces the executed builder's exact hash.
The successor has not produced a new GPU result. Original builder/manifest stay
unchanged. New receipts bind the successor; historical receipts stay unchanged.

## Portable preparation

Use Linux, Python 3.12, torch **2.13.0+cu130**, CUDA toolkit **13.0**, a clean
CUTLASS **v4.7.1** checkout and the exact vLLM pin. The run used GCC 11.4,
CMake 4.4.3 and Ninja 1.13.2. Both builds and tests ran on the rented H800 host,
not WSL or no-GPU mode. Two compile jobs and 35-minute build caps were used.

All paths below are operator-selected absolute paths; none is a machine default.
Define `PACKET`, `VENV`, `CUDA_HOME`, `SOURCE_REPO`, `BASE_SRC`, `FIX_SRC`,
`CUTLASS_SRC`, `REFERENCE`, `BASE_BUILD`, `FIX_BUILD`, `BASE_RESULTS` and
`FIX_RESULTS` before execution. Use new output/reference directories. Obtain
the exact source pin before creating worktrees; if unavailable, stop rather than
substitute current main. The saved #55537 patch alone is not a source archive.

```bash
git -C "$SOURCE_REPO" worktree add --detach "$BASE_SRC" 7b054aca96cea8be1369d651c3434ad140580b92
git -C "$SOURCE_REPO" worktree add --detach "$FIX_SRC" 7b054aca96cea8be1369d651c3434ad140580b92
git -C "$FIX_SRC" apply "$PACKET/candidate.patch"
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=9.0a
cmake -S "$BASE_SRC" -B "$REFERENCE" -G Ninja \
  -DVLLM_PYTHON_EXECUTABLE="$VENV/bin/python" \
  -DVLLM_TARGET_DEVICE=cuda -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

CMake may fetch its pinned dependencies; this step configures, not builds the
full extension. Use exact pin-declared dependencies, not arbitrary cached
checkouts. In this run, a stale FlashMLA cache lacked `sparse_prefill.cpp`;
configuration succeeded after fetching `0eee43b12f034b657133cf2afca6a72ebb6efccf`
into a new directory and passing `-DFETCHCONTENT_SOURCE_DIR_FLASHMLA=<that-dir>`.
The same approach is available when needed, not a required machine-specific path.

`REFERENCE` must contain the real `compile_commands.json` and generated
`torch_patched_headers`. The builder checks critical flags and hashes the command
file/headers. Patched headers come from pinned CMakeLists.txt's torch 2.13 hotfix,
not hand-written substitutes. Inspect their provenance and effective commands;
flag checks alone cannot authenticate an arbitrary supplied reference.

```bash
"$VENV/bin/python" "$PACKET/build_minimal_r2.py" --vllm-src "$BASE_SRC" \
  --cutlass-src "$CUTLASS_SRC" --cmake-reference "$REFERENCE" --arm base \
  --out "$BASE_BUILD" --timeout-seconds 2100
"$VENV/bin/python" "$PACKET/build_minimal_r2.py" --vllm-src "$FIX_SRC" \
  --cutlass-src "$CUTLASS_SRC" --cmake-reference "$REFERENCE" --arm fix \
  --out "$FIX_BUILD" --timeout-seconds 2100
"$VENV/bin/python" "$PACKET/run_isolated.py" --arm base \
  --receipt "$BASE_BUILD/build_receipt.json" --out "$BASE_RESULTS" --budget-seconds 420
"$VENV/bin/python" "$PACKET/run_isolated.py" --arm fix \
  --receipt "$FIX_BUILD/build_receipt.json" --out "$FIX_RESULTS" --budget-seconds 420
```

This recipe is available for independent reproduction; no non-author rebuild
or execution has yet been observed. K7 independent execution is not passed.
Historical logs/receipts include host paths and remain private. No portion of
the builder requires secrecy; external reproduction generates its own receipts.

## R1 evidence

The transported source had the right HEAD/content but no Git index. The repair
was `git -C <source> read-tree HEAD`, then `git -C <source> diff --exit-code HEAD`
and a clean status check. `source-diff-r1.txt` preserves the zero diff exit code;
`base-plan-r1.json` and both build receipts record subsequent source validation,
whose base gate requires a clean status. The archived index count of **0** and
tree count **7266** describe the failure **before** R1, not its repaired state.
The private archive digest is in the [result](RESULT_2026-09-30.md).
