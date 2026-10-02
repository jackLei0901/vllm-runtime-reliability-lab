# Per-token FP8 value-only reduction evidence

Operator-only H800 PCIe measurements, PyTorch 2.13.0+cu130, vLLM source
`378504a5442b8a9240b359dae3e6f75f35c38f19`. No model TPOT claim.

Files:

- `fp8_m_codegen_probe.py`: fresh-process max/amax/max codegen comparison.
- `fp8_m_perf_probe.py`: fresh-process max/amax/max/amax/max graph-replay timing.
- `fp8_amax_correctness.py`: native max/amax value checks.
- `codegen-summary.json`: codegen and output-hash results.
- `perf-4096-summary.json`: seven shapes at hidden 4096.
- `perf-additional-summary.json`: fourteen shapes at hidden 7168/14336.
- `fp8_special_correctness.py`: final patched-native eager/compiled check for finite inputs, NaN and ±Inf, BF16/FP32, with upper bound on/off. NaN scales use `equal_nan=True`, not a NaN payload identity assertion.
- `VALIDATION_20261002.md`: final fusion, model quality, lint and offline audit summary.

The original scripts and JSONs were scanned for rented-host identifiers and
absolute host paths. No matches were found, so no measurement data was rewritten.
File line endings are normalized to LF in this sharing package.

Use a Linux CUDA environment with vLLM and compatible native artifacts installed.
Run using its virtualenv Python; do not use system Python. The scripts guard
against source drift and require the original per-token `max` line to be present.
After applying the one-line source patch they intentionally fail this guard:
run the max/amax probes on unpatched source, and existing pytest tests on patched
source. `fp8_special_correctness.py` instead requires the patched native source;
download it from this directory and run it with the vLLM virtualenv Python.
Synthetic RMSNorm is not a vLLM IR/model fusion test.

```bash
OMP_NUM_THREADS=1 .venv/bin/python fp8_m_codegen_probe.py --out-dir codegen-out
OMP_NUM_THREADS=1 .venv/bin/python fp8_m_perf_probe.py \
  --hidden-sizes 4096 7168 14336 --out-dir perf-out
OMP_NUM_THREADS=1 .venv/bin/python fp8_amax_correctness.py
```

Outputs/scales matched across the tested arms. Eager and compiled execution need
not match each other bitwise due to fusion rounding; the evidence does not claim
universal bitwise identity on every backend or special input. These scripts are
standalone validation materials, not proposed additions to vLLM's test suite.
