# v0.1.0-alpha.3

This release makes the public validation path reproducible from a clean clone
and restores machine-checked requirement-to-test traceability.

## Corrections

- The README now installs the `dev` extra before running the test suite, so the
  documented clean-clone path includes `jsonschema` and Ruff.
- The installed schema path now uses
  `share/vllm-runtime-reliability-lab/schema`.
- Functional and safety requirements have stable `FR-*` and `SR-*` identifiers.
- `tests/cases.json` maps every requirement to one or more concrete tests.
- `test_testplan.py` checks bidirectional requirement coverage, non-empty
  mappings, and the existence of every referenced test method.

## Validation

The documented clean-clone path passes 33 tests on the release source, with the
POSIX permission test skipped on Windows. CI also runs lint, formatting,
compilation, CLI smoke tests, and the suite on Python 3.10, 3.12, and 3.13.

The runtime contract and schema are unchanged from alpha.2. The RTX 4090 fatal
matrix remains documented in
[`results/gpu-20260909-alpha2/VALIDATION_SUMMARY.md`](results/gpu-20260909-alpha2/VALIDATION_SUMMARY.md).

Wheel SHA-256:

```text
15b812029a382b8feeb5699daff186e389e9483bc1fbd861268d16f89590d5c4
```
