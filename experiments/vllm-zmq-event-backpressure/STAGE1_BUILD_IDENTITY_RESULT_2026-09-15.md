# Stage 1 exact-commit build identity result

Date: 2026-09-15

Verdict: **PASS for build identity; formal Stage 1 is not frozen or executed**

## Source and wheel

- Base tree: `b7061e73a6ed4773e16bd2ae3acf47aebfd1342d`
- Same-base #53883 tree: `46bc6e191b14ce12a04827454b4588ea5d3a435f`
- Wheel version: `0.1.1.dev19+g22258a26b`
- Wheel SHA-256:
  `1d7b69a0bc85dacd9722e2af896b6a1b602e6b4712e1ea9e81b4b4ea9fcab978`
- Wheel variant: `cu130`
- PyTorch: `2.13.0+cu130`
- CUDA reported by PyTorch: 13.0
- Driver: 580.105.08
- GPU: NVIDIA GeForce RTX 4090, capability 8.9

The downloaded wheel overlapped 3,061 tracked files in the base tree. All were
byte-identical; no mismatch was found. Both editable installs left their Git
trees clean.

## Loaded extensions

The generator required each loaded file to reside in its selected source tree
and to match the same member inside the exact-commit wheel.

| Module | SHA-256 |
| --- | --- |
| `vllm._C_stable_libtorch` | `0086011da596463c9e1d2ee8c591e9d7a9522737f29455d62fc34d8ec9b3f980` |
| `vllm._moe_C_stable_libtorch` | `bbf2462d63bac8de096d1cd1f48a979f5b39cb394238e4c8a7aebc179b13b7f7` |
| `vllm.vllm_flash_attn._vllm_fa2_C` | `f49b6ac53ef96d5bc457ef231b7cef46c8aff833431d820c2a27463a14c82dcf` |
| `vllm.vllm_flash_attn._vllm_fa3_C` | `8759be32480e0a4b0d964bd880da8b108d5647f6a46ff00472597f700d70c70e` |

All four identities were identical between the base and fix arms, as required
for a Python-only candidate patch.

## Environment separation

The two arm environments were hard-link seeded from a clean PyTorch 2.13 CUDA
13 environment to conserve disk, then received separate editable vLLM and test
plugin installations. A read-only dependency pool supplied non-vLLM packages.
It explicitly excluded vLLM, torch, Triton, NVIDIA packages, the experiment
plugin, editable-install files and all `.pth` files.

Each arm exposed exactly one vLLM distribution, and its imported
`vllm/__init__.py` came from the corresponding source tree. The old local vLLM
0.20.1 distribution was not visible in either arm's metadata search.

This environment construction is disclosed for review before freezing. The
identity PASS does not by itself approve or execute the four-cell campaign.

## Integrity

- `stage1-build-base.json` SHA-256:
  `317217c8b0fe7087fd52187cab368654c7eb8575ceaa467bd09da3b58292bbe7`
- `stage1-build-fix.json` SHA-256:
  `9a2b2b0cbe62bc49e3f74b05d7ace330f099131c4187821a624d61db00dece84`
- Independent verifier:
  `PASS: paired Stage 1 exact-commit build identities verified`

No wheel, model data, raw environment dump or machine-specific path is included
in the public result.
