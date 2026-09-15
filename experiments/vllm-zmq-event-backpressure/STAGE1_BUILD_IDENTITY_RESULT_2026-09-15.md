# Stage 1 exact-commit build identity result

Date: 2026-09-15

Verdict: **PASS for build and dependency identity; formal Stage 1 is not frozen
or executed**

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

## Installed wheel binaries

The first record located four extensions without loading them. The replacement
schema enumerates every `.so` or executable member in the exact wheel and
requires its installed source-tree copy to match byte for byte. EngineCore's
actual mapped subset is recorded separately at hook ready time.

The wheel contains 19 such members. All 19 installed copies match the wheel and
are identical across the two arms. The full path/hash list is retained in the
two public JSON records rather than duplicated here.

| Module | SHA-256 |
| --- | --- |
| `vllm._C_stable_libtorch` | `0086011da596463c9e1d2ee8c591e9d7a9522737f29455d62fc34d8ec9b3f980` |
| `vllm._moe_C_stable_libtorch` | `bbf2462d63bac8de096d1cd1f48a979f5b39cb394238e4c8a7aebc179b13b7f7` |
| `vllm.vllm_flash_attn._vllm_fa2_C` | `f49b6ac53ef96d5bc457ef231b7cef46c8aff833431d820c2a27463a14c82dcf` |
| `vllm.vllm_flash_attn._vllm_fa3_C` | `8759be32480e0a4b0d964bd880da8b108d5647f6a46ff00472597f700d70c70e` |

These four originally sampled identities remain identical; the schema-v2 JSON
extends the same check to all 19 wheel binaries.

## Environment separation and dependency identity

The two arm environments were hard-link seeded from a clean PyTorch 2.13 CUDA
13 environment to conserve disk, then received separate editable vLLM installs
and non-editable test-plugin installs. A read-only dependency pool supplies
non-vLLM packages
through a single `stage1-dependency-pool.pth` file in each arm environment.
It explicitly excluded vLLM, torch, Triton, NVIDIA packages, the experiment
plugin, editable-install files and all `.pth` files.

The replacement record hashes the `.pth` attachment and produces a manifest of
every visible distribution: normalized name, version, `RECORD` hash, and
whether it came from the arm or pool. Duplicate distribution names fail rather
than relying on import order. The paired verifier requires those manifests to
match except for vLLM and the experiment plugin.

Each arm exposes exactly one vLLM distribution, and its imported
`vllm/__init__.py` came from the corresponding source tree. The old local vLLM
0.20.1 distribution was not visible in either arm's metadata search.

The editable installs use local commits whose trees equal the pinned trees;
those local HEAD IDs and the differing generated vLLM versions are recorded
explicitly. Content identity is based on the Git trees, not reachability of the
local commit IDs on GitHub.

- Base local HEAD: `9935dfceb7535eb4b95d29e9b8d1c83c5f918d5f`
- Fix local HEAD: `6bf585185a38354285165164286cb137908f6437`
- Base generated version: `0.1.dev1+g9935dfceb.precompiled`
- Fix generated version: `0.1.dev2+g6bf585185.precompiled`

The version-string difference is an editable-build consequence and an explicit
cross-arm difference. With eager execution it is not expected to affect the
tested path; source identity is enforced by the two Git tree hashes.

The final manifests contain 188 visible distributions per arm. Their entries
match after excluding vLLM and the test plugin. Build-only packages that had
been introduced asymmetrically while preparing the editable installs were
removed before generation; neither arm depends on them at runtime.

## Integrity

- `stage1-build-base.json` SHA-256:
  `2994accfdddb56c0112e2c2a2849f16561425071797eb03c842114648270e92b`
- `stage1-build-fix.json` SHA-256:
  `5efa1e1b3097a934e99a301dabb2bf70b12597dcd3ddb1adf824968dc6c0f682`
- Independent verifier:
  `PASS: paired Stage 1 build and dependency identities verified`

No wheel, model data, raw environment dump or machine-specific path is included
in the public result.
