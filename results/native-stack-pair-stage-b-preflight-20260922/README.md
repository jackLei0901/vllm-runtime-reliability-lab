# Stage B retained-environment preflight

Status: `blocked_by_build_identity` — no vLLM server or model was started.

This run attempted to enter Block 5 Stage B on the retained #53859 GPU host.
The source trees, precompiled vLLM wheel, arm environments, and dependency-pool
symlinks were still present, but the pool's original installation target had
been deleted. The target path was reconstructed from the frozen package
names and versions before the existing build-identity generator was rerun.

The frozen identity did not reproduce:

| Check | Base | Fix |
| --- | ---: | ---: |
| Frozen distributions | 187 | 187 |
| Observed distributions | 188 | 188 |
| Missing distributions | 0 | 0 |
| Extra distributions | 1 | 1 |
| Version mismatches | 0 | 0 |
| RECORD SHA-256 mismatches | 144 | 144 |
| Verified-file-count mismatches | 6 | 6 |

The extra distribution was `nvidia-ml-py`. `pool_ownership` also differed in
both arms. The complete closed comparison is in
[`mismatch-summary.json`](mismatch-summary.json).

The frozen protocol requires regenerated build records to compare byte for
byte with the reviewed records. That gate failed, so the model was not
downloaded, the server was not launched, and no C0/F1 native capture was
attempted. This result makes no statement about PyStack/py-spy
interchangeability or the #53859 fault attribution.

## Environment and evidence identity

- lab commit: `27245741a734b2eb55a7763eba62214419b3e7ed`
- Python: `3.12.3`
- GPU: `NVIDIA GeForce RTX 4090`, compute capability `8.9`
- driver: `580.105.08`
- expected records:
  `results/vllm-zmq-backpressure-stage1-build-20260915/`
- observed base build record SHA-256:
  `eb784d6e5c7238684a37c73bdf8557338e9f8684c7ccbf5bd41632a20189c0a2`
- observed fix build record SHA-256:
  `578f242722f1766f95e23a009aceb205f0418cda0afe0cc67625166412402775`
- public mismatch summary SHA-256:
  `b4925352e43a26cd31cecf84e82fa0389db92007fd7dfdebcf11399ac7e6078d`

The observed full build records remain private because the compact comparison
is sufficient to establish the failed gate. Their digests above preserve
identity without presenting reconstructed files as reviewed evidence.

## Next valid entry

Stage B may be retried only from the original frozen dependency pool or from a
newly reviewed environment identity. Reinstalling the same versions is not an
identity-preserving substitute. Stage C remains blocked.
