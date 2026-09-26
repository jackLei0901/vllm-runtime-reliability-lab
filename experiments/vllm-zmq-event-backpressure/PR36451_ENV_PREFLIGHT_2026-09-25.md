# #36451 runtime environment preflight

Status: **environment NO-GO on the existing single-4090 instance**; no
runtime health-ping cell was started. This is not a NO-GO for the #36451
mechanism or the preregistered experiment.

Pinned source: `sihyeonn/vllm@206f28dc3b33abc4d0425fce8d8e14b1ea9cd7e0`.
Its [`pyproject.toml`](https://github.com/sihyeonn/vllm/blob/206f28dc3b33abc4d0425fce8d8e14b1ea9cd7e0/pyproject.toml)
specifies `torch == 2.11.0` for the build. The existing instance has one
RTX 4090 with 24 GiB-class VRAM, approximately 12 GiB free on the data
volume and 7.6 GiB free on the root volume. Its existing vLLM virtual
environment reports `torch 2.13.0+cu130`; the bounded search of existing
workspace/cache paths found no Torch 2.11 installation to reuse. The
installed vLLM environment and other checkouts were not modified.

Consequences:

- Running PR-head Python over the existing 2.13 binary environment would
  confound the exact-source experiment and is not acceptable evidence.
- Building a separate exact-head environment within the remaining 12 GiB
  has an unbounded disk-failure risk. No build/download was attempted and
  no old cache, model, checkout or virtual environment was deleted.
- The GPU itself is adequate for a DP=1 experiment; the blocker is an
  isolated, compatible software environment with sufficient working space.

Resume gate: provide an instance or enlarged data volume with at least
approximately 30 GiB **total free working space** (about 18 GiB more than
this instance currently has), compatible CUDA
driver, and permission to create an isolated source checkout and venv.
The previously retained model may be reused after identity/compatibility
checks. First build and smoke-test the exact commit, then test plugin
compatibility, and only then run the four preregistered GPU cells. If a
prebuilt matching wheel is available, re-evaluate the space requirement;
do not silently substitute an unrelated wheel.
