# Stage 1 runbook draft

Status: **frozen pre-execution runbook**

## Required environment

- Linux with one supported NVIDIA GPU;
- two source worktrees whose `HEAD^{tree}` values exactly match the base and
  same-base #53883 trees in the protocol;
- two fresh Python environments, one for each worktree, created from the same
  Python and the same locked dependency inputs;
- `py-spy` available to the campaign process;
- one small model whose request reports at least 16 prompt tokens and 32
  streamed completion tokens in both controls.

The field report's DP=8 setup is not required.

The machine's installed vLLM 0.20.1 wheel was rejected: its Python payload
differs from the pinned baseline tree. The prepared environments use the
upstream precompiled wheel for commit `22258a26` and pinned `torch==2.13.0`.
Their recorded identities are under
`results/vllm-zmq-backpressure-stage1-build-20260915/`.

Each arm has a file named `stage1-dependency-pool.pth` in its own
`site-packages`. The file contains exactly one absolute path:
`/root/stage1-runtime/dependency-pool`. The pool contains only symlinks to the
shared non-core dependencies. It excludes vLLM, torch, Triton, NVIDIA packages,
the test plugin, editable-install helpers and all source `.pth` files. Do not
attach the pool with an additional `PYTHONPATH` or `sitecustomize` entry.

The expected upstream wheel filename is:

```text
vllm-0.1.1.dev19+g22258a26b-cp38-abi3-manylinux_2_28_x86_64.whl
```

After downloading it once, use its absolute local path for both installs:

```bash
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_LOCATION=/absolute/path/to/vllm-22258a26.whl \
uv pip install --python /absolute/path/to/arm/bin/python -e /absolute/worktree
```

The formal freeze includes both build-identity JSON hashes and the build
generator and verifier.

## Plugin installation

Install the test hook into each vLLM environment without dependencies. Do not
use editable mode: it exposes both source `egg-info` and environment
`dist-info`, defeating the no-duplicate-distribution rule.

```bash
python -m pip install --no-deps --no-build-isolation \
  experiments/vllm-zmq-event-backpressure/stage1_plugin
```

The campaign sets the plugin allow-list and hook environment. Do not export
those variables globally.

## Exact inputs awaiting review and freeze

The proposed private command and request are documented in
`STAGE1_INPUT_REVIEW_2026-09-15.md`. The server command must include:

- one fixed model and revision;
- V1 serving;
- one API port;
- `--enable-prefix-caching`;
- `--block-size 16`;
- `--async-scheduling`;
- `--kv-events-config` with ZMQ enabled and `max_queue_size` set to 1;
- a fixed maximum model length and GPU-memory limit.

Create a synthetic streaming request JSON with deterministic sampling,
`stream_options.include_usage=true`, at least 16 prompt tokens and at least 32
output tokens. The request is private input; only its hash, progress counts and
usage token counts enter the public result.

## Per-cell command shape

```bash
python experiments/vllm-zmq-event-backpressure/stage1_campaign.py \
  --cell-index 1 \
  --source-arm base \
  --trigger control \
  --server-workdir /absolute/path/to/base \
  --server-command-json /absolute/path/to/server-command.json \
  --request-json /absolute/path/to/request.json \
  --build-identity-json /reviewed/results/stage1-build-base.json \
  --health-url http://127.0.0.1:8000/health \
  --stream-url http://127.0.0.1:8000/v1/completions \
  --private-dir /private/unique/base-control \
  --summary /reviewed/results/base-control.json
```

Run in this order:

1. base/control;
2. fix/control;
3. base/pause;
4. fix/pause.

Stop immediately after a failed control. Never reuse a private/control
directory. Inspect the structured result before deleting private logs.

Cleanup always signals the whole server process group, even if the API process
has already exited. A cell fails unless the group is gone and the bound
EngineCore PID plus start time no longer identifies a live process.

## Preflight checks

Before the first cell:

```bash
python experiments/vllm-zmq-event-backpressure/stage1_campaign.py --help
python experiments/vllm-zmq-event-backpressure/verify_stage1_build_identity.py \
  results/vllm-zmq-backpressure-stage1-build-20260915
python -c "import importlib.metadata as m; print([e.name for e in m.entry_points(group='vllm.general_plugins') if e.name == 'dfx_stage1_backpressure'])"
py-spy --version
git -C /absolute/path/to/base rev-parse HEAD^{tree}
git -C /absolute/path/to/fix rev-parse HEAD^{tree}
```

The freeze verifier must pass on the rented machine before any server is
started.

Then regenerate the two build records into a fresh private directory and
compare them byte for byte with the frozen public records:

```bash
CHECKPOINT=/root/stage1-runtime/build-checkpoint-before
rm -rf "$CHECKPOINT"
mkdir -p "$CHECKPOINT"

for arm in base fix; do
  /root/stage1-runtime/$arm-env/bin/python \
    experiments/vllm-zmq-event-backpressure/stage1_build_identity.py \
    --worktree /root/stage0-53859/$arm \
    --wheel /root/stage1-runtime/vllm-0.1.1.dev19+g22258a26b-cp38-abi3-manylinux_2_28_x86_64.whl \
    --dependency-pool /root/stage1-runtime/dependency-pool \
    --output "$CHECKPOINT/stage1-build-$arm.json"
  cmp \
    "$CHECKPOINT/stage1-build-$arm.json" \
    "results/vllm-zmq-backpressure-stage1-build-20260915/stage1-build-$arm.json"
done
```

All commands above must return zero. Repeat the same block after the fourth
cell with `CHECKPOINT=/root/stage1-runtime/build-checkpoint-after`. Preserve
the four generated JSON files privately until the result has been reviewed.
