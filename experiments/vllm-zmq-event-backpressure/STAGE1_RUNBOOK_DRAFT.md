# Stage 1 runbook draft

Status: **review only; do not execute yet**

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

The formal freeze must include both build-identity JSON hashes and both
implementation scripts. The values are recorded but not frozen yet.

## Plugin installation

Install the test hook into each vLLM environment without dependencies:

```bash
python -m pip install --no-deps -e \
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

The future freeze verifier must pass on the rented machine before any server is
started.
