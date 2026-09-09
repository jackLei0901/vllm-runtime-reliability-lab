# vLLM Runtime Reliability Lab

[![CI](https://github.com/jackLei0901/vllm-runtime-reliability-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/jackLei0901/vllm-runtime-reliability-lab/actions/workflows/ci.yml)

An opt-in, external flight recorder and fault-injection lab for investigating
vLLM runtime failures. It keeps a bounded history of low-cardinality health,
process, GPU and selected vLLM metrics, then writes a small incident artifact
when an externally observable condition is met.

This is an **alpha research tool**, not a production monitor and not a root-cause
classifier. An external observer can preserve chronology, but it cannot infer an
internal EngineCore exception kind or execution stage. The artifact records both
as `unknown` by construction.

## Why this exists

A final traceback explains where a process stopped. It often does not preserve
the bounded runtime history needed to answer whether KV pressure, waiting work,
preemption or process loss preceded the failure. This project makes that evidence
reproducible while testing a deliberately narrow privacy and resource contract.

The design was informed by the investigation behind
[vLLM #48966](https://github.com/vllm-project/vllm/issues/48966),
[PR #52178](https://github.com/vllm-project/vllm/pull/52178), and the
[runtime incident snapshot RFC](https://github.com/vllm-project/vllm/issues/54229).
It remains independent of vLLM and does not modify EngineCore.

## What the alpha provides

- `/health` and selected `/metrics` polling;
- explicit PID liveness and Linux RSS/VMS/thread sampling;
- aggregate `nvidia-smi` sampling on a slower, independent cadence;
- a fixed-size in-memory history;
- process-exit, health-loss, KV-pressure and preemption triggers;
- a closed JSON contract with `additionalProperties: false`;
- allow-listed aggregate fields only: no prompt, token IDs, request IDs or paths;
- per-process HMAC incident IDs that do not correlate across restarts;
- 256 KiB artifact cap, four-file rotation and POSIX mode `0600`;
- fail-open writer behavior: artifact failure does not signal the observed service;
- Markdown summaries and explicit signal-injection helpers.

## Five-minute CPU-only demo

Python 3.10 or newer is required. The runtime package has no third-party
dependencies.

```bash
python -m pip install -e .
python examples/fake_vllm_server.py --port 18000
```

In a second terminal:

```bash
vllm-dfx record \
  --base-url http://127.0.0.1:18000 \
  --output demo-output \
  --history 16 \
  --duration 5
```

The healthy demo does not manufacture an incident. To exercise the health-loss
path, stop the fake server while the recorder is running:

```bash
vllm-dfx record \
  --base-url http://127.0.0.1:18000 \
  --output demo-output \
  --history 16 \
  --unhealthy-samples 2 \
  --stop-on-incident
```

Then summarize the generated artifact:

```bash
vllm-dfx summarize \
  --input demo-output/incident-*.json \
  --output demo-output/incident-summary.md
```

The shell wildcard in the last command is intended for POSIX shells. On
PowerShell, pass the concrete artifact filename.

## Real vLLM usage

Wait until the server is ready, then provide the API-server PID explicitly:

```bash
vllm-dfx snapshot-env --output private-run/environment.private.json

vllm-dfx record \
  --base-url http://127.0.0.1:8000 \
  --pid "$API_SERVER_PID" \
  --output shareable-incidents \
  --sample-interval 1 \
  --gpu-interval 5 \
  --history 300
```

`snapshot-env` and `run-summary.private.json` are private lab metadata. Review
them before sharing. Files named `incident-*.json` follow the stricter external
artifact contract.

Raw timeline persistence is disabled by default. The explicit
`--private-raw-timeline` option writes an unbounded private debugging file and
should not be enabled for unattended production use.

## Fault injection

The tool never discovers a process through fuzzy command matching. Inspect the
process tree and pass the intended PID:

```bash
vllm-dfx inject-signal \
  --pid "$ENGINE_CORE_PID" \
  --signal SIGKILL \
  --event-log private-run/injections.jsonl \
  --dry-run
```

Remove `--dry-run` only in an isolated environment where process termination is
expected. The injection event log is private lab evidence and is not part of the
shareable incident schema.

## Artifact contract

The machine-checkable contract is
[`schema/external-runtime-observation-v1.schema.json`](schema/external-runtime-observation-v1.schema.json).
A synthetic example is available at
[`examples/external-runtime-observation-v1.json`](examples/external-runtime-observation-v1.json).

The shareable artifact contains:

```text
typed external trigger
runtime version and GPU-model allow-list
bounded health/process/metrics/GPU history
recorder overwrite/drop counters
writer success/failure counters
```

It intentionally omits prompt text, token IDs, request identifiers, model paths,
environment variables, command lines, tracebacks and arbitrary configuration.
The allow-list is the primary boundary; the legacy deny-list is only a secondary
guard for private lab metadata.

## Validation

Run the CPU suite:

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests
vllm-dfx --help
```

The repository includes reviewed summaries from earlier RTX 4090 experiments:

- startup KV-capacity boundary;
- paired KV-pressure/preemption behavior;
- a 1,200-request, ten-minute smoke soak;
- intentional SIGTERM, EngineCore SIGKILL and controlled `execute_model` OOM;
- separate TP=2 process-level validation associated with PR #52178.

These results establish test-harness behavior only for the pinned environments.
They do not establish long-term stability, DP/NCCL behavior or production value.
The post-alpha GPU validation plan is in [`TEST_PLAN.md`](TEST_PLAN.md).

## Documentation

- [`REQUIREMENTS.md`](REQUIREMENTS.md): scope and acceptance criteria.
- [`DESIGN.md`](DESIGN.md): trust boundary and component design.
- [`TEST_PLAN.md`](TEST_PLAN.md): CPU and GPU validation matrix.
- [`SECURITY.md`](SECURITY.md): privacy assumptions and reporting guidance.
- [`CHANGELOG.md`](CHANGELOG.md): release history.

## Status

`v0.1.0-alpha.1` validates the external artifact contract and CPU/local-service
behavior. GPU re-validation of this hardened schema is the next gate. Until that
is complete, treat this release as an evaluation build.

## License

MIT
