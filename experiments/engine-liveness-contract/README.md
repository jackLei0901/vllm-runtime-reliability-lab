# Engine liveness contract — K1/K2 CPU checks

Cheapest checks from the
[liveness contract inventory](../../docs/ENGINE_LIVENESS_CONTRACT_INVENTORY_2026-09-27.md).
Both exercise real vLLM functions loaded from an installation or an exact
source snapshot in a fake process tree; neither
starts a model, needs a GPU, or proves a downstream consequence.

Run on Linux inside the environment with the pinned vLLM
(`c8602c79062440074a018c1d5f875a5571eb6881`; the scripts record whether
`vllm.__version__` contains `gc8602c79` plus source digests, and still run on
another build so `main` can be compared):

```bash
.venv/bin/python experiments/engine-liveness-contract/k1_zero_grace_shutdown.py
.venv/bin/python experiments/engine-liveness-contract/k2_nested_shutdown_budget.py
```

K2 options: `--inner-x 1 2 5` (values of `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS`),
`--engine-teardown-delay 0.5` (EngineCore work before executor shutdown),
`--workers 2`, and `--engine-start-method fork` for a memory-limited Linux
CPU host. The default remains `spawn`. Fork changes only the fake EngineCore's
startup; both modes call the same vLLM shutdown functions. Record the method
with each cell. Do not use the fork option after initializing CUDA. Expect
roughly one minute per script; most is the spawned fake engine importing vLLM.

An uninstalled Git archive can report `vllm.__version__ == "dev"` and
`pinned_version_match == false` despite containing the exact pinned source.
In that case, independently pin the archive to the Git commit and compare the
source-file hashes; do not treat the version-string flag alone as source proof.

| Script | Positive control | Finding confirmed when | Other verdicts |
| --- | --- | --- | --- |
| K1 | `timeout=None` and `timeout=1` let the fake engine finish its 100 ms SIGTERM teardown and exit 0 | `timeout=0`: teardown never completes, engine exits by SIGKILL, and its child is killed with it (`c1_zero_grace_confirmed`) | `apparatus_failed` (control failed), `c1_not_reproduced` |
| K2 | outer grace = x + 4 s + delay + 6 s: the real `_ensure_worker_termination` completes, workers receive SIGTERM, engine exits 0 | for every tested x with outer grace = x: the inner schedule starts but does not complete, and the engine is SIGKILLed by the outer level (`c2_tested_nested_budget_shortfall`) | `apparatus_failed`, `c2_not_reproduced` |

K2 also reports, without scoring it, whether the inner SIGTERM step was
reached before the outer kill. With `--engine-teardown-delay 0` that is a
millisecond race; only completion of the full inner schedule is scored.
Every worker acknowledges that its SIGTERM handler is installed before the
fake EngineCore announces readiness. An absent inner-start marker or missing
worker acknowledgment is `apparatus_failed`, not a C2 result. The finite x
cells establish only those tested values; the general timeout-budget
inequality remains a separate source-level argument.

Not established by either check: that the vLLM call chain passes these values
in every deployment (the chain is cited from source in the inventory, and K1
records the launcher and config default as provenance), or any effect on
shared memory, NCCL teardown or clients. Those are K4/K5.

Before any future K5 GPU rerun, make the runner persist the effective
`async_scheduling` and `max_concurrent_batches` in each cell's result file.
The existing TP=2 observation remains unscored under its frozen marker; do
not broaden that rule after the fact.
