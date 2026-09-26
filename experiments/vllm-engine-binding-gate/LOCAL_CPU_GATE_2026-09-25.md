# Per-engine binding gate: local CPU result — 2026-09-25

Decision: **the local source/CPU gate supports per-engine evaluation and
fail-closed degradation, but does not establish a real engine-label-to-PID
binding.** Do not implement or claim an automatic engine-alive trigger on
these inputs alone.

## Fixed inputs and environment

- Lab source: `f66da70`; only existing `dfxlab.progress` and
  `dfxlab.prometheus` rules were executed. The new experiment file contains
  synthetic vectors, not a production parser or recorder change.
- vLLM source: #53859 base commit
  `22258a26bc090bccf5473cf681bbe9bac41bd035`, inspected from the local
  Git object database. No live vLLM process was launched.
- Local runtime: Windows CPython 3.14.2, standard-library `unittest`.
  `pytest` and `psutil` were not installed in that interpreter.
- Linux `/proc` smoke test: **not run**. The available `wsl.exe` returned
  `Wsl/EnumerateDistros/Service/E_ACCESSDENIED` in this execution context.
  This is an access limit, not evidence about vLLM or `setproctitle`.

## Pinned source chain

| Boundary | Source observation | What it does not prove |
| --- | --- | --- |
| Engine process naming | `vllm/v1/engine/core.py:1294-1302` selects `EngineCore_DP{dp_rank}` for DP and `EngineCore` otherwise, then calls `set_process_title`. | That `/proc/<pid>/cmdline` actually contains the title in this environment. |
| Optional producer | `vllm/utils/system_utils.py:184-198` returns when `setproctitle` cannot be imported. | Missing title is not missing engine. The configured process-name prefix also means exact full-title matching is inappropriate. |
| Client rank set | `vllm/v1/engine/core_client.py:690-706` derives `engine_ranks_managed` from DP index, local size and mode. | That every multi-API-server/dynamic mode shares a single global mapping without an additional check. |
| Metrics rank set | `vllm/v1/engine/async_llm.py:169-176` passes `engine_ranks_managed` to `StatLoggerManager`; `loggers.py:469-475` builds `(model_name, engine)` label values from those indexes. | That a sampled label and a particular operating-system PID are independently bound. |
| Counter producer | `loggers.py:705-711` creates generation-token counters per engine. | That the published R3 campaign retained the per-engine metric series; it did not. |
| Current Lab reader | `prometheus.py:48-55` sums matching label variants; `collectors.py:27-33` does not retain generation tokens in recorder observations. | Existing `record` cannot make a per-engine no-progress claim from its current public schema. |

The ordinary AsyncLLM source path gives a plausible `engine` label ↔ DP
rank *index* relation. The process title is an optional, self-reported
corroborator. Neither source reading nor the synthetic tests prove that a
particular label belonged to a particular live PID across all launch modes.
PID start identity must be checked separately; a title must never replace it.

## Executed CPU controls

Command from the Lab checkout on PowerShell:

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -m unittest discover -s experiments/vllm-engine-binding-gate -p test_cpu_contract.py -v
```

Result: **7 new experiment tests passed**. Existing nearby controls also
passed: 6 `test_progress.py`, 1 `test_prometheus.py`, and 11
`test_collect_verify_contract_cases.py` tests, for **25 passing tests in
total**. `compileall` passed for the experiment file. The observed
distinctions were:

| Control | Result and limit |
| --- | --- |
| Two labeled metric series through today's `select_metrics()` | Values 10 and 20 became one total 30; engine partition information was discarded. |
| Engine 0 flat, engine 1 rising | Existing pure producer evaluator returned `flat` and `progressing` respectively; the summed 30→31 series returned `progressing`. This is a synthetic partition input, not a new parser. |
| Continuous work versus idle | Flat counter with no admitted demand yielded `undetermined`, not no-progress. |
| Cached-only counter values | `insufficient_evidence`, not `flat`. |
| No bound process | Flat progress with no process target yielded `undetermined`. This models unavailable title *and no other accepted binding*; no title collector ran. |
| Changed process start identity | `process_missing` outranked flat progress. No real PID was reused in this test. |
| Request flat while its engine counter rose | Engine counter remained progressing; request conflict stayed visible, without converting the engine to no-progress. |

The v0.2 server-counter verdict still reports `progress_scope="service"` in
these tests. That is a useful negative control: the proposed engine scope
requires a future versioned claim contract, not a relabel of v0.2 output.
The 7 tests check existing rule composition after an engine partition has
been supplied. They **do not** test label-preserving metric parsing, a real
title, a live `/proc` start tick, real DP scheduling, or the actual recorder
capture path.

## Next gate, not silently satisfied

On a Linux host with an executable pinned vLLM build, perform a single-
engine smoke check of `/metrics`, `/proc/<pid>/cmdline`, and
`/proc/<pid>/stat` across several fresh scrapes, including a missing-title
control. Record the exact build and configuration. Only a DP=2 run can
exercise one engine flat while another advances; multi-API-server modes
need their own mapping check. If title or rank mapping is unavailable, the
result is **binding unavailable**, not “engine absent,” and the automatic
engine-alive trigger remains NO-GO. No GPU run is claimed here.

## Secondary-witness CPU matrix (2026-09-26)

`binding_evidence.py` is a separate, CPU-only experiment. It accepts a
caller-verified fresh log boundary and process-tree snapshots at both ends
of one evaluation window. An anchored EngineCore prefix is only a candidate
rank/PID witness; a positive binding also requires two matching lines, a
stable direct-child PID/start identity, matching PID namespace and boot ID,
and an expected local engine name. Missing or contradictory evidence yields
`binding_unavailable`, never `engine_missing`.

The 21 cases in `test_binding_evidence.py` pass. They cover two DP names,
missing and conflicting witnesses, a spliced prefix, embedded carriage
return, worker/API-server names, one- and two-sided restarts, PID reuse,
ancestry, namespace, launch-root changes and unsupported Ray launch. The
earlier seven rule-composition controls also pass, so the publishable CPU
set has 28 passing tests. Run from the repository root with `PYTHONPATH=src`:

```bash
python -m unittest discover -s experiments/vllm-engine-binding-gate \
  -p 'test_binding_evidence.py' -v
python -m unittest discover -s experiments/vllm-engine-binding-gate \
  -p 'test_cpu_contract.py' -v
```

This establishes internal consistency of the proposed join, not real log
custody, an observed DP=2 mapping, or a new v0.2 verdict. The caller must
verify the run boundary and retain its digest before setting
`log_custody_verified=True`. PIDs and start ticks remain private; this
helper is not a public bundle producer or automatic recorder trigger.
