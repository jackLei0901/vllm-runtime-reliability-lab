# V1 recorder–verdict integration gate — design for review, 2026-09-25

Status: **proposed design, not implemented or accepted**. This document names
what must be true before a future recorder can automatically retain its
pre-fault ring on a demand-gated no-progress claim. It does not modify the
released v0.2 `collect/verify` contract or assert that #53859 was invisible
to Prometheus.

Source baseline: Lab `1b28aff` (the six-document DFX review commit). Case
baseline: published #53859 Stage 1 R3 four-cell summaries and replay
manifest. No new GPU run or metrics time series was used.

## Problem and existing boundary

`record` already keeps a bounded `ExternalObservation` ring, nominally 300
samples at a 1-second default interval. `IncidentRecorder.classify()` has
four triggers: process exit, health loss, KV pressure, and preemption storm.
A live process with 2xx health and no useful output can fire none of them,
so the ring may be overwritten rather than frozen for V1. Meanwhile
`collect/verify` can make `alive_health_ok_no_progress` under its separate
v0.2 bundle contract.

This is a contract join, not a missing `if`:

- `prometheus.py` parses `vllm:generation_tokens_total`, but
  `collectors.py:METRIC_FIELDS` does not retain it in `MetricsObservation`.
- `external-runtime-observation-v1` has no generation-token or PID-start
  identity field. `process_snapshot()` reports PID survival, not PID reuse.
- Cadenced observations carry cached values between due times; only a source
  named in `sampled_sources` represents a fresh query. Collection errors must
  remain distinguishable from flat values.
- `select_metrics()` sums label variants, so today's aggregate token counter
  can rise while one DP engine's counter stays flat. At the pinned #53859
  base `22258a26`, vLLM metrics have `model_name` and `engine` labels
  (`loggers.py:469,705-711`). R3's admitted client stream describes one
  *request*. Neither an aggregate nor request fact silently establishes an
  engine-scoped verdict. R3 has no metrics series to score either counter.

For the admitted #53859 mechanism, the scoped choice has a source argument:
`ZmqEventPublisher.publish()` performs a blocking queue `put` in
`kv_events.py:391`; its only engine call site is
`Scheduler.update_from_output()` at `scheduler.py:2258`, inside an
EngineCore step. While that call remains blocked, the scheduler cannot
complete that step for the *affected engine*. This justifies an engine-level
progress test for V1, not a generic single-request detector. It does **not**
upgrade R3's one observed request to an empirical all-request or DP-wide
stall claim. The publisher is DP-rank-specific and stamps its event batch
with that rank; another engine can continue to produce tokens.

## Proposed invariant

An optional future trigger may fire only from a **recomputed** no-progress
claim over one bounded evaluation interval. The required inputs are the
same semantic facts as v0.2: operator-supplied process and endpoint targets,
stable process identity, process alive throughout, fresh all-2xx health,
flat *selected* progress producer, admitted continuous demand, adequate
sample count/span, and no higher-precedence process or health loss. It may
not fire from GPU utilization, a stack frame, a lone flat counter, a cached
sample, or a stored verdict string. Stack/GPU evidence may enrich the saved
artifact but cannot decide the trigger.

The chosen producer kind and **partition** are decisional because they fix
claim scope. In frozen v0.2, `server_counter` maps to `service` and its
label variants are summed; `client_request` maps to one `request`. A future
per-engine counter must declare a new `engine` scope in a versioned claim
contract. Reusing the v0.2 producer/demand evaluators does not authorize
silently relabeling their `service` verdict as `engine`. Producer
implementation (parser or client library) remains non-decisional. A
conflict with a present second producer must be retained, not resolved by
silently switching producers.

## Narrow candidate for a future version

Use a **versioned recorder observation**, not an extra optional field in
v1. The first candidate is a passive, **per-engine** counter trigger. It
must preserve each `(model_name, engine)` partition from one fresh metrics
scrape through the evaluation interval; `select_metrics()`'s summed output
is not an eligible input. Within each partition, retain finite,
nonnegative generation-token totals and running/waiting request counts
from the *same scrape*, with monotonic sample time and an explicit
collection outcome. The exact label tuple and its stable mapping stay
private; the public artifact may expose a run-local ordinal plus a
non-correlatable identity digest. A changing label set, missing partition,
counter reset, or mismatched demand partition yields insufficient evidence
for that engine rather than inheriting another engine's progress.

The engine label alone is not a process identity. A positive
`alive_health_ok_no_progress`-like engine claim requires an explicit,
provenance-recorded binding from that label partition to the
operator-supplied EngineCore PID and stable process start time throughout
the window. This binding remains an operator assertion unless independently
verified; absent or unstable binding blocks the *alive* claim. Fresh 2xx
health is endpoint responsiveness, not proof that this engine is healthy.
The pinned source offers a non-attaching binding candidate:
`core.py:1298-1301` requests the process title `EngineCore_DP{dp_rank}`
(or `EngineCore` without DP). But `system_utils.py:190-193` silently skips
that title when `setproctitle` is absent. A title observed in cmdline is
provenance to cross-check against PID start identity, not identity proof;
a missing title is **binding unavailable**, never “engine absent.” The
metric `engine` label-to-`dp_rank` correspondence must be verified for the
exact build and launch mode, including multi-API-server layouts, before
this candidate can decide a claim.
The v0.2 producer/demand state evaluators may be reused on each partition;
scope selection, precedence, and public claim shape need a separately
versioned rule and field-role audit. No second unreviewed verdict algorithm
may emerge inside `recorder.classify()`.

This candidate does **not** claim to detect a single stuck request while
*its own engine* continues to make tokens. That is a different mechanism
from #53859's blocking scheduler call. Request-scoped automatic capture is
a separate opt-in design: an already admitted, explicitly authorized stream
must span the whole interval; private request content stays private. Do
not make a background active inference probe the default of `record`.

Before choosing between an in-recorder rule call and a companion v0.2
`collect` witness, compare their identity and clock costs. A companion
witness is acceptable only with a verifiable same-target/same-boot time
join, window overlap, bundle digest, and explicit delayed-event outcome.
Filename proximity or two operator assertions are not enough. The
in-recorder candidate avoids that cross-process join but requires a new
schema and field-role audit. The architecture choice is **open** pending
the CPU matrix below.

## Capture and re-arm semantics

The recorder must retain enough *fresh* history to cover the evaluation
interval and the pre-fault context. If the ring overwrote the needed
interval, the semantic verdict may still be calculable from another
producer, but the saved artifact must not claim a complete pre-fault ring.
Collection end, trigger time, evaluation bounds, engine ordinal and its
binding, process start identity, source cadence, and writer outcome must be
distinguishable.

A positive transition captures at most once per continuous no-progress
episode. The existing per-reason cooldown alone is insufficient: a long
stall would otherwise cause repeated expensive captures. Re-arm only after
fresh observed progress (or a new, identity-bound run) **and** cooldown;
if recovery evidence is missing, stay latched. Process loss and health loss
keep their current higher precedence. An unavailable writer or exhausted
artifact budget is a typed capture failure, never a successful incident.
No automatic native attach or Flight Recorder dump is implied by this gate.

## CPU-only falsification matrix to approve before code

Each row needs a fixed input timeline and an expected trigger/non-trigger;
the future test must mutate the named fact and fail when the assertion is
removed. Reuse the v0.2 progress vectors directly where shapes permit.

| Case | Expected result and claim limit |
| --- | --- |
| Flat counter, no admitted work | No V1 trigger; idle is not a hang. |
| One engine's fresh flat counter, continuous same-engine demand, stable bound PID start identity, all-2xx health | One engine-scoped V1 trigger after the full configured window; no request-wide or DP-wide inference. |
| One engine flat with demand while other engines' token counters rise | Trigger only for the flat engine if its binding and other conditions hold; the aggregate's positive delta must not suppress it. |
| Cached-only flat metrics or one fresh sample | No trigger; insufficient evidence. |
| Missing/malformed `/metrics` or sampling timeout | No trigger; producer unavailable or insufficient, not flat. |
| Counter reset, label-set discontinuity, missing engine partition, or changed engine/PID binding | No trigger under the old window; do not stitch unlike counters. |
| Process title absent or inconsistent with the metric partition, with no other accepted binding | No engine-alive trigger; mark binding unavailable, not engine absent. |
| One stuck client request while its engine's counter rises | No engine-scoped trigger; preserve producer conflict if the client probe was explicitly supplied. |
| Health was never 2xx, later health loss, PID disappearance, or PID reuse | No V1 trigger; apply higher-precedence/undetermined outcome as appropriate. |
| Demand briefly vanishes or spans less than the evaluation interval | No trigger; demand evidence is discontinuous or insufficient. |
| Prolonged flat episode beyond cooldown | One capture only; no repeated attach/write storm. |
| Fresh progress resumes then a second complete flat episode occurs | Re-arm only after progress and cooldown; a second capture is permitted. |
| Ring truncation, writer failure, or budget exhaustion | Explicit incomplete-history/capture-failure outcome; never a complete successful artifact. |

No R3-to-v2 recorder projection can satisfy the fresh-metrics rows from the
published R3 data: that series was not retained. A CPU fixture can test the
rule, but cannot convert R3 into a metrics-based baseline. Any claim that
existing vLLM metrics would or would not catch that run requires a new
same-trigger comparison with the exact runtime build and preserved samples.

## Decision gates and stop rule

1. Freeze the target/scope, observation schema version, closed producer
   outcomes, leaf-field roles, same-scrape and same-clock semantics, and
   privacy allowlist before changing recorder code.
2. Make the CPU matrix pass and mutation-fail on each decisive fact. Run
   existing v0.2 contract/replay tests unchanged. A change to their verdict
   is a regression or a separately reviewed contract revision.
3. Demonstrate one named failure distinction that the *current recorder*
   misses and the candidate captures, while crediting existing per-engine
   metrics as the underlying signal. The Lab's proposed increment is the
   demand/identity rule and bounded capture, not invention of that metric.
   No new GPU booking solely to make this design look complete.
4. If the distinction is already available through retained existing DFX
   data, or if identity/demand cannot be bound without an invasive default
   probe, record **no integration needed yet**. Keep `collect/verify` as the
   explicit path and do not expand the public recorder schema.

The source answers the earlier service-versus-request choice for this V1
mechanism: a blocking scheduler call warrants an *affected-engine* scope.
The remaining review question is whether per-engine metric partitions can
be bound to stable EngineCore process identity and continuous same-engine
demand cheaply enough for an opt-in recorder trigger. A negative answer is
a valid design result, not a reason to weaken the evidence rules.

## 中文审阅摘要

现有 recorder 有事前环形历史，却不会被“进程仍在、健康检查 2xx、但有需求时
token 不前进”自动触发。#53859 的阻塞发生在 engine 的 scheduler step，因而
未来候选应按 **engine** 而非全部 engine 的 counter 总和评估。新版本 observation
需保存同一 engine、同一次 fresh metrics scrape 的 token 与需求计数，并显式绑定
EngineCore PID 启动身份；一个 engine 卡住、其他 engine 前进时，只能触发该
engine 的有界结论。不能借此宣称 R3 已实测所有请求停滞，也不能用其缺失的
metrics 序列证明 Prometheus 失效。先审 CPU 负对照、历史完整性、一次故障只
捕获一次及降级结果；若绑定无法建立，结论就是暂不加触发器。
