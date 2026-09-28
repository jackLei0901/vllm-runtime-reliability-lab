# Lab runtime model: serving process liveness

Version: **0.1.0-draft** (2026-09-27). Chinese companion:
[RUNTIME_MODEL.zh-CN.md](RUNTIME_MODEL.zh-CN.md). Change history:
[CHANGELOG.md](CHANGELOG.md).

This is the Lab's **descriptive** model, not vLLM's agreed public contract.
Its primary source pin is vLLM
[`c8602c79062440074a018c1d5f875a5571eb6881`](https://github.com/vllm-project/vllm/commit/c8602c79062440074a018c1d5f875a5571eb6881).
The [source inventory](../ENGINE_LIVENESS_CONTRACT_INVENTORY_2026-09-27.md)
gives the detailed S1–S11 and T1–T14 source map and evidence grades. A later
read-only check at [`2407f405`](https://github.com/vllm-project/vllm/commit/2407f405b51abd23adbf0203b98464f448c58edf)
confirmed the C1-related `utils.py`, CLI `serve.py`, `core_client.py`, and Rust
managed-engine files have the same Git blobs as the preceding `55de40a2fc`
check. **That does not refresh every M2–M6 claim against current main.**
Experiments keep their own source pins, scores and limitations.

The model covers the online serving process tree and the boundary between
life, responsiveness, useful work and shutdown. A GPU kernel, allocator or
collective is an opaque worker-stage leaf unless a separate producer proves
more. Model IDs are Lab-internal references, never v0.2 field names or fault
categories.

## M1 — topology and observation ownership

| ID | Role and observed holder | Scope/source |
| --- | --- | --- |
| M1.API | API server and `AsyncLLM` hold the client-visible engine-dead/output-handler state. | [Pinned `async_llm.py`](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/v1/engine/async_llm.py#L1086-L1096), inventory §1. |
| M1.CLIENT | `MPClient` holds EngineCore sentinel-monitor state. | [Pinned `core_client.py`](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/v1/engine/core_client.py#L708-L733); not the headless CLI parent. |
| M1.CORE | `EngineCoreProc` owns busy-loop/shutdown state. | [Pinned `core.py`](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/v1/engine/core.py#L1378-L1389). |
| M1.EXECUTOR | Uni- and multiprocess executors have different wait/detection contracts; a multiprocess monitor observes worker sentinels. | [Pinned `multiproc_executor.py`](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/v1/executor/multiproc_executor.py#L286-L305); [uniproc timeout rejection](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/v1/executor/uniproc_executor.py#L32-L34). |
| M1.WORKER | Worker process owns its shutdown request and device work; a stack or host marker is not a device-progress proof. | [Pinned `multiproc_executor.py`](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/v1/executor/multiproc_executor.py#L795-L841). |
| M1.PARENT | Launcher or CLI parent owns outer tree shutdown. Headless and multi-API modes do not use the same `MPClient` path. Rust wraps a headless Python parent. | [Current headless and multi-API entry paths](https://github.com/vllm-project/vllm/blob/2407f405b51abd23adbf0203b98464f448c58edf/vllm/entrypoints/cli/serve.py#L253-L260); [multi-API budget](https://github.com/vllm-project/vllm/blob/2407f405b51abd23adbf0203b98464f448c58edf/vllm/entrypoints/cli/serve.py#L398-L413); [Rust child command](https://github.com/vllm-project/vllm/blob/2407f405b51abd23adbf0203b98464f448c58edf/rust/src/managed-engine/src/process.rs#L57-L75). |

DP supervisor, coordinator, Ray and opt-in FT are named topology variants,
not inferred to be equivalent to the default path. The
[inventory](../ENGINE_LIVENESS_CONTRACT_INVENTORY_2026-09-27.md) covers only
the bounded local topology; multi-node behavior remains outside this version.

## M2 — lifecycle axis

The pinned EngineCore has an owned shutdown state and handling path
([`core.py`](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/v1/engine/core.py#L1378-L1389),
[`core.py` shutdown handling](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/v1/engine/core.py#L1460-L1504)).
The model records these transition classes without claiming one code enum:

| Transition class | Owner/observable boundary | Status |
| --- | --- | --- |
| startup → ready | client/EngineCore handshake and readiness wait | source-observed; see inventory T1/T2 |
| ready → intentional shutdown | parent signal/request, then EngineCore and executor teardown | source-observed; entry-path budget varies (M5) |
| ready → terminal failure | process exit, EngineCore exception or worker failure | source-observed; propagation varies (M6) |
| ready ↔ intentional pause/sleep | public pause/sleep controls can stop token production without fault | source-observed controls; **not** a universal shared state enum |

The proposed vocabulary `draining`, `stopping`, `suspect` and their public
semantics belong in the [contract outline](../ENGINE_LIVENESS_RFC_OUTLINE_2026-09-27.md),
not in this source fact table. In particular, a lack of tokens during an
intentional pause is not itself a health failure.

## M3 — health and progress facts

| Fact ID | Meaning and scope | Producer/availability at this pin |
| --- | --- | --- |
| M3.ALIVE | a named process is live now | process sentinel or external `/proc`; an old log line alone is insufficient. External Lab process snapshots do **not** capture PID start ticks by themselves. |
| M3.IDENTITY | observation belongs to the same process/engine over the bounded window | requires a separately recorded PID/start identity and binding; missing binding is `unobserved`, not `engine_missing`. |
| M3.LOOP | EngineCore loop answered a request | not provided by default [`/health`](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/v1/engine/async_llm.py#L920-L923); [#36451](https://github.com/vllm-project/vllm/pull/36451) is a proposal, not an existing fact producer. |
| M3.DEMAND | work was admitted and should be advancing | must be scoped to request or engine and exclude idle/paused/remote-KV/dummy-batch controls. |
| M3.PROGRESS | useful work advanced within an observation window | Lab `collect/verify` can score bounded progress; current recorder does not trigger on S10. Summed metric labels cannot prove a particular engine advanced. |
| M3.UNOBSERVED | a required producer is absent, stale or conflicting | an evidence state, never evidence that the target is absent or healthy. |
| M3.TERMINAL | failure was confirmed by an owned error/process exit | [sentinel/output paths](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/v1/engine/core_client.py#L708-L733); distinct from no progress. |

M3 facts are not verdicts. The frozen five-value `progress.py` vocabulary and
precedence remain unchanged. Its mapping to M3 is documented in the
[architecture proposal](../LAB_ARCHITECTURE_PROPOSAL_2026-09-27.md), with
paused/sleeping and loop responsiveness explicitly identified as input gaps.

## M4 — signal path and producer outcomes

The [inventory §2](../ENGINE_LIVENESS_CONTRACT_INVENTORY_2026-09-27.md)
names S1–S11. The model keeps each signal's **condition, producer, transport,
consumer and availability** separate. At the pin: S1–S3 and S6 reach a
terminal/health path; S4 has a multiprocess RPC deadline; S5 and S10 have no
equivalent producer; S7 is implemented but unwired; S8/S9 are log-only; S11
propagates engine death to launcher exit. For example, worker exit takes
[executor sentinel → callback](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/v1/executor/multiproc_executor.py#L286-L305)
and then [EngineCore failure handling](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/v1/engine/core.py#L1534-L1535);
an unwired [`check_health`](https://github.com/vllm-project/vllm/blob/c8602c79062440074a018c1d5f875a5571eb6881/vllm/v1/executor/multiproc_executor.py#L505-L507)
is not a serving health signal. A producer returning `unavailable` must not
be scored as a negative target-state observation.

## M5 — time and shutdown budgets

The [inventory §3](../ENGINE_LIVENESS_CONTRACT_INVENTORY_2026-09-27.md)
pins T1–T14 individually. This model does not collapse them into one timeout:

| Budget group | Present behavior and required relation |
| --- | --- |
| startup (T1–T3) | client handshake and coordinator waits have independent owners; they are not a serving-progress window. |
| per-step/health (T4–T6) | multiprocess RPC default T4 is 300 s; uniproc has no equivalent deadline. T5 is unwired; T6 has no reader at the pin. |
| request drain/parent process (T7–T8) | explicit `shutdown_timeout=0` is not `None`: in the Python shutdown helper it bypasses the 5 s fallback, sends SIGTERM then proceeds to force-kill live children. Headless/multi-API signal paths also pass zero. |
| worker and delivery (T9–T11) | background outer wait `x` and inner worker wait `x` leave no reserved room for the inner 4 s SIGTERM escalation or delivery margin; K2 tested selected `x` values on a fake tree. |
| monitoring/FT (T12–T14) | launcher polling, warning and opt-in FT recovery are distinct deadlines, not evidence of an EngineCore progress budget. |

At the [current refresh](https://github.com/vllm-project/vllm/blob/2407f405b51abd23adbf0203b98464f448c58edf/vllm/v1/engine/utils.py#L46-L69),
ROCm alone grants a separate 15 s EngineCore cleanup window when both request
and process timeouts are zero. This is a source fact and a C1 design question
for CUDA, not a chosen new CUDA default. The four entry paths and the
signal/non-signal distinction are in the [outline budget table](../ENGINE_LIVENESS_RFC_OUTLINE_2026-09-27.md).

## M6 — failure and shutdown propagation

| Boundary | Observed state at pin | Lab case/limit |
| --- | --- | --- |
| worker → EngineCore | multiprocess worker exit becomes executor failure; a live stuck uniproc worker has no equivalent deadline | C3; [K5](../reviews/ENGINE_LIVENESS_K5_DUAL_GPU_RESULT_2026-09-27.md) scored TP=1 only |
| EngineCore → client/API | sentinel or error output can set `engine_dead`, which `/health` observes; no-progress S10 cannot | V1, C3/C4; no default loop ping |
| API/parent → supervisor | exit code may lose a fatal cause; intentional SIGTERM must remain distinguishable | V2/[#52178](https://github.com/vllm-project/vllm/pull/52178); do not claim PR merged |
| DP supervisor → caller | child failure may not survive the supervisor exit boundary | V3 Gate 0 only; full-package Gate 1 pending |
| shutdown → client | request abort output and parent cleanup can have different budgets | C1/C2/C8; [#36964](https://github.com/vllm-project/vllm/pull/36964) owns prior abort-output work |

The timeout text in K5 named the `sample_tokens` future being awaited while
the injected worker held `execute_model`; that arm remains formally unscored.
It is a diagnosability example for [#54638](https://github.com/vllm-project/vllm/pull/54638),
not a separate new upstream defect.

## Upstream vocabulary adapter (documentation only)

| Lab fact | Upstream vocabulary | Boundary |
| --- | --- | --- |
| M3.TERMINAL | `engine_dead`, `EngineDeadError`, opt-in FT `DEAD` | same word does not imply identical ownership or timing |
| non-terminal suspect derived from M3 | opt-in FT `UNHEALTHY` | proposal only for default serving; FT scope differs |
| M2 shutdown + M3 reachability | `/health` and proposed `/live` in [#36258](https://github.com/vllm-project/vllm/pull/36258) | no endpoint change in Lab |
| M3.LOOP | health ping in [#36451](https://github.com/vllm-project/vllm/pull/36451) | proposal; loop response is not useful-work progress |
| M5 drain/grace | [#24885](https://github.com/vllm-project/vllm/issues/24885) semantics and ROCm [#52281](https://github.com/vllm-project/vllm/pull/52281) | the former closed stale, the latter platform-limited |

An upstream naming decision changes this adapter, not M1–M6 or historical
Lab results. Intake can assign an issue to an existing M-part or record a
`model_gap`; neither action changes the fault taxonomy without its own
admission gate.
