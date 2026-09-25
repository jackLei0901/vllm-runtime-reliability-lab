# #53859 existing-DFX versus Lab evidence ablation — 2026-09-24

Decision: **the retained case proves a health-only blind spot and a stronger
base/fix evidence claim; it does not prove that vLLM metrics could not detect
the stall.** This is an offline ablation of retained evidence, not a new GPU
run, operator study, or measured comparison of deployed tools.

## Fixed inputs and rule

- Case: `vllm-project/vllm#53859` and proposed fix `#53883`. The lab
  independently validated this reported issue; it did not discover it.
- Published input: Stage 1 R3 four-cell summaries and `replay-manifest.json`
  under `results/vllm-zmq-backpressure-stage1-r3-20260916/`; the source/result
  narrative is `experiments/vllm-zmq-event-backpressure/STAGE1_R3_RESULT_2026-09-16.md`.
- Lab checkout audited at `8c6b9d788fe283d6011441c44f2a28d15621376c`.
  Base vLLM commit: `22258a26bc090bccf5473cf681bbe9bac41bd035`;
  proposed-fix head: `1a2b85306b6d13033bbecc693e6cb776acb4bcaa`.
- `vllm-dfx replay results/vllm-zmq-backpressure-stage1-r3-20260916`
  reproduced its published verdict locally on 2026-09-24 using the checkout's
  CLI entry with `src` on `sys.path`. This rechecks archived evidence only.

Each row below reveals one more existing retained input. A row may use only
the facts named in its input column; it may not borrow a conclusion from a
later row. “Not scored” is not evidence of inability.

| Input allowed | What it supports in this case | What it does not support | Status |
| --- | --- | --- | --- |
| Repeated `/health` only | The API endpoint answered 2xx during the injected interval. | That an admitted request advanced, EngineCore was alive, or inference was healthy. | **Observed:** affected cell records `health_during_stall=2xx`. |
| `/health` plus vLLM Prometheus counters | A fresh token counter and admitted-work signal *could* reveal aggregate no-progress if their identity, cadence, and window suffice. | A request-specific stall from an aggregate counter; an internal queue cause; the four-cell fix trade-off. | **Not scored:** R3's published summaries retain no `/metrics` sample series, so no metrics-only verdict can be recomputed. |
| `/health` plus admitted client stream | The affected request emitted 11 progress events before release, then no content-bearing progress during the held interval; it resumed after release. | That every concurrent request stalled, or why this request stopped. | **Observed:** client/request scope only. The private request content is not published. |
| Above plus EngineCore identity | The campaign bound the real EngineCore and observed it alive while the request stalled. | That the endpoint and PID are literally the same process; process survival alone is not useful work. | **Observed under the frozen campaign relation**, not automatic process discovery. |
| Above plus bounded external stack | The sampled EngineCore execution point matched publisher → queue `put` → condition wait during the held window. | A stack snapshot alone proving service-wide no-progress or downstream event delivery. The test wrapper frame is not upstream code. | **Observed attribution**, not an independent progress verdict. |
| Four-cell base/fix contract, counters, identities, and verifier | Both controls completed; the base/pause cell stalled then recovered; the proposed-fix/pause cell completed 64 tokens with one accepted and four dropped event batches. File and source identities are checkable offline. | Merge acceptance, DP-wide impact, production drop rate, or reliable event delivery. | **Observed, replayed bounded claim.** |

The separate Stage B v2 py-spy/PyStack A/B/A2 result strengthens the
version-bound `queue_wait` attribution on a restored RTX 4090 environment.
It is **not** another row of the same R3 input ablation: it has its own run,
identity, normalizer, and negative control. See
`results/native-stack-pair-stage-b-v2-20260922/README.md`.

## Honest comparison with existing DFX

Current vLLM documentation offers metrics for request load and token output,
and its troubleshooting guidance offers more frequent stats, detailed logging,
and function tracing. These are genuine capabilities, not straw men:

- https://docs.vllm.ai/en/latest/usage/metrics/
- https://docs.vllm.ai/en/latest/usage/troubleshooting/

The documented metric names can evolve; the latest docs are not a substitute
for the exact `22258a26` runtime's exported metric samples. The published R3
files contain client progress offsets, health classification, a stack match,
EngineCore binding, counters, and source identities, but not a fresh metrics
series. Consequently this review **cannot** claim Lab detects a stall that
Prometheus could not detect. It can claim that `/health` alone missed it, and
that the Lab's additional value in this run is explicit request demand and
scope, process/endpoint relation, fault-window stack attribution, negative
controls, source/file integrity, and the liveness-versus-event-loss trade-off.

The published summaries also do not retain every raw sampler datum. The
privacy-bounded result verifies the frozen campaign's claims; it is not a
general proof that any third party can reconstruct every internal sample from
the public JSON alone. Raw logs, stack text, and request content remain private.

## What would settle the metrics comparison later

Only if this comparison changes an upstream decision, repeat the **same**
existing trigger under a preregistered protocol that retains privacy-bounded,
fresh `/metrics` samples for the exact vLLM build, plus the same client
progress/health/process observations. Score two predeclared questions
separately: (1) can metrics detect aggregate or request-scope no-progress,
and (2) can they attribute the publisher queue wait and the event-loss
trade-off? A metrics-only pass on (1) must be credited to existing vLLM DFX.
No new fault area, generic metric collector, or GPU booking is authorized by
this review.
