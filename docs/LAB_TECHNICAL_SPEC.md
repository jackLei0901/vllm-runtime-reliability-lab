# Lab technical specification: evidence rules at runtime boundaries

Status: local draft, 2026-09-28; not approved or published. The
[Chinese version](LAB_TECHNICAL_SPEC.zh-CN.md) is authoritative; this is its
English counterpart. It responds to [LAB_REQUIREMENTS](LAB_REQUIREMENTS.zh-CN.md)
and changes no v0.2 verdict, historical score, or Q4 preregistration.

## 1. Question this specification answers

Is the Lab's current technical architecture sufficient for its goal? The goal:
someone other than the original author, facing a real vLLM/PyTorch runtime
fault, reaches a more reliable judgment than ordinary logs or an ordinary
reproduction would give, because a verifiable evidence rule tells them what an
observation proves, what it does not, and what to check, test, or fix next.

**Answer:** existing components can support a bounded, single-target progress
case or a case-specific validation, but they do not cover every runtime
boundary. The first named external candidate determines whether that coverage
is sufficient. Before delivery, it needs a frozen comparison against an
ordinary reproduction and current source pins. An executable termination check
is conditional on a termination candidate; it is not authorized here. No
framework layer, new classifier, or new collector is specified. See §5.

## 2. Top-down analysis: where fault signals change meaning

A vLLM server running on PyTorch is a multi-process stack. Each layer produces
signals for its own purpose; a different layer, or a human, consumes them.

| Layer | Owner and main signal producers | What its signals are for |
| --- | --- | --- |
| L5 Frontend | API server, `AsyncLLM` output handler, `/health`, metrics, launcher watchdog | Serve requests; report whether the engine client is errored |
| L4 EngineCore | Busy loop, scheduler, KV-cache manager; `ENGINE_CORE_DEAD` | Step the model; report its own death |
| L3 Executor | `MultiprocExecutor` worker monitor, RPC deadlines, shared-memory ring buffer | Dispatch steps to workers; detect worker exit or RPC timeout |
| L2 Worker | Model runner, CUDA graphs, custom ops, `torch.compile`, profiling/warm-up runs | Execute kernels on real and synthetic inputs |
| L1 PyTorch runtime | c10d `ProcessGroupNCCL` watchdog and heartbeat monitor, Flight Recorder, caching allocator | Detect collective timeouts; dump collective history |
| L0 CUDA/NCCL/driver | Sticky CUDA errors, NCCL async errors, NCCL RAS | Report device and communicator failure |

Source facts for L3–L5 come from the
[liveness inventory](ENGINE_LIVENESS_CONTRACT_INVENTORY_2026-09-27.md) (signals
S1–S11, budgets T1–T14, vLLM pin `c8602c79`). Facts for L1 come from the
[c10d lifecycle deep dive](reviews/C10D_LIFECYCLE_DEEP_DIVE_2026-09-23.md)
(PyTorch pin `a339711`).

The following selected lifecycle and diagnostic cases involve a signal crossing
a boundary and a consumer inferring more than its producer established. This is
not a claim about every Lab finding:

| Tempting inference | What the signal actually asserts | Lab evidence |
| --- | --- | --- |
| `/health` 2xx means requests progress | The engine client is not errored (S6/S1). No progress signal exists (S10 absent). | [#53859 R3 bundle](../results/vllm-zmq-backpressure-stage1-r3-20260916/), replayed offline from a fresh venv |
| Top-level exit 0 means an intentional, clean stop | Before #52178, EngineCore SIGKILL could also end with exit 0 | [fault-recovery](../experiments/fault-recovery/README.md) base/fix table |
| A shutdown timeout bounds the whole teardown | Outer grace can be shorter than the nested inner schedule (T7–T10) | [K1/K2](reviews/ENGINE_LIVENESS_K1_K2_CPU_RUN_2026-09-27.md): real vLLM functions, fake process tree, finite tested values only |
| A missing Flight Recorder dump means the rank is absent | The dump responder can be unavailable during communicator destruction | [c10d validation](../experiments/pytorch-c10d-shutdown-dump/UPSTREAM_FIX_VALIDATION_2026-09-16.md), PyTorch #197232 |
| Matching versions mean the same runtime | Build and process identity must be bound separately | [Runtime identity](MATCHING_VERSIONS_ARE_NOT_RUNTIME_IDENTITY.md) |

In the Q4 keyword-limited sample, 30 of 40 eligible reports were labelled
`outside_model` in the AI first pass; four more were `insufficient_information`.
All 30 `outside_model` entries carry `fault_domain=leaf`, as guide v2 requires,
but these AI labels have not been audited item by item. They do not establish
that the faults were in fact leaf-level or that their errors were
self-explanatory. Human agreement remains unscored, and these counts are not a
prevalence estimate. The design choice follows the bounded Lab goal, not this
sample: **test what a cross-boundary signal permits a consumer to infer**,
without duplicating mature leaf-level capture tools.

## 3. Reliability levels (direction, not commitment)

| Level | Content | Lab scope |
| --- | --- | --- |
| 0 | Raw signal: log line, metric, exit code, dump | Baseline tools |
| 1 | Identity-bound observation: which process instance, when, how fresh; build identity separately pinned where available | Supported for specific recorded targets, not automatic discovery |
| 2 | Contract-checked fact: one contract, controls, closed outcomes, stated limit | **The Lab's layer** |
| 3 | Diagnosis rule combining several facts | Only after measured false-positive and false-negative rates on controlled cases; none claimed yet |
| 4 | Automated action on a diagnosis | Out of scope |

A higher level may consume only facts from the level below it, with their
limits carried along. Any automated consumer (alerting, diagnosis tooling) first
needs the Level 3 error-rate evidence. No current Lab artifact provides it.

## 4. Technical components

These are existing tools and evidence practices used by separate checks, not
one implemented pipeline or a proposal for a new layer. §5 lists only
candidate-specific work.

| Component | Responsibility | Current implementation |
| --- | --- | --- |
| C1 Signal-semantics source maps | Per signal: producer, consumer, what it asserts, forbidden inference, source pin | Liveness inventory (S1–S11, T1–T14); c10d deep dive. Not a software registry. |
| C2 Capture and identity | Bounded observation and, when supplied, process start identity; build pinning is a separate case record | `collectors`, `recorder`, `collect_bundle`, `native_producers`, `stacks`; endpoint–PID relation remains operator asserted. |
| C3 Packaging and redaction | Closed-shape bundles with digests; private material excluded | `bundle`, `collect_bundle`, `verify_bundle`, `schema`, `external_schema` |
| C4 Contract checks, kept separate (R7) | Each judges one contract with controls and closed outcomes | Progress: `progress.derive_verdict`. Termination propagation: `experiments/fault-recovery`. Lifecycle budgets: K1/K2 scripts |
| C5 Offline checks | Re-derive a supported result without rerunning its GPU campaign | `verify_bundle` for v0.2 collect bundles; `replay` is hard-coded to #53859 R3. The clean-venv check used a locally built wheel after direct `pip install .` failed at the package proxy. |
| C6 Delivery checklist | State what a non-author receives in one thread | Existing review entries are examples; §5 G1 is a per-candidate checklist, not a new component. |

The checks reuse evidence practices where applicable; they do **not** all share
the same identity, freshness, redaction, replay implementation, or decision
logic. No component combines their verdicts.

## 5. Fit/gap against the requirements

| Need | Status | Gap and condition for closing it |
| --- | --- | --- |
| A non-author can inspect or run the material without asking (value test) | Partial: #53859 replay works offline; K1/K2 need Linux and pinned vLLM functions; the termination result's raw evidence is private | **G1 Candidate-specific delivery checklist** below. A checklist alone does not establish independent use. Close before delivery. |
| Signal claims are current (R3, R6) | Inventory rows are pinned to `c8602c79`; upstream has moved | **G2 Re-pin on use:** before a delivery cites a row, re-check it at the thread's commit and record the new pin or mark it `unverified`. Procedure, not tooling. |
| Closed outcomes with an apparatus control (R4) | Progress: `undetermined`. K1/K2 have case-specific apparatus outcomes. Termination: base/fix exit-code tables, not a common verifier | **G3 Conditional termination check:** only if a named external termination case cannot be served by the existing table and tests. Freeze mutually exclusive `propagated`, `not_propagated`, or `unscored` (with `apparatus_failed` as a reason), plus the SIGTERM control; require a failing test first. |
| One contract per check (R7) | Satisfied by keeping C4 separate | None |
| PyTorch-side boundaries | c10d covered by one case; no general PyTorch capture | None now. A new PyTorch boundary enters through a ten-hour boundary study, not a new component. |

**G1 delivery checklist.** Before a decisive run or public draft, freeze items
1–3 and the outcome that would refute the incremental claim. Each delivered
artifact then states:
1. The thread and the decision it faces.
2. What an ordinary reproduction would establish. This is the frozen comparison from the requirements.
3. The claim, its evidence grade, and the one contract checked.
4. Build and runtime identity; any re-pinned signal rows.
5. A command a non-author can run, or the exact files to inspect.
6. Positive and negative controls, and the closed outcome set, including `unknown`/`unscored`.
7. Where the evidence stops, one plausible counterexample, and the predeclared
   result of the incremental comparison.
8. The target project's current contribution/AI-disclosure rules, privacy and
   link check, and post-publication readback. These are human checks, not a
   claim that `verify_bundle` enforces publication safety.

## 6. Acceptance and non-goals

**Technical-fit test:** walk the first named candidate (due 2026-10-12)
through C1–C5 and G1/G2 before any build. Record whether existing tools can
produce the predeclared incremental fact and a runnable or inspectable artifact.
If not, record `unsupported` and its precise missing fact; do not add a
component by default. G3 is available only under its condition. Passing this
test says the architecture can deliver that *one* case, not that an external
person used it or that the Lab's overall goal has been met.

**Non-goals:** a framework layer joining checks; a new classifier or taxonomy
expansion; automatic `src/dfxlab` growth; Level 3–4 claims; acceptance measured
by the number of features, documents, or checks. Personal impact may follow from
sustained useful contributions but is not an acceptance metric.

**Stop rule:** follow the requirements. If repeated delivery into active
threads produces no observable use, distinguish review delay from feedback
that the artifacts were unnecessary before deciding on maintenance mode. Do
not respond with a larger architecture.
