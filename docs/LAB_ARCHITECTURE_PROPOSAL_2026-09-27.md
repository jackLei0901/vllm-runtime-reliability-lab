# Lab architecture proposal: a runtime model as the backbone

Chinese companion: [中文版架构提案](LAB_ARCHITECTURE_PROPOSAL_2026-09-27.zh-CN.md).

Status: **Lab-only architecture work in progress**, 2026-09-27. The liveness
baseline was isolated in local commit `373f6dd`; this proposal is on branch
`lab/runtime-model` and is not published. Nothing here changes a published
contract, verdict, schema or result. It does not depend on any upstream response to the
[liveness RFC outline](ENGINE_LIVENESS_RFC_OUTLINE_2026-09-27.md).
The [complete migration map](model/ARCHITECTURE_MAPPING_2026-09-27.md) lists
every current `src/dfxlab` Python module, experiment directory and top-level
Lab document, including items intentionally outside the runtime model.

## 1. Decision requested

Adopt one new organizing layer, a versioned **runtime model** of the vLLM
serving process tree, and index the Lab's taxonomy, verdicts, experiments and
issue intake against it. Everything the Lab already has stays in place. The
change is what those things are organized by: model elements instead of cases,
issues and dates.

This is the "top-down framework plus bottom-up patches" approach. The model
is the framework, and each issue either lands on a model element or exposes a
gap in the model.

## 2. Why the current structure is not enough

The [roadmap](../PRODUCT_ROADMAP.zh-CN.md) §1 ranks the Lab's value as taxonomy
→ domain interpretation → verdict contract → evidence standard → collector.
Each layer exists. What is missing is a shared description of the **system**
they are about. The symptoms are already recorded in Lab documents:

| Symptom | Where it is recorded |
| --- | --- |
| The taxonomy has no coverage denominator. Categories are admitted per case, so there is no way to say which runtime boundaries are unexamined. | [Fault taxonomy v0](FAULT_TAXONOMY_V0_2026-09-24.md), "What this taxonomy does not answer yet" |
| Producers and verdicts do not meet. The recorder triggers on exit, health loss, KV pressure and preemption, but not on the V1 no-progress verdict that `collect/verify` can reach. | Fault taxonomy v0, "Actionable V1 gap" |
| Four vocabularies describe overlapping facts with no mapping: `progress.py` verdicts, taxonomy V1–V3/X1–X2, liveness health facts and C1–C8, and upstream's FT `UNHEALTHY`, `/live` and #24885 terms. | `src/dfxlab/progress.py`; the liveness outline §3 |
| The 15 experiment directories are named by case or issue. They mix vLLM and PyTorch work, active and paused lines, and alpha-era stubs that were never run (`oom-boundary`, `preemption`, `soak`). | `experiments/` |
| Dozens of top-level and review documents are dated files with no index by subject. Finding "everything about shutdown budgets" needs a text search. | `docs/`, `docs/reviews/` |

The liveness inventory showed the missing piece. Once processes, signals,
budgets and transitions were written down with owners, the findings
(C1–C8), the upstream map and the experiments (K1/K2/K5) all fell into place.
This proposal generalizes that inventory into the Lab's backbone.

## 3. The runtime model

One document, `docs/model/RUNTIME_MODEL.md`, derived from the
[liveness inventory](ENGINE_LIVENESS_CONTRACT_INVENTORY_2026-09-27.md), pinned
to a vLLM commit, with a drift column for current `main`. It has six parts with
stable IDs:

| Part | Contents | Seeded from |
| --- | --- | --- |
| **M1 Topology** | Process roles and their owners: API server or Rust frontend, CLI parent (single, headless, multi-API), DP supervisor, Rust managed-engine parent, `MPClient`, EngineCore, executor (uni/multiproc/Ray), workers, DP coordinator. Launch modes are part of the element, not a footnote. | Inventory §2, outline §3 budget table |
| **M2 Lifecycle** | Source-observed state holders and transitions, each with its actual owner. Candidate names such as paused/sleeping and draining stay in the separate contract outline until their source mapping is explicit. | Inventory transitions, outline §3 |
| **M3 Health and progress facts** | Process alive and identity-stable; loop responsive; admitted demand; progress within a window; producer absent or stale; terminal failure. Each fact records producer availability, freshness and scope (request/service/engine); an absent producer is recorded as absent, not invented. | Inventory S-signals, `progress.py` states |
| **M4 Signals** | S1–S11: who produces each signal, who consumes it, and over which transport (RPC, sentinel, ZMQ, HTTP). | Inventory §3 |
| **M5 Budgets** | T1–T14 and their nesting relations, per entry path. | Inventory §4, outline budget table |
| **M6 Propagation** | How a failure reaches API status, logs and metrics, and exit status, per launch mode. | Taxonomy V2/V3, inventory C6 |

Rules for the model:

- **Lab-internal IDs only.** Upstream terms appear in a separate mapping table
  (§6). If maintainers choose other names, only the mapping changes.
- **Descriptive, never aspirational.** Each element states what the pinned
  source does, with line anchors. Proposed contracts live in RFC-style
  documents that *reference* the model. The liveness outline is the first of
  these.
- **Versioned.** Every revision has a dated changelog entry naming the issue,
  experiment or source drift that forced it. A model change never silently
  rescores an old result; results keep the model version they were scored
  against.
- **Bounded scope.** The model covers the serving process tree and its runtime
  lifecycle. Kernels, collectives, the allocator and model correctness sit
  *below* it and appear only as "opaque stage" leaves (for example, "worker is
  inside `execute_model`"). That is where the paused TP/NCCL line would attach
  if it resumes.

## 4. How existing work maps onto the model

This mapping is the acceptance test for M1–M6. The
[complete map](model/ARCHITECTURE_MAPPING_2026-09-27.md) accounts for every
current module, experiment directory and top-level Lab document. If something
cannot be placed, either the model is wrong or the item is outside its
bounded scope; neither outcome silently expands M1–M6.

### Taxonomy and liveness findings

| Item | Model element |
| --- | --- |
| V1 useful work stops behind a responsive endpoint | M3 progress under admitted demand; M4 S10 (absent) |
| V2 fatal cause lost at API exit | M6 EngineCore → API-process exit status |
| V3 child failure lost at DP-supervisor exit | M6 DP supervisor exit status; M1 supervisor role |
| X1 diagnostic producer disappears | M3 "producer absent", but cross-stack (PyTorch); stays in the X section |
| X2 local correctness presents as distributed hang | Out of model (below M1 leaves); stays in the X section |
| C1, C8 | M5 drain vs. process grace, per M1 entry path |
| C2 | M5 nested worker budget |
| C3 | M3 × M1 executor kind (detection differs by executor) |
| C4 | M4 unwired health signals |
| C5 | M4 RPC reply correspondence |
| C6 | M6 intentional shutdown reported as crash |
| C7 | M5 configured budget with no reader |

### Verdict code (frozen v0.2, documentation-level mapping only)

| `progress.py` verdict | M3 facts it asserts |
| --- | --- |
| `process_missing` | not (alive ∧ identity-stable) |
| `health_lost` | health surface lost after being observed ok |
| `progress_observed` | progress within window (request or service scope) |
| `alive_health_ok_no_progress` | alive ∧ identity-stable ∧ health ok ∧ demand present ∧ flat |
| `undetermined` | required facts absent, stale or conflicting |

`progress.py` is not changed. Two gaps show up immediately and become model
questions, not code changes. First, M2 lifecycle state is not an input: an
intentional `pause_generation(mode="keep")` would currently look like
`alive_health_ok_no_progress`. Second, "loop responsive" (the #36451 ping) has
no slot at all.

### Experiments

Each experiment directory gets a short status header (§5). Proposed
classification:

| Directory | Model cells | Status |
| --- | --- | --- |
| `engine-liveness-contract` | M5 (K1, K2), M3 × M1 executor (K5) | active |
| `vllm-zmq-event-backpressure` | M3 progress / M4 S10 (#53859, #36451) | active (Q4 Q1/Q2) |
| `vllm-engine-binding-gate` | M3 identity; M1 EngineCore | active (Q4 Q4) |
| `vllm-dp-supervisor-exit` | M6 supervisor exit (V3) | gate 1 pending |
| `fault-recovery` | M6 API exit (V2, #52178) | closed, upstream PR open |
| `native-evidence-capability` | attribution layer (not a model fact) | closed |
| `overhead` | Lab-tool cost, not a model fact | harness only |
| `vllm-tp-dfx` | below-M1 leaf: collectives | paused; closure doc pending |
| `organic-hang`, `pytorch-c10d-shutdown-dump`, `pytorch-unused-grad-dtype` | cross-stack X1/X2 | closed or waiting on #197232 |
| `vllm-mm-uuid-encoder-cache` | out of model (cache correctness) | draft |
| `oom-boundary`, `preemption`, `soak` | not placed; each would need admission | stub, never executed; propose archive status |
| `VLLM_CONCURRENCY_56251_ASSESSMENT` | M4 and M5 risk map | reference |

## 5. Repository changes (small, link-preserving)

1. **Add** `docs/model/RUNTIME_MODEL.md` (M1–M6) and
   `docs/model/CHANGELOG.md`.
2. **Add** `docs/INDEX.md`: one section per model part linking the relevant
   docs and reviews. Files are **not moved or renamed**, because published
   results and upstream comments link to current paths.
3. **Add a status header** to each experiment README, in a fixed shape:
   `model_cells`, `status` (active / paused / closed / stub / archived),
   `upstream_exit`, `last_scored`. Stubs are marked `archived` in place, not
   deleted.
4. **Add one CPU test** that parses the headers and fails if a header names a
   model ID that does not exist. This checks references, not that an experiment
   really supports the named fact; evidence review still owns that judgment.
   No model engine, registry service or new CLI command.
5. **Leave unchanged:** `src/dfxlab`, `external-runtime-observation-v1`, the
   v0.2 field roles and verdict precedence, `schema/`, `results/`, and all
   privacy rules.

## 6. Upstream vocabulary mapping

A table in the model document, updated only when upstream decides something:

| Lab model | Upstream term | Source and state |
| --- | --- | --- |
| M3 terminal failure | `engine_dead`, `EngineDeadError`, FT `DEAD` | pinned source |
| M3 non-terminal suspect | FT `UNHEALTHY` (opt-in, different scope) | #44428, merged |
| M2 draining vs. M3 fatal | `/live` vs. `/health` | #36258, open |
| M2/M5 shutdown semantics | "graceful quiescence", drain timeout | #24885, closed as stale |
| M3 loop responsive | EngineCore health ping | #36451, open |

This table is how the Lab absorbs whatever the RFC discussion decides without
restructuring again.

## 7. Issue intake: where bottom-up meets top-down

The Q4 plan already defines a private, deduplicated pain-point ledger. Add one
required field: **`model_cell`** (one or more M-IDs) or **`model_gap`** (a
short statement of what the model cannot express). Two consequences:

- A `model_gap` is the only way the model grows. At the monthly review, gaps
  either change the model (with a changelog entry), are declared out of scope,
  or stay open. The taxonomy admission gate is unchanged; a model change is not
  a new fault category.
- The planned n=40 coverage sample can label against **M-parts** as well as
  V1–V3. That turns "we have no denominator" into "these model parts received
  k of 40 sampled reports". This must be decided **before the frame is frozen
  in week 1 (09-28 to 10-04)**; adding labels after reading issues would break
  the preregistration.

## 8. Relationship to the Q4 plan (a deviation to acknowledge)

The Q4 plan says one PoC runs at a time and the #53859 loop closes before the
next PoC is chosen at the end-of-October review. The liveness work (inventory,
K1/K2/K5, RFC outline) started before that gate. It was source-first and
mostly CPU, and it turned out to be the model this proposal needs, but under
the plan's own rules it is an early second line. Record it as one of these:

- **(a)** the liveness line *is* the next PoC, admitted early, with the
  October review confirming or closing it; or
- **(b)** the liveness work is reclassified as model-building (architecture)
  rather than a PoC, and the October review still picks the next PoC.

This restructure adopts **(b)** as an internal planning classification: the
liveness inventory and K1/K2/K5 form source-anchored model-building evidence,
not a second active Q4 PoC. They keep their original scores and experimental
limits. The #53859 PoC gate and October candidate review remain unchanged.

Q4 items this proposal touches: Q3 (ledger field, sample labels), Q4 (the G0
identity rules become M3 identity), Q1/Q2 (re-indexed, not changed). Q5 and Q6
are unaffected.

## 9. Phases and exit criteria

| Phase | Work | Exit criterion |
| --- | --- | --- |
| P0 | Isolate the liveness evidence and tests after a publishability review. | Done locally in `373f6dd`; unrelated TP/timeline/user work remains untouched and uncommitted. No public push is implied. |
| P1 | Write `RUNTIME_MODEL.md` M1–M6 from the inventory, pinned, with a `main` drift column. | Every row of §4 placed or explicitly out of model; no element lacks a source anchor. |
| P2 | Experiment headers, `docs/INDEX.md`, the header-lint test. | The test passes; every experiment has a status. |
| P3 | Ledger field; n=40 label decision before the frame freeze. | Decision recorded by 10-04. |
| P4 | First monthly review using `model_gap`. | At least one gap resolved, declared out of scope, or kept open with a reason. Zero model changes is an acceptable outcome. |

Estimated effort: P1 is about two sessions of source-anchored writing, since
most content exists in the inventory. P2 and P3 are small. No GPU time.

## 10. How to tell whether this worked

These are falsifiable checks, reviewed at the end-of-quarter audit:

- A new runtime issue can be placed (cell or gap) in about ten minutes from the
  model and index alone, without re-reading source.
- "Which model parts have no experiment?" has a written answer, and it
  influences the next-PoC choice.
- At least one upstream comment or review cites a model-derived fact (for
  example, the M5 entry-path table) rather than a single experiment.
- The model changed at least once because of an issue that did not fit.
  A model that never changes is either too vague or being ignored.

If after the quarter the model is not consulted when choosing work, retire it
to reference status rather than maintaining it.

## 11. Non-goals

No new collector, schema version, automatic trigger, controller or restart
policy. No change to v0.2 semantics. No renaming of published artifacts. No
claim that the model is vLLM's contract. It describes source, and upstream
contracts are negotiated in upstream threads. Self-healing stays out of public
framing.

## 12. Open decisions for the owner

1. Label the n=40 sample by M-part as well as V1–V3: yes or no, before 10-04.
2. Archive status for `oom-boundary`, `preemption`, `soak`: agree, or name one
   to re-admit.
3. `RUNTIME_MODEL.md` gets a `zh-CN` companion, consistent with this proposal
   and the liveness outline; the exact release timing remains open.
