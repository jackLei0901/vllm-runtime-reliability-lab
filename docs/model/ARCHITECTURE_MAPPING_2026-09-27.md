# Runtime-model migration map

Status: **Lab-only architecture inventory**, 2026-09-27. Companion to the
[proposal](../LAB_ARCHITECTURE_PROPOSAL_2026-09-27.md), not a change to a
published verdict or artifact. This maps the files present in this checkout;
it does not rename or move any of them. `M1`–`M6` are the proposed model parts,
not fault categories or new v0.2 field names. `Evidence layer` means the file
supports capture, verification or presentation but does not itself describe a
vLLM runtime state. `Outside` is an explicit scope decision, not a deletion.

## `src/dfxlab`: every Python module

| Module | Current role | Proposed placement |
| --- | --- | --- |
| `__init__.py` | package metadata | Delivery; no runtime-model fact |
| `__main__.py` | command entry | Delivery; no runtime-model fact |
| `cli.py` | user commands and wiring | Delivery; links model-indexed work, no verdict change |
| `collectors.py` | cadenced external observation | M3 fact producers; declare scope/freshness in documentation |
| `prometheus.py` | metrics selection/projection | M3 progress/demand producer; summed labels cannot establish per-engine progress |
| `stacks.py` | bounded native stack capture | M4 diagnostic producer; stack alone is not progress proof |
| `native_producers.py` | native acquisition paths | M4 producer provenance and outcome |
| `native_evidence.py` | native lifecycle evidence rules | M2/M4 attribution; existing rules remain unchanged |
| `recorder.py` | ring and trigger policy | M3/M4 consumer; current triggers do not implement S10 no-progress |
| `progress.py` | closed progress verdicts | M3-facing **verdict layer**, not the runtime model itself; frozen precedence |
| `faults.py` | test fault injection | Validation harness; no production runtime fact |
| `external_schema.py` | external observation contract | Evidence layer; `external-runtime-observation-v1` stays stable |
| `external_writer.py` | external incident persistence | Evidence layer; no schema change |
| `schema.py` | schema validation helpers | Evidence layer |
| `bundle.py` | bundle construction/helpers | Evidence layer |
| `collect_bundle.py` | bounded evidence bundle creation | Evidence layer; links facts to provenance |
| `verify_bundle.py` | fail-closed bundle verification | Evidence layer; no scoring change |
| `replay.py` | offline replay | Evidence layer; results retain original version/score |
| `report.py` | bounded output report | Presentation layer; model IDs are future index metadata only |
| `timeline.py` | timeline view | Presentation layer, not a new progress producer |
| `timeline_bundle.py` | timeline bundle composition | Evidence/presentation layer |
| `timeline_facts.py` | typed timeline facts | Evidence/presentation layer; existing edits in the working tree are unrelated to this migration |

No `src/dfxlab` module is moved in the first phase. In particular, the
runtime model must not be smuggled into `progress.py` as a sixth verdict or
into `external_schema.py` as a new v0.2 observation.

## `experiments/`: every directory

| Directory | Model placement | Current classification for indexing |
| --- | --- | --- |
| `engine-liveness-contract` | M1/M2/M3/M5; K1/K2/K5 | active model-building evidence, not a second Q4 PoC |
| `fault-recovery` | M6 API/EngineCore fatal propagation | closed experiment; upstream thread remains separate |
| `native-evidence-capability` | M4 attribution/evidence layer | completed capability check |
| `oom-boundary` | Outside current model; no admitted scored run | archived in place; never executed |
| `organic-hang` | M3 hypothesis input; mechanism not established | retained case, no automatic fault category |
| `overhead` | Evidence-layer cost, not an M-fact | harness/reference |
| `preemption` | Outside current model; no admitted scored run | archived in place; never executed |
| `pytorch-c10d-shutdown-dump` | Cross-stack X1; possible M4 transfer only | separate PyTorch case, not a vLLM occurrence |
| `pytorch-unused-grad-dtype` | Cross-stack X2, below runtime-tree leaves | separate PyTorch case |
| `soak` | Outside current model; no admitted scored run | archived in place; never executed |
| `vllm-dp-supervisor-exit` | M1 DP supervisor, M6 exit propagation | Gate 1 pending; preserve Gate 0 score |
| `vllm-engine-binding-gate` | M1 EngineCore topology, M3 identity/freshness | active binding gate; no rank inference from missing log |
| `vllm-mm-uuid-encoder-cache` | Outside runtime-liveness model; cache correctness | retained draft, no automatic admission |
| `vllm-tp-dfx` | Opaque worker collective leaf below M1 | paused; no in-flight localization claim |
| `vllm-zmq-event-backpressure` | M3 useful progress, M4 S10 absence | active #53859 PoC and #36451 review evidence |

## Top-level Lab documents

This includes every file directly under `docs/` at the mapping date, plus
root `DESIGN.md` and `PRODUCT_ROADMAP.zh-CN.md`. `docs/reviews/` is indexed by
subject in a later phase; its individual review files keep their present
paths and evidence grades.

| Document | Placement |
| --- | --- |
| `DESIGN.md` (root) | Product/evidence boundary; model index must not rewrite its published-contract claims |
| `PRODUCT_ROADMAP.zh-CN.md` (root) | Governance; model is an organizing layer, not a new collector roadmap |
| `docs/ADOPTION_PLAN.md` | Governance/upstream exit |
| `docs/BLOCK5_NATIVE_CAPABILITY_RUN.md` | M4 native-evidence capability and provenance |
| `docs/COLLECT_VERIFY_V0_2.md` | Evidence layer; published v0.2 collect/verify contract |
| `docs/ENGINE_LIVENESS_CONTRACT_INVENTORY_2026-09-27.md` | Source map seeding M1–M6; observations remain pinned to their stated revision |
| `docs/ENGINE_LIVENESS_RFC_OUTLINE_2026-09-27.md` | Proposed contract, not the descriptive model |
| `docs/ENGINE_LIVENESS_RFC_OUTLINE_2026-09-27.zh-CN.md` | Chinese companion of the proposed contract |
| `docs/EVIDENCE_TO_CLAIM_BLOCK4.md` | Evidence layer: claim admissibility, not an M-fact |
| `docs/FAILURES_THAT_NEVER_REACH_THE_SUPERVISOR.md` | M6 propagation case study |
| `docs/FAILURES_THAT_NEVER_REACH_THE_SUPERVISOR.zh-CN.md` | Chinese M6 case-study companion |
| `docs/FAULT_TAXONOMY_V0_2026-09-24.md` | Taxonomy V1–V3/X1–X2 mapped to model, not replaced by M-IDs |
| `docs/LAB_ARCHITECTURE_PROPOSAL_2026-09-27.md` | Governance for the restructure; not an M-fact |
| `docs/LAB_ARCHITECTURE_PROPOSAL_2026-09-27.zh-CN.md` | Chinese companion of the architecture proposal |
| `docs/LOW_LEVEL_CAPABILITY_REQUIREMENTS.md` | Acquisition/probe gate; below-model leaves remain bounded |
| `docs/MATCHING_VERSIONS_ARE_NOT_RUNTIME_IDENTITY.md` | M3 identity/provenance case study |
| `docs/MATCHING_VERSIONS_ARE_NOT_RUNTIME_IDENTITY.zh-CN.md` | Chinese M3 identity companion |
| `docs/NATIVE_EVIDENCE_DESIGN.md` | M4 evidence-producer design |
| `docs/PAIN_POINT_DISCOVERY_2026Q4.zh-CN.md` | Bottom-up intake; add `model_cell` or `model_gap` only after protocol decision |
| `docs/PLAN_2026Q4.zh-CN.md` | Governance and PoC scheduling; model-building is not a second active PoC |
| `docs/STAGE_B_IDENTITY_POSTMORTEM.md` | M3 identity/provenance boundary |
| `docs/STAGE_C_JOIN_CONTRACT_PROPOSAL.md` | Cross-producer X1/M4 join proposal, not a shipped M3 binding |
| `docs/STAGE_C_RETAINED_INPUT_AUDIT_2026-09-23.md` | Evidence-layer NO-GO on historical join inputs |
| `docs/TIMELINE_FACT_CONTRACT.md` | Evidence/presentation contract; no runtime-state claim from visualization alone |
| `docs/V0.2_EXTERNAL_TRIAL.md` | Released evidence/trial record |
| `docs/V0.2_FIELD_ROLES.md` | Released decisive/non-decisive roles; immutable under this restructure |
| `docs/V0.2_LAUNCH_POST.md` | Release communication, not an M-fact |
| `docs/V0.2_LAUNCH_POST.zh-CN.md` | Chinese release communication |
| `docs/V0.2_RELEASE_PLAN.md` | Release governance and compatibility boundaries |

## Gaps and stop rules exposed by the mapping

1. The current recorder has no S10 no-progress trigger, while `progress.py`
   can score a bounded no-progress verdict. This is a model-to-producer gap,
   **not** authorization to add a trigger or alter v0.2.
2. Paused/sleeping state and loop responsiveness have no corresponding fields
   in the frozen progress verdict inputs. Record this as a model-to-verdict
   mapping gap; do not rescore historical runs.
3. The three stubs, cache-correctness draft and paused TP line have explicit
   homes *outside* the serving-liveness backbone. New work there needs its
   own admission, not an artificial M-ID.
4. If an organic issue cannot map to one of M1–M6 or an explicit outside
   category, enter `model_gap` in the private intake ledger. A monthly review
   may extend the model with a new version, keep the gap open, or rule it out.
