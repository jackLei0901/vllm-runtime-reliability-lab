# Fault taxonomy v0 — admitted cases only

Status: **bounded analysis draft**, 2026-09-24. This is a fault/diagnosability
map, not an expansion of `dfxlab.progress`'s five verdicts, a classifier, or a
claim to cover all vLLM failures. It uses only cases already investigated by
the lab. A fault category names the *failed boundary or mechanism*; a verdict
names *what the available observations permit us to conclude*. One verdict
can occur under several mechanisms, and the same mechanism may be
`undetermined` when required evidence is missing.

## vLLM serving and lifecycle categories

| ID and failed boundary | Misleading visible signal | Discriminating evidence and admitted result | Counterexample / stop condition | Coverage |
| --- | --- | --- | --- | --- |
| **V1 — useful work stops behind a responsive endpoint**. In #53859 the specific mechanism was publisher event-queue backpressure in EngineCore. | `/health` stayed 2xx and the process stayed alive. | Admitted incomplete request, flat content-bearing progress during the bounded pause, stable EngineCore identity, and a queue-wait stack. Four cells distinguish trigger from baseline and show proposed #53883 progress with four dropped batches. | No admitted demand, stale progress producer, or a progressing decision source prevents the no-progress claim. A stack without progress evidence cannot create it. | Demonstrated for one single-GPU EngineCore and deterministic queue trigger; not DP-wide or a generic hang classifier. Lab independently validated, did not discover #53859. |
| **V2 — fatal EngineCore cause is lost at the API-process exit boundary**. | Top-level exit status 0 could be read as clean shutdown after unexpected EngineCore death. | #52178 process-level validation distinguishes unexpected EngineCore/worker failure from intentional SIGTERM; proposed-fix trials restored nonzero status while intentional termination remained 0. | Intentional SIGTERM is a required negative control; a generic nonzero exit for every shutdown would be wrong. | Lab provided system-level validation of an independently discovered vLLM issue. It is not a lab-originated finding. |
| **V3 — managed-child or probe failure is lost at the DP-supervisor exit boundary**. | The supervisor returned 0 after a child status 17 or post-ready probe failure. | Lab Gate 0's four cells retained child status, real server readiness/probe path, subject status, and intentional SIGTERM control. | If a full-package Gate 1 does not reproduce the process-level exit contract, keep the claim at the thin Gate 0 boundary. | Lab-confirmed Gate 0 on the pinned implementation; no upstream issue/fix claimed, full-package Gate 1 not run. Distinct from V2's single API-server path. |

The categories are not a severity ranking. V1 is a progress failure; V2 and
V3 are failure-*propagation* failures. None by itself says that a GPU, NCCL
collective, or kernel stopped making progress.

**Actionable V1 gap — recorder and verdict paths do not yet meet.**
The recorder already has bounded pre-fault history, but `recorder.py:87-110`
triggers on exit, health loss, KV pressure, and preemption, not demand-gated
no-progress. The separately invoked `collect/verify` path can reach that
verdict. A V1 incident with a live process and green health need not fire a
recorder trigger, so the flagship fault does not automatically freeze the
pre-fault ring.
This is not a one-line trigger addition: `external_schema.py:86-99` does not
carry generation-token progress in `MetricsObservation`, and a sound trigger
must also enforce admitted demand, fresh progress, identity, evaluation
window, precedence and cooldown rather than equating a flat counter with a
hang. v0.2's closed artifact contract remains frozen. The next step is a
CPU-only design/negative-control review for an optional future integration,
not an unreviewed v0.2 schema edit or new collector.

## Cross-stack transfer cases, not admitted vLLM categories

| ID | Demonstrated elsewhere | Transfer limit |
| --- | --- | --- |
| **X1 — diagnostic producer disappears while participant survives** | PyTorch #196968: rank 1 remained alive in teardown while its Flight Recorder dump was absent; #197232 is a proposed C++ fix. `producer_missing != member_missing` changes the investigation. | No vLLM-specific rank/responder occurrence has been demonstrated; Stage C's stronger cross-producer join is NO-GO on retained historical inputs. Do not present X1 as a general vLLM join capability. |
| **X2 — local correctness failure presents as a distributed hang** | PyTorch #196996: a mixed-gradient-dtype assertion was reduced to one GPU with a uniform-dtype control. | This falsifies the assumption that every distributed symptom is a collective fault. It does not establish a vLLM FSDP2 fault or authorize NCCL instrumentation. |

## Admission and change rule

For each new category, require a named discriminating fact, a reproducible
positive case, a negative control, exact source/build/topology bounds, and a
counterexample or `unknown` exit. Existing metrics/logs/Flight Recorder/NCCL
RAS must be compared before naming an irreducible observability gap. A
missing producer is not a negative target-state observation. No category is
added merely because an issue title or stack frame is new.

V1's same-case DFX ablation is
[`reviews/VLLM_53859_DFX_BASELINE_ABLATION_2026-09-24.md`](reviews/VLLM_53859_DFX_BASELINE_ABLATION_2026-09-24.md).
The cross-case evidence rules are in
[`EVIDENCE_TO_CLAIM_BLOCK4.md`](EVIDENCE_TO_CLAIM_BLOCK4.md).
X1's retained-input limit is in
[`STAGE_C_RETAINED_INPUT_AUDIT_2026-09-23.md`](STAGE_C_RETAINED_INPUT_AUDIT_2026-09-23.md).

## What this taxonomy does not answer yet

- It is not an exhaustive corpus mined from the vLLM issue tracker and has no
  measured coverage denominator.
- It cannot identify which of N EngineCore/worker processes failed from one
  v0.2 `collect` bundle; general cross-process identity/window joining is
  absent.
- It does not prove that existing metrics cannot detect V1. The published R3
  bundle has no retained metrics time series to score that baseline.
- It does not classify CUDA OOM, allocator state, graph capture, encoder cache,
  speculative decoding, or disaggregated KV transfer. These are not empty
  categories to fill by collecting unrelated issues.

## Coverage denominator protocol — proposed, not executed

To measure whether V1–V3 cover a meaningful fraction of *reported* runtime
reliability issues without admitting new categories, preregister a fixed
GitHub snapshot, an exact closed-issue search/inclusion rule, explicit
exclusions, a random seed, and a sample size of 40 before reading issue
contents. Exclude install/build problems, accuracy-only bugs, feature
requests, and performance-only reports without a reliability failure from
the sampling frame. Label each included sampled issue `V1`, `V2`, `V3`,
`none_of_v0`, or `insufficient_information`; retain sampled IDs, exclusion
reasons, and counts at each exclusion stage in a public manifest. Group
duplicates in a sensitivity analysis so one widely reported incident does
not dominate the count. A second pass on an independently chosen subset
should check label consistency.

This is *measurement*, not category mining: no new issue is filed and no new
taxon is admitted from the sample. Issue text alone usually cannot establish
which deployed DFX tools were enabled or what their retained output proved.
The sample may estimate taxonomy coverage of the chosen reporting frame;
it **cannot** by itself measure the prevalence of production faults or prove
which DFX method “misses most.” That stronger conclusion requires existing
telemetry artifacts or a controlled same-case baseline. This protocol has
not been run and contributes no current denominator. At `n=40`, even a
simple random sample near 50% has a nominal 95% margin of roughly ±15
percentage points, before reporting-frame, labeling, or duplicate bias.
The sample can separate coarse "many" from "few," not rank close categories.

The next useful progress is not category count: review the V1 recorder/verify
integration gate, preregister the coverage sample if a denominator is needed,
or execute V3's already registered Gate 1 protocol when the environment is
ready. An external counterexample may also narrow a row. Any of these may
leave the taxonomy unchanged.
