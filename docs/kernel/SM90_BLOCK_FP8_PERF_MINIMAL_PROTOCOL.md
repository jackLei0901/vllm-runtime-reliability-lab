# SM90 block-FP8 dispatch variant: minimal decisive protocol

Status: 2026-10-01, review draft r6 (one online freeze verification; offline receipts);
public freeze pending; not authorization to book a GPU. The
[Chinese version](SM90_BLOCK_FP8_PERF_MINIMAL_PROTOCOL.zh-CN.md) is
authoritative. This protocol **supersedes** the Session A/B plan in the
[charter](SM90_BLOCK_FP8_PERF_CHARTER.md) and the unpublished recovery plan. Those
documents, freeze `2c32e134…` and the 2026-10-01 records stay unchanged as
history.

## 1. The decision this session feeds

> Is the one-line dispatch change (`swap_ab` for all M ≤ 64) correct and
> measurably faster at the decode sizes forced-CUTLASS serving actually
> executes, so that it is worth proposing upstream?

Every item below exists because this decision needs it. Anything that doesn't
change the decision is dropped.

**Scope of the conclusion:**
- It is limited to SM90, the forced CUTLASS fallback and the four
  Qwen3-8B-FP8 shapes.
- A positive result supports a **kernel optimization proposal for the
  forced-CUTLASS fallback**. It is not a demonstrated end-to-end improvement
  in default serving, which selects FlashInfer/DeepGEMM on Hopper.

**Prior evidence, not re-derived here:** [#44572](https://github.com/vllm-project/vllm/pull/44572)
reports about 17–18 µs against about 50 µs, swapped against padded, at
(4096, 7168) for small odd M. This session measures the M % 4 = 0 points
#44572 did not.

## 2. What is kept, and what is dropped

**Kept from the frozen packet, unchanged:**
- the patch;
- the four shapes and the 20 M values;
- the 176-case correctness suite;
- rotating-weight CUDA-graph timing, A-A calibration and ABBA pairing;
- the constants W, R, n, b, q, k, e_rel and e_abs, and the clock rules;
- the per-cell classification (`positive` / `negative` /
  `below_positive_gate`);
- the M = 64 dispatch-identity probe.

**Dropped, because the decision does not depend on them:**
- the capture-op → replay-node mapping and its manual review;
- grid-based shape attribution;
- the V1 runner;
- the M = 63/64 gap gate as a precondition, since the variant is tested
  directly;
- the hot-versus-rotating contrast;
- the default-backend trace;
- split bookings;
- the end-to-end (Amdahl) estimate and the kernel time share.

## 3. Evidence and pass rules (frozen before booking)

### E1 correctness (required)

The 176 arm-cases, as already defined, must all pass.
- If base passes everything and only the variant mismatches, the outcome is
  **not worth proposing** (the variant is incorrect).
- Any other failure, or any unscored case, makes the session `unscored`.

### E2 dispatch identity (required)

At M = 64, `identity_probe.py` plus `trace_tools.validate_dispatch` must show
three launches per arm: cooperative for base and ping-pong for the variant.
Otherwise the session is `unscored`.

### E3 paired timing (decision evidence)

Run `timing.py` in its single-session mode:
- a fresh A-A calibration for all 80 cells;
- base/variant pairs for all 80 cells;
- the frozen bound and q-of-b rules.

**Precondition:** E3 starts only after compilation, serving and profiling have
all stopped. Before starting, check and record in the receipt that:
- no compile processes are running;
- no serving process group remains;
- no compute process is on the GPU, checked **before** the timer initializes CUDA;
- exactly one GPU reports memory.used ≤ 128 MiB. Empty PID inventory alone is
  not enough. Unknown PID/memory inventory fails closed.

Run the identical context-free idle check in the initial five-minute preflight.
This catches container PID/inventory problems early. An E3 timeout or a first-three-cell
projection exceeding the remaining fixed window gives `insufficient_evidence`,
retains partial records and stops without retry. Projection is only a conservative
cost estimate, not a measured negative result.

**Cell classification is the existing scorer's** (base is the left arm, the
variant the right):
- `positive`: the difference exceeds the bound under the q-of-b rule, so an
  **improvement is demonstrated**;
- `negative`: the median paired difference is below −bound, a
  **regression**;
- `below_positive_gate`: **improvement not demonstrated**. It does not mean
  equivalence was demonstrated;
- `unscored`: calibration failed or too few valid blocks.

**Cell groups:**
- **Changed served M:** {4, 8, 16, 24, 32, 40, 48, 56, 64}, so 9 × 4 = 36
  cells;
- **Controls:** the other 11 M values, where the variant leaves the code path
  unchanged (served M 1 and 2 included), so 44 cells.

**Decision rule (applied in order; CPU tests cover every branch):**

1. **Completeness.** If any of the 80 A-A records or 80 pair records is missing
   or `unscored`, the outcome is **insufficient evidence**.
2. **Controls.** If any control cell's |median paired difference| exceeds its
   bound, the outcome is **insufficient evidence**. A difference on an
   unchanged path points at the apparatus, not at the variant.
3. **Regression.** If any changed cell is `negative`, the outcome is **not
   worth proposing**.
4. **Improvement.** A changed served M counts as "improved" if and only if at
   least 3 of the 4 shapes are `positive` at that M.
   - At least 7 of the 9 such M improved: **worth proposing**.
   - Otherwise: **not worth proposing**.

**Notes on the rule:**
- It counts by M on purpose. The question is whether the variant helps across
  the decode sizes that actually run, and it tolerates one shape not
  benefiting.
- 7 of 9 and 3 of 4 are an engineering acceptance rule chosen in advance,
  **not** a statistical confidence claim.
- **Not worth proposing** means: on a complete measurement with passing
  controls, enough improvement was not demonstrated at the frozen bound, or a
  regression appeared. It does not mean the arms are equivalent.

### E4 serving execution path (supporting, qualitative only)

Take one Nsight Systems trace of forced-CUTLASS serving with the default
runner (V2):
- using `--cuda-graph-trace=node`;
- at concurrencies 1, 16, 63 and 64;
- with the fixed workload unchanged (128 input / 64 output tokens, fixed
  token IDs, ignore EOS).

E4 establishes execution paths and launch counts only. The replay-only witness
passes if, in at least 10 steady decode steps per level:
- **launch count:** each step has exactly 144 `cutlass_3x_gemm_fp8_blockwise`
  kernels;
- **kernel class:** all are ping-pong at concurrency 1, and all are
  cooperative at 16, 63 and 64.

**Defining steps and levels:**
- a step is one decode graph launch, identified through CUDA API correlation;
- levels are assigned from the batch time windows the collector records;
- any ambiguous assignment leaves that level unwitnessed.

**Padding corroboration:** record the resolved configuration after startup
(`cudagraph_capture_sizes`, graph mode), and require that:
- 63 is not a capture size;
- the smallest capture size ≥ 63 is 64.

A cooperative kernel at concurrency 63 establishes only the M % 4 = 0 dispatch
path. If a step actually has 63 decode tokens, the resolved configuration predicts
padding to 64. The trace confirms the cooperative path but does not directly
measure actual M. Client concurrency does not guarantee the engine batch size,
so concurrency 63 is not treated as proof of execution at exactly M = 64.

**What E4 does not do:**
- It does not infer shapes or tile counts from grids. Under the persistent
  scheduler the grid is not the logical tile count, and two shapes share
  N = 4096.
- It does not compute time shares or an end-to-end estimate. Builds run
  concurrently with E4, so kernel durations are not used quantitatively.

If E4 fails or doesn't fit, E3 still stands as a kernel-level result, with the
serving execution path recorded as unwitnessed.

## 4. Tool changes (CPU side, at most 3 working hours)

The original frozen packet does not implement this protocol. The independent
[`sm90-block-fp8-perf-minimal`](../../experiments/kernel-operand-contracts/sm90-block-fp8-perf-minimal/README.md)
successor implements the changes below; the original directory and manifest
remain untouched. CPU tests cover decisions and parsing, not GPU execution.
Check the test results and successor manifest before public freezing.

### 4.1 `timing.py` single-session entry point

- Add `--session S`, leaving the existing A/B behaviour unchanged.
  - The plan is B's: 80 A-A records plus 80 variant pairs.
  - It requires base, variant, dispatch (`witnessed`, both builds) and
    correctness (all 176 passed, same GPU).
  - It does not require `--session-a` or `--workload`.
- Record the E3 precondition check in the result.
- Implement the §3 decision rule as a pure function. It outputs:
  - `worth_proposing`, `not_worth_proposing` or `insufficient_evidence`;
  - the per-shape count for each changed M.
- State in the result that `below_positive_gate` means "improvement not
  demonstrated".

### 4.2 Aggregation tests (CPU)

Cover, with synthetic records:
- the 7/9 versus 6/9 boundary;
- the 3/4 versus 2/4 boundary;
- a missing cell and an `unscored` cell;
- a control failure, in both directions;
- a regression in a changed cell;
- one case that separates the "by M" and "by shape" formulations. For
  example, every M improves on 3 shapes but the missing shape rotates: by M
  it is worth proposing; by shape it fails.

The existing A/B tests keep passing.

### 4.3 E4 checker and collection settings

- **The checker:** small and replay-only, over the exported SQLite.
  - It splits steps by graph-launch correlation.
  - It assigns levels from the collector's batch windows.
  - It classifies kernels by name substring (`Cooperative` / `Pingpong`).
  - It counts launches per step.
  - It checks the capture sizes in the resolved configuration.
  - It fails closed on any ambiguous assignment. Check the first ten attributable
    decode graph launches at each level; do not select successful steps.
  - Its CPU tests are built from the existing V2 export's table schema. That
    data is parser-development input only and is not re-scored.
- **Serving launch settings:**
  - `VLLM_SERVER_DEV_MODE=1` on localhost only;
  - the serving environment's `bin` directory on PATH;
  - the forced-CUTLASS variables;
  - the collector records UTC and monotonic batch windows; export SQLite with
    UTC normalization and reject clock-span disagreement over 50 ms;
  - all of these recorded in the receipt.

### 4.4 Freeze

- Create a new successor freeze manifest that references this protocol and the
  revised tool digests.
- The `2c32e134…` freeze manifest is not edited, and the original tools stay
  reproducible at that commit; the original directory is not edited either.

Nothing else is built. If these changes take more than three hours, stop and
reconsider rather than extend.

## 5. One session, at most 90 minutes

| Step | Cap | Notes |
| --- | ---: | --- |
| Identity, PATH, ninja, disk | 5 | Reuse the existing environment and model; record the GPU UUID |
| Base and variant builds, with E4 concurrently only if the memory gate passes | 35 | E4 establishes execution paths and launch counts only; kernel durations are not used quantitatively |
| E2 dispatch probe | 5 | |
| E1 correctness | 10 | |
| E3 precondition check and paired timing | 20 | Starts only after builds, serving and profiling have all stopped; 160 records × 392 replays |
| Seal, download, verify, power off | 10 | Confirm billing has stopped in the console |

Two rules:
- **Don't start a step** unless the remaining time covers its cap.
- **An overrun** leaves the remaining items unscored; if it touches E3, the
  outcome is insufficient evidence.

Build base then variant **sequentially**, using `build_pair.py` and one 2,100-second
UTC deadline. Each compiler subprocess is capped at min(1,200, seconds remaining);
the second build does not renew the window. Failed base means no variant build.
Record physical and cgroup memory in the initial preflight; known available memory
must be at least 32 GiB for builds. E4 can overlap builds only if the effective
limit is at least 96 GiB and available memory at least 64 GiB, rechecked by the
collector before launch. Otherwise skip E4 as unwitnessed. This conservative
snapshot is not a guarantee of peak compiler memory; no parameter-changing retry.

Before E2, end E4 at build completion or the shared deadline, whichever comes first.
If still running, terminate only the verified collector/server process groups;
allow at most 10 seconds for SIGTERM cleanup, then SIGKILL any remaining owned
groups. Do not wait for collection or profiler flush to finish. Preserve partial
receipts and record interrupted E4 as unwitnessed. Run `apparatus.py --idle-only`
before E2 and again before E1; neither may start unless the context-free idle
check passes. Cleanup time is inside the existing cap, not additional time.
Defer E4's `replay_check.py`/SQLite export until after E3 if time remains, preferably
offline after power-off, using the sealed report and verified digests. Incomplete
or interrupted E4 stays unwitnessed. No E4 export/review overlaps E2/E1/E3.

**Budget accounting:** the 2026-10-01 session used 38 minutes of H800 time.
This session's 90 minutes brings the performance project to at most
128 minutes of GPU time. Report working hours spent so far against the
18-hour budget before booking. The exact historical duration was 38 min
15 s, making the total cap 128 min 15 s; "128 minutes" is rounded.

## 6. Stop rules

In the initial five-minute step only, verify the GitHub raw manifest byte-for-byte
with the reviewed academic proxy enabled and save a private `public_freeze.json`.
Record commit, manifest SHA-256, URL, verification time and success status.
Every later tool uses `SM90_PUBLIC_FREEZE_RECEIPT` to verify the receipt and all
local byte bindings offline. Missing/mismatched receipts fail closed without a
network fallback. Child processes inherit it; offline E4 review uses the same
downloaded receipt. Network failure stops cheaply before builds and is retained.
The receipt is a trusted local provenance record, not an independent signature.

- **Apparatus failure or insufficient evidence:** if this session reaches no
  decision for apparatus reasons (builds, unscored E1/E2, tooling, incomplete
  E3 or a failed control), record `unscored` or `insufficient_evidence` and
  **close the experiment**. There is no third attempt. This is not "no gap".
- **Not worth proposing:** close this experiment. The performance workstream
  continues with a new question.
- **Worth proposing:** the next step is an upstream conversation, framed as a
  follow-up to #44572, under the current PR and issue constraints. It is not a
  further local study. The claim is limited to a kernel-level improvement on
  the forced-CUTLASS fallback.

## 7. Before booking

1. Implement and CPU-test the §4 changes. Freeze the successor manifest,
   commit and push. Record the public commit.
2. Write two short paragraphs in your own words:
   - why M = 64 runs the cooperative 128×128 tile and M = 63 the swapped tile;
   - why a step with 63 decode tokens is expected to pad to 64, and why client
     concurrency and the trace do not establish its actual M.

   Review them before the session.
   Recorded on 2026-10-01: the user supplied a draft, the assistant corrected its
   generic alignment/padding explanations, and the user confirmed adoption of
   the corrected paragraphs in the runbook. This is assisted review, not an
   independently authored explanation or an independent understanding test.
3. Write down the expected outcome: from #44572, improvement at the changed
   points.

## 8. Output

The result is one page:
- the decision (worth proposing / not worth proposing / insufficient evidence
  / unscored);
- the per-shape cell classification table at the served M values;
- the shape count for each changed M;
- E1/E2/E4 status;
- limits, including the scope in §1;
- digests.

Raw data stays private under the existing rules.
