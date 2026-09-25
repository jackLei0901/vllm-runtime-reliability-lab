# Review entry — Lab value gate and parallel C++ track, 2026-09-25

## Outcome

The previously reviewed DFX source audits, bounded taxonomy, #53859
ablation, and roadmap were committed as `1b28aff`. The mm-uuid experiment
and older c10d/DP drafts were not staged. This new review set is
documentation-only and was committed after source review. It adds no
collector, fault category, upstream issue, PR, or v0.2 verdict change.

Read these three new documents in order:

1. [V1 recorder–verdict integration gate](V1_RECORDER_VERDICT_INTEGRATION_GATE_2026-09-25.md)
   — precise future trigger preconditions, per-engine/request scope boundary,
   ring and capture semantics, CPU negative-control matrix, and an explicit
   “no integration needed yet” exit. The pinned #53859 source places the
   blocking publish call inside one engine's scheduler step; summed metrics
   across DP engines would hide a blocked engine.
2. [DFX claim-delta ledger](DFX_CLAIM_DELTA_LEDGER_2026-09-25.md)
   — five existing cases with baseline evidence, Lab's added inference,
   and the facts still unproved. R3 metrics-only detection remains **not
   scored**, not “failed.”
3. [#55537 CUTLASS contract read](VLLM_55537_CUTLASS_CONTRACT_READ_2026-09-25.md)
   — pinned read-only analysis of a dispatch-layout boundary on the
   independent vLLM CUDA/C++ training track. The guard reaches blockwise
   FP8 A/B/output, but this review ran no blockwise padded-view test. It
   does not count as a Lab DFX finding or as kernel implementation.

## Decisions requested from reviewers

- For the admitted V1 mechanism, is the future **engine-scoped** claim
  correctly bounded? One engine can block in scheduler publication while
  another advances; a summed service counter must not decide both.
- Can the per-engine token/demand partition be bound to the affected
  EngineCore PID and start time without adding an invasive default probe?
  The optional `EngineCore_DP{dp_rank}` process title is a non-attaching
  candidate, not proof: missing `setproctitle` removes it, and label/rank
  correspondence still needs a launch-mode-specific check.
  If not, keep the explicit `collect/verify` path rather than dilute the
  trigger preconditions.
- Before code, should a future recorder reuse the pure v0.2 progress rules
  inside a versioned observation, or consume a separate collect witness?
  The latter requires a real target/clock/identity join, not colocated files.
- Does the claim ledger ever treat missing retained metrics as a metrics
  failure, or upgrade PyTorch transfer evidence into a vLLM occurrence?
  #196968 supplies a bounded same-run method comparison, not a completed
  Stage C closed join.
- Does the #55537 note keep dispatch-contract work separate from measured
  kernel correctness and performance work, and does its test matrix cover
  the blockwise FP8 behavior the guard now reaches?

## Verification and limits

The new documents were checked against the local `1b28aff` Lab source, the
`7b054aca` #55537 checkout, and the #53859 vLLM base source `22258a26`.
Markdown links and whitespace were checked.
No GPU, vLLM runtime, full Python suite, CI, or live upstream status check
was performed for this documentation-only round. The proposed CPU matrix
has **not** been implemented or executed; its expected outcomes are a design
contract for review, not test results.

## 中文审阅入口

本轮先把已审阅的六个 DFX 文件精确提交为 `1b28aff`，没有混入 mm-uuid 或较早
草稿。新材料经源码审阅后单独提交：V1 设计只讨论未来如何让有界 recorder 历史在严格
“有需求且无进展”的条件下保存；评价单位是单个 engine，不能把多个 DP engine
的 counter 相加后宣称都在前进。现有案例对照明确哪些结论已证明、哪些未测；
#55537 是独立的 CUDA/C++ dispatch 合同阅读，不记作 Lab 的 DFX 发现或
kernel 实现；其 guard 也覆盖 blockwise FP8，但尚无相应的 padded-view 测试。
请优先挑战 per-engine 证据与 PID 的绑定是否足够，以及现有 DFX 的比较有没有
过度宣称。
