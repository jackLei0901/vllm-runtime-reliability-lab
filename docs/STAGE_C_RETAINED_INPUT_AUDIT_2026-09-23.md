# Stage C retained-input audit — 2026-09-23

Decision: **NO-GO for a retrospective Stage C join of the retained #196968
evidence**. Keep `join_contract_required`; do not implement a verifier against
invented fields, reinterpret the published Gate 1g result, or infer a C++ probe
need from missing retained data. This is an input-availability audit, not a new
GPU experiment or a retraction of the existing bounded `producer_missing`
case conclusion.

## Scope and method

Audited the lab tree at `056f0d644c6ba2d76108c2d12bec0a26c30dd5b0`;
the audit itself was first recorded at `62020c2baaf2d80def5f77b0e6addaafc3d7a648`.
For PyTorch source reading, the local proposed-fix branch was
`a33971146e1728d451f9cb5318fac63ffe5edc56`, compared with its fetched
`origin/main` at `51aa1bf319ea65e7070d9da5ceb8deb5dc54f600`.

Three evidence classes were kept separate: **S** = source reading of the
frozen Gate 1f/1g runners or pinned PyTorch checkout; **R** = inspection of
retained public JSON or private archive entries/markers; **I** = SHA-256
identity check against the published archive record. Source behavior is not
presented as a retained per-run observation.
Private log text, paths, stacks, and pickle contents are not republished here.
The archive SHA-256 values matched the already published validation record:
`9f7d058373b3e1ff78ecf60fa7e684fdb33a1b92916673e9933ff2bafbae4af6`
and `24566095846fda6662d475c8a55d1642c92ed0476099b57eb3b9aa67b9a1856e`.
The first archive contains 10 log entries; the second contains 3. Neither is
a structured Stage C sidecar. This audit does not certify that every possible
private or external copy has been searched.

## Contract input inventory

| Required join input | Evidence class | Retained evidence and source boundary | Audit result |
| --- | --- | --- | --- |
| Independent rank-to-PID/start-ticks binding | S + R | Gate 1g reproducer atomically wrote `rank-{rank}.pid`; `wait_for_rank_identities()` returned `ProcessIdentity` objects after reading `/proc` start ticks. The public JSON retains only `rank_identities_recorded: true`, not the binding values or a separately digested launcher record. The private fix-validation archives contain logs, not a structured binding. | **Missing for retrospective join.** The historical safety check was computed but cannot be independently rejoined by subject identity. |
| Cross-rank causal marker covering the held teardown state and peer dump request | S + R | Gate 1g retains rank-0 enqueue and rank-1 marker-observed/teardown-enter records. This orders user-space setup. Rank-0 dump broadcast and rank-1 shutdown are retained as separate log flags, without a pre-registered cross-rank acknowledgment or bounded join window relating *those* events. | **Partial.** Enqueue-before-teardown is supported; teardown-held-before-dump-request is not a separately verifiable Stage C relation. |
| Independent stack and lifecycle producer records | S + R | Gate 1g retains bounded external stack summaries and per-rank library-log flags; raw output and state are scoped to the runner's temporary directory. The flags are derived from one log scan, not a separately digested `lifecycle_stage_flags` producer with monotonic stage sequence. The positive fix-validation archives retain logs but no Stage C stack/flag sidecars. | **Partial for historical case, missing for the proposed join shape.** Do not make the stack digest vouch for flags. |
| Closed expected-file manifest for rank-1 dump absence | S + R | Gate 1g created a fresh temporary state directory and scanned the two expected `trace_{rank}` names (or `.pickle` alternatives). Public JSON retains the observed rank-0 file digest and a rank-1 missing-dump summary. It does not retain a canonical enumeration of every directory entry, a manifest digest, writer-set proof, or a pre-registered closure instant. | **Missing.** Historical absence remains bounded to the runner's tested paths, not the new closed-manifest claim. |
| Fix-arm responder deadline and same-clock observation bound | S + R | Proposed C++ source computes `options_->timeout + getDumpTimeout()` when enabling the shutdown responder. The source test demonstrates a successful rank-1 dump during destruction, but the retained logs do not expose the calculated monotonic deadline or a same-clock request/observation bound. An observed successful dump does not need an absence classification; a missing fix-arm dump after expiry cannot be scored. | **Missing for a fix-arm absence join.** A formula and successful positive run cannot substitute for a retained deadline event. |
| Comparable base/fix environment for a causal contrast | R + I | The existing validation explicitly used different PyTorch builds and NCCL versions; source identities and archive hashes are recorded. | **Not met for isolated patch-effect/performance claims.** Separate bounded base and fix observations remain valid. |

The temporary-directory statement is **code-derived**, not a host-observed
post-run search: `gate1g_campaign.py` creates `state_dir` inside
`TemporaryDirectory`, returns its summary from inside that scope, and the
context exits before the caller persists the returned JSON. The retained
public JSON confirms `raw_persisted: false`; this audit did not independently
inspect the original execution host after cleanup.

## Why the historical result still stands

Gate 1g's frozen verifier established its own narrower result: rank 1 was
sampled in teardown, rank 0 produced a decodable dump after broadcasting a
request, and the expected rank-1 dump was absent from the runner's checked
paths. The standalone fix validation separately showed complete rank-0 and
rank-1 traces in three positive trials. Those claims were made under their
own frozen protocols. The proposed Stage C join demands stronger independent
provenance and cross-producer binding; failure to reconstruct that *new*
contract does not erase the earlier experiments or upgrade them.
**No artifact already linked from #197232 requires amendment because of this
Stage C audit.** It applies a new, stronger join standard; it does not revise
the claims made under the earlier frozen protocols.

## Go/no-go and next action

**No-go:** no retrospective Stage C verifier, joined attribution, native
producer-interchangeability claim for #196968, or C++ probe decision from
these retained inputs. The only permitted Stage C status remains
`join_contract_required`.

**Conditional next step:** if a new run is authorized later, first review a
single bounded capture amendment for the *same* two-rank reproducer: retain a
separately digested launcher rank/PID/start-ticks map, causal marker protocol,
per-producer capture windows and digests, canonical closed file manifest, and
fix-arm responder deadline provenance. Freeze healthy/fault negative controls
and exact source/build identity before GPU execution. If any input cannot be
captured without broad C++ instrumentation or privacy leakage, return
`join_not_scorable`; do not loosen the join. Only after that input review
should a CPU fixture/verifier or GPU run be considered. This audit does not
authorize either.

The first two retention changes are small and reviewable: serialize the
`ProcessIdentity` objects already returned by `wait_for_rank_identities()`
alongside the launched rank assignment, then digest that binding; enumerate
and canonicalize the already-fresh `state_dir` at a pre-registered closure
instant, with its expected file set and known writer set, then digest the
manifest. They do not require new process discovery or a new dump collector.
The causal request marker, independently sourced lifecycle windows, and
fix-arm deadline still need their own feasibility review; this paragraph does
not assert they are already available.

## Supersession

This NO-GO applies only to the retained inputs inspected as of 2026-09-23.
A later capture under an amended, frozen protocol may receive a new Stage C
go/no-go decision with its own source and artifact identities. It supersedes
this audit **for that new run**, not retroactively for Gate 1g or the 2026-09-16
validation archives. A changed source tree or newly discovered archive alone
does not silently alter this decision; it requires a recorded re-audit.

## 中文审阅摘要

结论是 **NO-GO：现存 #196968 材料不能事后拼成 Stage C join**。Gate 1g 当时确实
读取了 rank PID 和 `/proc` start ticks，但公开结果只保留布尔值，临时目录已删除；
enqueue marker 只覆盖用户态准备，不足以独立证明 teardown 与后续 peer dump 请求
的跨 rank 因果顺序；缺失 dump 的文件扫描没有封闭目录 manifest；fix arm 也没有
保留与同一 monotonic 时钟绑定的 responder deadline。私有归档哈希匹配既有记录，
但其条目是日志而非结构化 join 输入。这不推翻 Gate 1g 或修复验证的原有有界
结论，只意味着不得用新契约追认旧证据。Stage C 保持 `join_contract_required`；
如果以后获准重跑，先评审同一复现器的最小采集补丁和负对照，再谈 verifier/GPU。
本审计不要求修改 #197232 已引用的公开材料；它只适用于截至 2026-09-23 核查的
保留输入。未来按冻结的新协议采集的运行可另作 go/no-go，但不会倒改旧运行结论。
