# Stage B environment-identity postmortem

Date: 2026-09-22

Status: retained-environment execution closed under the artifacts currently
available

## Decision

The retained #53859 Stage 1 environment must not be reconstructed from package
names and versions and then presented as the original environment. The first
Block 5 Stage B entry stopped at build identity, before a model or vLLM server
was started.

The retained-environment route is closed unless an external snapshot restores
the original dependency-pool target. A restored snapshot does not reopen the
route by itself: both regenerated arm records must first compare byte for byte
with the reviewed records in
`results/vllm-zmq-backpressure-stage1-build-20260915/`.

If no such snapshot is restored, the only valid continuation is a newly
reviewed Stage B v2 baseline. That baseline is a new experiment environment,
not an exact rerun of Stage 1 R3.

## What remained

The GPU host still contained:

- the reviewed base and fix source trees;
- the precompiled vLLM wheel used by Stage 1;
- the two arm environments;
- dependency-pool symlinks pointing at the original installation path;
- the frozen server and request inputs.

The original dependency installation target and local model directory were no
longer present. No recoverable copy of either was available among the retained
artifacts inspected during the run. This statement does not rule out a cloud
snapshot outside the instance; such a snapshot would have to be restored and
verified separately.

## Failed reconstruction

The missing installation target was recreated at the same absolute path from
the frozen distribution names and versions. The old dependency-pool symlinks
then resolved, and both arm identity generators completed, but their outputs
did not match the reviewed records.

| Check | Base | Fix |
| --- | ---: | ---: |
| Expected distributions | 187 | 187 |
| Observed distributions | 188 | 188 |
| Version mismatches | 0 | 0 |
| RECORD SHA-256 mismatches | 144 | 144 |
| Verified-file-count mismatches | 6 | 6 |
| Extra distributions | 1 | 1 |

The closed mismatch record is published at
[`../results/native-stack-pair-stage-b-preflight-20260922/mismatch-summary.json`](../results/native-stack-pair-stage-b-preflight-20260922/mismatch-summary.json).

## Root cause and boundary

The immediate cause was loss of the original dependency-pool installation
target. Reinstalling the same versions recreated a semantically similar package
set, but not the same installed artifact identity. RECORD contents can differ
because of wheel provenance, generated entry points, installation layout, and
installer behavior. A version lock therefore cannot substitute for an
installed-file identity gate.

The run does not establish which of those mechanisms caused every individual
hash difference. That distinction was unnecessary for the stop decision: the
frozen protocol requires byte-identical build records, and the records differed.

## Corrective actions

1. Preserve the complete installed dependency pool or an immutable archive of
   it, not only a package/version list.
2. Preserve every input wheel and its digest.
3. Pin the model to an immutable revision and retain its file manifest.
4. Freeze the canonical extraction path when RECORD or generated scripts can
   encode installation paths.
5. Run the build-identity gate before downloading a model or starting a server.
6. Keep the preflight result as evidence; do not overwrite it with a later run.

The reviewed continuation contract is
[`../experiments/native-evidence-capability/STAGE_B_V2_PREREGISTRATION.md`](../experiments/native-evidence-capability/STAGE_B_V2_PREREGISTRATION.md).

## 中文结论

本次失败不是 GPU、vLLM 或 native producer 失败，而是实验环境身份无法恢复。虽然
distribution 版本没有差异，但两个 arm 均出现 144 个 RECORD SHA-256 差异，因此不能
把重装环境称为原 Stage 1 环境。当前 retained-environment 路线停止；除非外部快照能
恢复原 dependency pool，且重新生成的两份 identity record 与冻结记录逐字节一致。
否则只能建立一个明确标记为新基线的 Stage B v2 环境。
