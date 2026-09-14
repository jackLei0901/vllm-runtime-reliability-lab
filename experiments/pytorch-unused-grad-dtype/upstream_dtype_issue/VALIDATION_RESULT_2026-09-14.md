# PyTorch FSDP2 dtype issue validation result — 2026-09-14

## Verdict

**Issue evidence PASS.** Both single-rank triggers reproduce the same FSDP2
mixed-gradient-dtype assertion on the tested current nightly. The final
flag-off recheck proves that `last-microbatch` does not depend on
`set_reduce_scatter_unused_params()`. The two-rank run is retained only as an
impact observation, not as a separate teardown or process-group defect.

## Environment

- PyTorch: `2.15.0.dev20260914+cu130`
- PyTorch git revision: `7d5f0216450b8253e62d88284d49de725d88bd42`
- Python: 3.12.3
- CUDA used to build PyTorch: 13.0
- loaded NCCL: 2.30.7
- GPUs: 2 x NVIDIA GeForce RTX 4090
- driver: 580.105.08
- two-rank backend: nightly default `nccl2`

The Conda packages printed by `collect_env` belong to an unrelated base
environment. All validation commands used the pip environment at
`/root/venvs/fsdp-dtype-nightly`.

## Observed matrix

| ranks | case | unused-parameter API | result |
| ---: | --- | --- | --- |
| 1 | `control` | off | `COMMAND_RC=0`; completed and cleaned up |
| 1 | `last-microbatch` | off | `COMMAND_RC=1`; exact `{bfloat16, float32}` assertion |
| 1 | `unused-placeholder` | on | `COMMAND_RC=1`; same assertion |
| 2 | `rank-divergent-unused` | on | rank 1 asserted; rank 0 did not complete; external bound returned 124 |

The initial `unused-placeholder` and two-rank runs used the first script
revision, where the unused-parameter API was enabled unconditionally. The final
revision enables it only for those two cases; therefore their executed code
paths are unchanged. The final flag-off revision was run for `control` and
`last-microbatch`, the two cases whose API independence needed confirmation.

## Artifact hashes

```text
reproducer.py
FBD48185F9AA77566037B23C7C4E625B33A87A676819F75D5CF86704258CCF09

single-control-flag-off-final.log
906B81C65CC9FF52A3A6AF87B24CE054A014D46CDE6C49FA129225D594055881

single-last-microbatch-flag-off-final.log
B6FEC59F99DC0C3031E0DECF7D6863E1C200F98A7FD7721BABCD63A5EFFB01E0

single-unused-placeholder.log
92016D9145A6CDE753BC27F3C454BE648ED0A9DB79C1BDCFA229E4ACEB6CFF12

two-rank-divergent-unused-final.log
0192153E2C9B80F64BB98F35970CAF2311E1FF544121EC562C053D7462BD7F61

dtype-version.txt
4A34175F0ED6CD26D598E72C106202E7B16FA44F72DF6819E27DDABC8C3264AA

dtype-collect-env.txt
0D9C446F7FBA6F8992133377C55EF7CD9DE995C86BD3423EECAE658D83D5550B
```

Raw logs remain outside the public repository pending privacy review. The
public issue should quote only the minimal assertion, return codes and required
environment output.
