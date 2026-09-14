# Gate 1f GPU result — 2026-09-13

Verdict: **PASS for the frozen shutdown-stage diagnostic. Gate 1e remains
FAIL-CLOSED.**

The single predeclared affected trial ran from the exact committed tree at
`6db7069e916571a4e9a5f213bf63f881929a53a1`. Because GitHub access from the
GPU host timed out, that tree was transferred as a `git archive`; its SHA-256
was `ba0a78eacd20dec8d14cc5904cac20eda06aba2f0dfea7423ecfc3da6b9a9200`.
All six freeze verifiers passed from the extracted tree before execution, with
15, 7, 7, 9, 10 and 12 frozen files respectively.

The environment was two NVIDIA GeForce RTX 4090 GPUs, PyTorch
`2.13.0+cu130` at git revision
`cf30153c4c131c8164ee7798e5022d810682e2cb`, NCCL 2.29.7 and py-spy 0.4.2.

## Frozen observations

The trial retained no raw launcher or library log text. Eight fixed
`ProcessGroupNCCL` messages were reduced to these per-rank flags:

| flag | rank 0 | rank 1 |
| --- | --- | --- |
| shutdown started | false | true |
| operations flushed | false | true |
| watchdog joined; destroying communicators | false | true |
| destroy complete | false | false |
| dump signal observed from another rank | false | false |
| dump signal broadcast succeeded | true | false |
| dump signal broadcast failed | false | false |
| dump succeeded | true | false |

`library_log_scan_error` was null. Both the rank-0 and rank-1 frozen flag maps
matched exactly.

The previously established mechanism and termination observations also held:

- mechanism: `rank1_assertion_rank0_barrier_wait`;
- termination: `wall_bound_rank1_teardown_wait`;
- duration: 60.689 seconds;
- both external rank stacks captured at the frozen wait sites;
- exactly one decodable Flight Recorder dump, from rank 0;
- strict dump status `incomplete_dump_set`, with rank 1 missing;
- no tracked orphan processes;
- `raw_output_persisted: false`.

The independent verifier reported:

```text
PASS (Gate 1f: NCCL 2.29.7 shutdown-stage prediction confirmed; strict Gate 1e capture remains failed closed)
```

The retained JSON has SHA-256
`5ca23c73f4f5dd38cb709b917f85dbcc510641a10247e7c7472b02a362e611a7`.

## Interpretation boundary

The result confirms that rank 0 successfully sent the cross-rank dump request
and wrote its own dump. Rank 1 had logged the stage reached after its heartbeat
monitor was stopped and before communicator destruction completed, but it did
not observe or complete the later dump request.

This supports a PyTorch diagnosability issue: after a rank-local failure, a rank
following the process-group shutdown path can lose its Flight Recorder responder
before shutdown itself stalls. The result does not directly instrument NCCL and
therefore does not prove that the exact blocking call is `ncclCommDestroy`. It
does not establish behavior on other PyTorch or NCCL versions.

No repetition is permitted or needed under the Gate 1f protocol. A separately
frozen, FSDP-free two-rank reproducer would make an upstream report more
self-contained, but it is a new experiment and cannot alter this result.
