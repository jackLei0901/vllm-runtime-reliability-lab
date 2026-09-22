# Native stack producer Stage A CPU result

Date: 2026-09-22

Harness source commit: `496bde643dc0cb8fd3d26c2215e52a402af62bcf`

Public capture record:
[`capture-record.json`](capture-record.json), SHA-256
`89e0f9b4f1f19cf9fc0cf662ccf604b4f5485ce7f8e6b41ba2a0b7e1aaf4a0a1`.

## Question

Can the bounded Block 5 harness execute the real `py-spy -> PyStack -> py-spy`
sequence against one stable Linux process without a GPU?

This is an acquisition-capability check. It is not the #53859 Stage B
healthy/fault comparison and it cannot admit a target-runtime attribution
rule.

## Environment

- Ubuntu 22.04, Linux 5.15, x86-64;
- CPython 3.12.3;
- no GPU device;
- `kernel.yama.ptrace_scope = 1`;
- container without `CAP_SYS_PTRACE`, with seccomp mode 2;
- `libdw.so.1` and `libelf.so.1` available;
- `py-spy 0.4.2` and PyStack `1.7.1` in an isolated virtual environment;
- deterministic Python target blocked in `threading.Condition.wait()`;
- target-scoped `PR_SET_PTRACER_ANY` authorization; no system-wide ptrace
  relaxation.

## Result

| Capture | Outcome | Identity stable | Raw output |
| --- | --- | --- | --- |
| `py-spy-a` | `execution / produced` | yes | private; digest retained |
| `pystack-b` | `execution / produced` | yes | private; digest retained |
| `py-spy-a2` | `execution / produced` | yes | private; digest retained |

The two `py-spy` raw-output digests are identical. All public and private files
were written mode `0600`; raw stack text was not copied into the repository.
The public capture record validates as three closed
`native-evidence-experiment-v0` envelopes.

The first attempt also exposed a harness defect: `py-spy 0.4.2 dump` rejects
the previously supplied `--idle` argument. The command was corrected to the
supported `dump --pid <pid> --native` form and pinned by a regression test
before the successful run.

## Claim boundary

The record deliberately states:

```json
{
  "pairing_status": "normalization_pending",
  "claim": null
}
```

Identical repeated-producer digests establish acquisition-level stability for
this controlled target. They do not establish that PyStack and `py-spy` are
interchangeable, and they say nothing about the #53859 queue-wait fault. Those
claims require an admitted version-scoped rule and the retained Stage B
healthy/fault pair.

## Next gate

Run the same bounded sequence against the retained #53859 C0/F1 targets in a
GPU environment. Until that run is reviewed, do not add a target-runtime rule,
normalize these generic frames into `blocked_in`, or proceed to Stage C.
