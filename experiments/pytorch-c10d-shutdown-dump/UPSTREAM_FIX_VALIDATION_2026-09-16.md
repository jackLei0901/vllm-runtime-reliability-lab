# PyTorch c10d shutdown dump: upstream fix validation

Date: 2026-09-16

Upstream issue: [pytorch/pytorch#196968](https://github.com/pytorch/pytorch/issues/196968)

Draft fix: [pytorch/pytorch#197232](https://github.com/pytorch/pytorch/pull/197232)

## Why this case matters

The investigation began while correlating evidence across ranks for a
distributed hang. Rank 0 produced a Flight Recorder dump, while rank 1 was
still alive in `destroy_process_group()` but produced no trace. The missing
artifact could not safely be interpreted as non-participation by that rank.

External stack sampling and privacy-bounded shutdown-stage flags showed that
rank 1 had stopped its heartbeat monitor and then remained in communicator
destruction. The same monitor was the default process group's responder for a
peer `exception_dump` request. This reduced the broader hang to a standalone
two-rank c10d reproducer without FSDP, vLLM, model code, or gradient
accumulation.

## Tested change

The proposed fix keeps a narrowly scoped peer dump-signal responder alive while
legacy `ProcessGroupNCCL` destroys communicators. It does not keep watchdog
classification active, symbolize Python stacks, call communicator dump APIs,
or terminate the process.

The product binary was built from:

- source commit: `696eb9ad00789be279bf84bdacd3078d42e76ca2`;
- wheel: `torch-2.15.0a0+git696eb9a-cp312-cp312-linux_x86_64.whl`;
- wheel SHA-256:
  `0dfff8bf28996208dd156f9ece4f49a9118d7e93f696e6a1825f662bca72cc92`.

The positive test executions used the test file as of `ac058178`, whose only
change after `696eb9ad` was removal of a bare class decorator that caused
pytest to collect zero instances of the new test. Each scored run explicitly
reported one collected test and `1 passed`.

## Results

| Arm | Runtime | NCCL | Result |
| --- | --- | --- | --- |
| Unpatched | PyTorch 2.13.0+cu130 wheel | 2.29.7 | Expected failure in 3/3 runs: rank 1 produced no complete trace |
| Patched | Source-built wheel at `696eb9ad` | 2.30.7 | Passed in 3/3 runs: ranks 0 and 1 produced complete, separately named, decodable traces |

Positive-run durations were 12.45 s, 11.96 s, and 11.98 s. In every run,
rank 0 broadcast `exception_dump`; rank 1 observed the request while
communicator destruction was in progress, wrote its trace without stack
symbolization, and subsequently completed destruction.

Additional checks on the patched build:

- six collected destruction and Flight Recorder regression tests passed;
- a temporarily enabled `test_timeout_dumps_on_stuck_ranks` passed;
- `git diff --check`, Ruff, Python compilation, and clang-format passed.

The existing dormant `test_timeout_dumps` was then rewritten to use the same
bounded complete-pickle loader as the new regression test. With its class
temporarily enabled, both `timing_enabled=False` and `timing_enabled=True`
passed in 3/3 runs on the patched build. Every invocation validated rank 1's
single completed entry, and the inherited strict rank-1 exit-code requirement
of zero also passed. The tested source revision was `6342275774378594`; the
only remote-only change was removal of the class decorator required to collect
the dormant test.

## Interpretation boundary

The negative and positive trials were not run in a byte-identical environment:
they differ in PyTorch build, CUDA toolkit, and NCCL version. The negative arm
is evidence that the released implementation exhibits the gap; the positive
arm is evidence that the proposed implementation closes it on current source.
It is not presented as a same-environment performance comparison.

The validation covers the selectable legacy `ProcessGroupNCCL` backend with
`TORCH_DIST_USE_NCCL2=0`. It does not change or validate the default `nccl2`
backend.

The test demonstrates complete Flight Recorder artifacts from both ranks. It
does not identify the exact NCCL function in which communicator destruction
waits, and rank 1's shutdown-time dump intentionally contains no Python stack
frames.

## Evidence integrity

Raw logs are retained privately because they include process and environment
details outside the public schema. Their archive is content-addressed:

- archive: `pytorch-196968-validation-20260916.tgz`;
- SHA-256:
  `9f7d058373b3e1ff78ecf60fa7e684fdb33a1b92916673e9933ff2bafbae4af6`.

The three repeated runs of the rewritten existing test are retained separately:

- archive: `pytorch-196968-timeout-rewrite-validation-20260916.tgz`;
- SHA-256:
  `24566095846fda6662d475c8a55d1642c92ed0476099b57eb3b9aa67b9a1856e`.

The public claims above are limited to the reviewed, closed-shape results and
the upstream code and test changes.

## AI assistance disclosure

AI tools assisted with source review, protocol review, and drafting. The
repository owner ran the validation, retained the evidence, and reviewed the
claims in this record.
