# PyTorch #196968 fix: negative-control validation

Date: 2026-09-16

## Scope

This record covers only the negative control for the proposed upstream
regression test. It does not claim that the C++ fix has passed a patched build.

The test was authored against PyTorch `main` at
`51aa1bf319ea65e7070d9da5ceb8deb5dc54f600`. For the negative control, the test
body was applied to the v2.13.0 test file with a small compatibility adapter
that removes `skip_but_pass_in_sandcastle` from this class only. The installed
v2.13 test utility exposes that helper as a decorator factory; without the
adapter, the class is replaced by a function and no test executes. The adapter
does not change the process-group workload, assertions, or expected traces.

## Environment

- GPUs: 2 x NVIDIA GeForce RTX 4090
- PyTorch: `2.13.0+cu130`
- PyTorch git version: `cf30153c4c131c8164ee7798e5022d810682e2cb`
- CUDA: `13.0`
- loaded NCCL: `2.29.7`
- test file SHA-256 after applying the test patch and adapter:
  `785ef005f2e4dbf05c9c90e01db8778f7d86d27ecd01e0985f7cd5b948cbffd2`

## Command

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_PRINT_REPRO_ON_FAILURE=0 \
timeout -k 10s 180s \
python test/distributed/test_c10d_nccl.py \
  -k test_peer_dump_request_during_shutdown
```

## Result

The unpatched implementation failed in the intended way in three consecutive
trials:

| Trial | Duration | Exit | Decisive assertion | Private log SHA-256 |
| --- | ---: | ---: | --- | --- |
| 1 | 40.285 s | 1 | `rank 1 did not write a complete trace` | `6b8a7c40634578f2f03f7856394637a585b5f9df3eca982ca139a9ed4b3e8664` |
| 2 | 40.007 s | 1 | `rank 1 did not write a complete trace` | `d663c1869585f5391e9038608b249c3017cb2158ea88778fdf1ecf32e6fd67d8` |
| 3 | 40.039 s | 1 | `rank 1 did not write a complete trace` | `33a084b81a403dfcc141d1b6d5d47a40260f28b8dc3636e0e4cc4c3fa20c4e92` |

The parent process successfully loaded rank 0's complete trace before checking
rank 1. Therefore this is not a generic failure to trigger or decode a Flight
Recorder dump: the missing artifact is specifically rank 1's trace while that
rank is in process-group shutdown.

## Patch status

Local upstream branch: `fix/196968-shutdown-dump-responder`

- regression-test commit: `2298154`
- implementation commit after clang-format correction: `696eb9a`
- `git diff --check`: pass
- `clang-format 18.1.8 --dry-run --Werror` on the two changed C++ files: pass
- Python test compilation: pass

The rented machine has only a source archive without Git submodule contents and
about 19 GiB free on its data volume. It is not a valid environment for a full
PyTorch build. Positive patched validation, the two existing NCCL regression
tests, and the full linter/build checks remain required before the PR is marked
ready for review. They should run in a complete recursive checkout or upstream
CI; this negative-control result must not be presented as patched-build proof.
