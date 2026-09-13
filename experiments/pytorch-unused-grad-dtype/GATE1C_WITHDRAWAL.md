# Gate 1c withdrawal record

Status: **withdrawn before execution**

Gate 1c was frozen for review and was never run. Its mechanism matrix remains
valid, but two campaign-level choices were too strict for the intended test:

- the runner stopped after a termination-prediction mismatch even though the
  protocol described mechanism and termination as independent gates;
- the 45-second wall bound left little margin for two plain-file Flight Recorder
  dumps expected around 36-40 seconds after launch.

Two additional hardening changes were accepted at the same boundary: kernels
without Yama must not fail an unconditional `PR_SET_PTRACER`, and result
verification must compare the current reproducer bytes with the hash retained
by the campaign before using source-derived stack line numbers.

`GATE1C_FREEZE.json` and all seven files it hashes remain unchanged as an audit
record. Gate 1d supersedes it.

