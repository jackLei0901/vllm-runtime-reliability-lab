# Gate 1d execution stop — 2026-09-13

Gate 1d started only after all four freeze verifiers passed in a clean Linux
checkout at commit `9bd622224d31a896ad134d21f7d904500798f424`.

Control trial 1 completed with the frozen mechanism and termination outcomes:
`completed_symmetric_fp32` and `normal_completion`. During control trial 2, the
runner raised `json.decoder.JSONDecodeError: Extra data` before it could write a
result. The campaign stopped automatically. No affected trial ran.

The failure occurred while parsing adjacent structured markers written by two
ranks to the shared launcher output. The Gate 1d parser used a line-oriented
regular expression whose capture could contain more than one JSON object when
two complete marker writes appeared on the same line. This is a runner evidence
retention defect, not a mechanism or termination observation.

The temporary raw launcher output was deleted by the existing privacy boundary.
The one completed allow-listed control summary is retained. No claim is made
from the incomplete Gate 1d campaign. The retained summary is
`results/pytorch-unused-grad-dtype-gate1d-stop-20260913/control-trial-1.json`.

Gate 1d remains frozen as the executed record. Its replacement must use a new
freeze, parse each marker object independently, test the adjacent-marker case,
and stop automatically on any control-arm termination mismatch.
