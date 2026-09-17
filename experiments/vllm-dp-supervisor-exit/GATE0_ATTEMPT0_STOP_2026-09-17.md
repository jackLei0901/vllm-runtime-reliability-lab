# Gate 0 attempt 0 stop — 2026-09-17

Attempt 0 did not produce a mechanism observation.

Both fresh subject processes reached the 15-second outer bound before emitting
their closed-shape record. They had return code `null`, empty stderr hashes, and
no parsed subject result. A separate unscored diagnostic run of the abnormal
cell completed correctly in about 32 seconds; most of that time was vLLM import
on the 0.5-core host.

This is classified as an execution-bound defect in the runner, not as evidence
for or against the DPSupervisor hypothesis. The scored revision raises only the
per-cell outer bound to 90 seconds and records elapsed time. Its predictions,
process events, and acceptance criteria are unchanged.
