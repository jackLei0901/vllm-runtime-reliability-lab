# Stage B v2 scored native-producer result

Commit: `456d425`

- C0 healthy control: campaign `pass`; all three attributions `unmatched`; `negative_control_passed`.
- F1 held backpressure fault: campaign `pass`; py-spy A, PyStack B, and py-spy A2 all exactly match `vllm-53859-stage-b-v2-queue-wait-v1`; `blocked_in=queue_wait`; pairing `interchangeable`.
- A/A2 stability control passed in both cells. Frame-sequence equality was not compared.
- F1 health remained 2xx while request progress stalled, then recovered after release.
- Post-run base/fix build identities equal the pre-run restored identities. All process groups and EngineCore subjects were gone after cleanup.
- Raw stacks and server logs are intentionally excluded. Published capture records retain only typed outcomes, identities, bounds, producer provenance, and raw-output digests.

Preflight history is retained in `run-manifest.json`. Missing `torchvision`, a missing frozen `ninja` PATH entry, and an unauthorized login-shell watcher were rejected before the scored run; none is part of the scored claim.
