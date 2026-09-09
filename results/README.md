# Results policy

Each experiment uses its own directory and records the environment, raw timeline, workload output, service logs, injection events, and summary. Large reviewed bundles may be attached to a GitHub Release, but the README must record the SHA-256 digest, code commit, and generation command.

Do not present any of the following as a validated conclusion:

- example data that was not executed
- performance comparisons with mismatched environments or server arguments
- failure trials that preserve a log excerpt but not the complete exit code or request result
- multi-GPU conclusions extrapolated from a single-GPU run
