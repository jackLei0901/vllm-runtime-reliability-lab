# Freeze hash line-ending correction — 2026-09-13

The first clean Linux checkout performed immediately before Gate 1d execution
failed all four freeze verifiers. No GPU campaign trial had started.

The failure was caused by mixed line-ending representations in the original
Windows working tree. Three reused tracked files were hashed as CRLF bytes,
while the newly added experiment files were hashed as LF bytes. Git stores all
of them with LF endings, so a clean Linux checkout correctly rejected the CRLF
hashes.

This correction:

- adds a repository `.gitattributes` rule fixing text files to LF;
- replaces only the affected CRLF hashes with the SHA-256 values of the LF bytes
  stored by Git;
- does not change a protocol, reproducer, campaign, expected outcome, stop rule,
  result or interpretation.

The affected reused files are:

- `experiments/organic-hang/fetch_and_prepare_reproducer.py` in the Phase 2
  freeze;
- `experiments/organic-hang/normalize_flight_recorder.py` in the Gate 1b, 1c
  and 1d freezes;
- `experiments/organic-hang/process_lifecycle.py` in the Gate 1b, 1c and 1d
  freezes.

After this correction, every freeze verifier must pass from a clean Linux
checkout before Gate 1d may run.
