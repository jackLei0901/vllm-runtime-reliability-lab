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

The byte-level corrections were:

| file | old CRLF SHA-256 | new LF SHA-256 |
| --- | --- | --- |
| `fetch_and_prepare_reproducer.py` | `d05d18e0801061f3693ccfce32691cfcb79dc784e5abb0f162688eaaeb08da5f` | `f17d7932056f388ad711cbcf5101f37f6c59793571dbbe0716b0149dd03b06f9` |
| `normalize_flight_recorder.py` | `43569fb021c8c3f1fb3b21abd9227bb4ad90dbaa4ef3c403db1a308bd18c06f6` | `2a56065f105792889bfdf261295605e53e81f87e9f0892e18a503a35231a32d0` |
| `process_lifecycle.py` | `80d4e9f3e5f90a15c231745a1e9ea3bebc11002ab0ed74900c58c928d57bab14` | `01dfdfde715587b0c65ee37fd042abd3355ab3edc333e4f5b306bb9678667e2e` |

Each new digest is the SHA-256 of the corresponding old file after converting
CRLF line endings to LF and making no content change.

After this correction, every freeze verifier had to pass from a clean Linux
checkout before Gate 1d could run.
