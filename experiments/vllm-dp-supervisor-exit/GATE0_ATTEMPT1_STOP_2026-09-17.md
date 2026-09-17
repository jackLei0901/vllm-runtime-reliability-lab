# Gate 0 attempt 1 stop — 2026-09-17

Attempt 1 completed both subject processes inside the revised bound, with
process return code 0, but the verifier rejected both cells because their
closed-shape subject records were missing.

The direct diagnostic output showed that vLLM's log decorator prefixes printed
lines with the process identity. The runner required `GATE0_SUBJECT_JSON=` at
the beginning of a line, so it failed to parse a uniquely marked record that
appeared later in the line.

No child exit status was retained in the scored summary, so this attempt is not
used as mechanism evidence. The next revision locates exactly one marker within
each decorated line, adds a stdout hash, and keeps all predictions and subject
behavior unchanged.
