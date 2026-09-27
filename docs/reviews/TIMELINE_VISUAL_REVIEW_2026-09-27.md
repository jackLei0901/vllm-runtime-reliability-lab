# Timeline visual review — 2026-09-27

Status: **revised prototype; HTML visual acceptance pending; Perfetto not
accepted as the explanatory view**. This record describes staged, uncommitted
Lab changes. It does not claim an independent-reader result.

## Review entry

- [Standalone #53859 page](../demos/53859-r3/timeline.html)
- [Chrome trace-event JSON](../demos/53859-r3/timeline.trace.json)
- [One-shot local Perfetto opener](../demos/53859-r3/open_in_perfetto.py)
- [Sample explanation and regeneration command](../demos/53859-r3/README.md)
- [Fact contract](../TIMELINE_FACT_CONTRACT.md)

The input is the public, replay-verified
[`vllm-zmq-backpressure-stage1-r3-20260916` bundle](../../results/vllm-zmq-backpressure-stage1-r3-20260916/).
This is four **independent** cells, not a synchronized multi-process run.
The page shows the base/pause progress gap, bounded health/stack sample times,
release, completion, and the fix arm's untimed dropped-batch count. It draws no
causal arrow. The #36451 ping campaign is not represented.

## Completed checks

| Gate | Result | Evidence |
| --- | --- | --- |
| Verify before render | Pass | CLI replayed the published bundle; tamper tests reject output |
| Deterministic regeneration | Pass | `vllm-dfx timeline` produced byte-identical HTML and trace from the public bundle |
| Public-data boundary | Pass for this sample | No raw host path, credential, request hash/content, or stack; trace IDs are synthetic cell ordinals 1–4 |
| Closed timing semantics | Pass in tests | 262 instant events, 3 intervals (one observed gap and two possible sample-time windows); unknown instants remain windows |
| Test suite | Pass | `python -m unittest discover -s tests -q`: 296 tests, 3 skipped on this checkout |
| Staged diff | Pass | Explicit path staging and `git diff --cached --check` clean |

Generated file SHA-256, as independently regenerated with the installed
`vllm-dfx` entry point:

| File | SHA-256 |
| --- | --- |
| `timeline.html` | `a2e3776fb9746e5c1aea798ddfcff073abc59c0dcc77950d9877dc10135fdbd1` |
| `timeline.trace.json` | `b7ea05a74fcdd8e688c60ec244e15c07be8edaa5d2813dbca1070eabb26aec8b` |

## First visual review and revision

Claude inspected the **previous** page at 1280 and 390 CSS px before reading
this record. It found three substantive failures: a shared grid and weak labels
made independent cells look synchronized; one `/health` sample was worded as a
continuously green endpoint; and the 10.07 s result was outside the phone's
initial viewport. The old plot also put untimed text on the axis under a
release marker. Its partial Perfetto import showed four process groups but
could not inspect expanded tracks. This was a failed first visual gate, not
an acceptance of the revised artifacts.

The revision changes the replay wording to **one probe returned 2xx inside the
no-progress window**; puts per-run identity and time origin inside each row;
breaks the grid between rows; places the 10.07 s result and gap endpoints
above the scrollable plot; moves untimed Cell 4 text outside the axis; prefixes
Perfetto lane names with `Independent run`; and drops metadata for empty
tracks. The Perfetto window is still a slice, explicitly named as the
possible time of **one** sample, not an observed continuous state. The sample
files are now written as UTF-8 with LF on every host, so the SHA-256 pins
refer to the same bytes after checkout.

The one-shot Perfetto helper's local HTTP/CORS and digest behavior was tested
without opening the browser. It does **not** prove that the revised trace
renders as intended in Perfetto.

The owner subsequently opened the revised trace in Perfetto and supplied a
screenshot of the default, collapsed view. It shows four independent-run
groups, but Cell 3's observed gap and two possible sample-time windows are
compressed into three nearly indistinguishable, unlabeled long bars. The
progress instants are too dense to read at this scale. Thus local import
succeeded, while the Perfetto explanatory-readability check **failed**. The
`local_cache_key` viewer URL is browser-local; the supplied screenshot, not
that URL, is the visual witness. The owner then supplied an expanded Cell 3
screenshot: the token-progress track visibly separates progress instants from
the observed gap (10.07 s in the trace); the `/health` and stack tracks each
show a separately labelled possible time of **one** sample, not a continuous state.
This passes the narrow expanded-track distinction check. The long solid
possible-time slices may still be misread as sustained states without their
labels, and Cells 1, 2, and 4 were not inspected expanded. The standalone
HTML remains the intended explanatory artifact; the Perfetto export is
optional event-level evidence, not a visual-acceptance result.

## Pending checks — do not upgrade to a visual-acceptance claim

1. **Reinspect revised page at 1280 and 390 CSS px.** Confirm the per-row
   origin/independence labels, broken grid, visible above-plot 10.07 s and
   0.31–10.38 s values, and untimed note outside the axis. Check that neither
   the headline nor Verified panel implies health was continuously 2xx. Phone
   scrolling may remain for detail but must not hide the main claim.
2. **Perfetto trace semantics (optional view).** Import and the expanded Cell 3
   track-distinction check succeeded; the collapsed-view readability check
   failed. If the export is retained, confirm no empty health/stack tracks in
   Cells 1, 2, or 4. Perfetto's one visual axis and slice styling must never
   be presented as the primary explanation or as proof of synchronized runs
   or continuous states.
3. **Independent reader.** Give the page to someone who did not build it.
   Before showing this answer key, ask:
   - What was observed during the base/pause cell when one `/health` probe returned 2xx?
   - Do the left and right edges of a dashed health/stack bracket identify
     when that sample happened?
   - Does the figure establish the pause's exact start time or prove a cause?
   - Are the four cells on one synchronized clock?

   Acceptance: the reader identifies the progress gap, says the dashed
   brackets only bound unknown sample instants, and rejects both exact-start
   and cross-cell causal/synchronization claims. Record their wording and
   any misreading before calling this gate complete.

The main project-wide Ruff command currently reports seven `E501` lines in
three **unstaged** TP-V2 test files; the timeline-related files pass Ruff.
Those TP files are outside this review set and must be fixed before a broad
commit/CI run includes them.
