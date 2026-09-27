# Verified evidence timeline: fact contract (prototype)

The timeline is a presentation of **already verified evidence**, not another
verdict engine. Run `vllm-dfx timeline SOURCE --out-dir NEW_DIRECTORY`. `SOURCE`
may be the published #53859 R3 replay directory or a native v0.2 incident
bundle. The command writes a standalone `timeline.html` and Chrome Trace Event
`timeline.trace.json`; it refuses to overwrite an existing output directory.
Both files are written in a temporary sibling directory and published by one
rename, so a failed write cannot leave a public half-rendered output.

Both adapters must verify their input first (`replay` for R3, `verify_bundle` for
v0.2), then compare the bytes actually projected with verified digests. They
then project onto the same closed fact vocabulary:

| Fact kind | Meaning | Trace form |
| --- | --- | --- |
| `instant` | An observation has a recorded time | Instant marker |
| `observed_gap` | No relevant progress event observed in a bounded interval by an available producer | Solid bracket |
| `analysis_window` | Interval selected for evaluating a verdict, not a measured fault state | Thin dotted bracket |
| `possible_window` | One observation occurred somewhere within these bounds | Span labelled *possible time*, not continuous state |
| `untimed` | Verified fact with no defensible time | Text/metadata, never an axis marker |

Every lane declares its time origin. R3 cells are independent runs aligned to
their own request observer start; their offsets are **not** a synchronized
multi-process timeline. The native bundle uses monotonic offsets from its
collection start. Trace process/thread IDs are synthetic ordinals, not target
PIDs or OS thread IDs. Temporal order alone never adds a causal edge.
Adapter labels must describe observations, never assert an unverified cause;
the closed shape cannot police free-text wording, so adapter review is required.

The v0.2 adapter exports normalized process, health, and decision-producer
sample events plus the evaluation interval. Its recomputed verdict and stack
availability remain untimed claims. It does not export PID, process start
identity, endpoint ID, client request digest, or raw stack. A missing producer
does not become a missing process or an invented event.
An available request producer that observed no chunks (or only non-content
chunks) in the evaluation window gets an explicit `observed_gap`, bounded to
the intersection of the evaluation window and the request's open lifetime.
If the request was not open for the whole window, an untimed fact says so; an
unavailable decision producer is an explicit untimed fact. An empty track
must never stand in for either claim.
For a server counter, a `flat/equal_samples` decision gets an observed gap
between its first and last fresh sample, and each fresh sample after the
baseline carries its delta. The verifier requires both producer and demand
windows to equal the bundle's evaluation window; the adapter still projects
from the selected producer's own verified window.
Observations outside the collection window are counted as untimed facts,
not silently discarded or assigned a negative offset.

This is a two-format capability test, **not** a promise to visualize every Lab
result. A third format needs a verified adapter that can name each fact's time
status and origin. If those cannot be established, the adapter must leave the
fact untimed or decline to render it. The #53859 explanatory page remains
case-specific; only the fact model and trace renderer are shared so far.
