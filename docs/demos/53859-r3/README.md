# #53859: verified failure-path timeline

Open [the standalone page](timeline.html) for the four-cell comparison. The
[trace](timeline.trace.json) can also be loaded into
[Perfetto](https://ui.perfetto.dev/) for event-level inspection, but it is
**not the explanatory view**: its shared axis, dense progress markers, and
slice rendering make the failure harder to read. The inputs are the published
[`vllm-zmq-backpressure-stage1-r3-20260916` replay bundle](../../../results/vllm-zmq-backpressure-stage1-r3-20260916/).

The base/pause cell shows recorded progress events, a 10.07-second interval
without token progress, a bounded (not precisely timed) health and stack
observation, consumer release, and eventual completion. The fix/pause cell
completes faster but records four dropped event batches. The cells are
**independent runs** aligned to their own request-observer starts; the picture
does not establish cross-cell simultaneity or causation.
Perfetto places them on one shared visual zero because this export uses a
single trace time axis; that alignment is for comparison, not a common clock.
Its possible-time windows render as slices, not continuous observed states.
Do not infer a continuous health state, synchronized cells, or causation from
that view. Prefer the standalone page when sharing this case with a reader.

On Windows, run this one-shot local opener from the repository root to check
the current pinned trace in Perfetto:

```console
.venv/Scripts/python.exe docs/demos/53859-r3/open_in_perfetto.py
```

It verifies the trace SHA-256, serves that file only on `127.0.0.1:9001`,
opens the Perfetto UI, then stops after one successful fetch or two minutes.
Perfetto's own [command-line opener](https://perfetto.dev/docs/faq#how-do-i-open-trace-in-ui-from-command-line)
uses the same localhost port and URL approach. The browser view still needs
human inspection; successful serving is not a UI-import acceptance result.

In the reviewed default view, Perfetto collapsed each `Independent run`
group. To inspect the base/pause evidence, click the disclosure arrow beside
the `Independent run 3` group. Read the progress events and observed gap on
the token-progress track separately from the health and stack possible-time
tracks. The collapsed summary bars cannot distinguish an observed no-progress
interval from a range containing
one untimed sample. Even expanded, use the standalone page for the claim and
Perfetto only to inspect individual events.

To regenerate from the committed input bundle:

```console
vllm-dfx timeline results/vllm-zmq-backpressure-stage1-r3-20260916 --out-dir out/timeline-53859-rebuilt
```

The command verifies the replay before rendering and refuses an existing
output directory. The HTML and trace contain normalized public facts, not
raw server logs, process identities, or request content. This is a display
example, not a new verdict or a #36451 health-ping result.
