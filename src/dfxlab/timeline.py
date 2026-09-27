"""Render a replay-verified #53859 R3 bundle as a failure-path timeline.

The input must first pass the fail-closed ``replay`` verifier. The timeline
draws only facts present in the verified records: progress offsets, the
release offset and completion. Observations whose instant was not recorded
(the health probe and the stack sample) are drawn as the interval that the
campaign's control flow bounds them to, and labelled as such. Facts with no
recorded time (dropped event batches) are listed, never placed on the axis.
No edge between events is drawn: order in time is not a causal claim. Each
cell is an independent run aligned to its own request observer start.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from dfxlab.bundle import BundleError
from dfxlab.replay import CELLS, MANIFEST_NAME, ReplayError, replay
from dfxlab.timeline_bundle import build_bundle_timeline
from dfxlab.timeline_facts import to_trace, validate_timeline

TRACE_NAME = "timeline.trace.json"
HTML_NAME = "timeline.html"
STACK_SHAPE = "EngineCore → ZmqEventPublisher.publish → Queue.put"
BOUNDED = "sample occurred somewhere inside this window; not a duration"

NOT_ESTABLISHED = [
    "The instant the consumer pause began. The drawn gap is the observed "
    "interval between progress events, not the injection interval.",
    "The exact instants of the /health probe and the stack sample. Both are "
    "drawn as the interval the campaign's control flow bounds them to.",
    "When the fix arm dropped its event batches. Only the count is recorded.",
    "The EngineCore health ping from PR #36451. It was tested in a separate "
    "campaign and is not part of this bundle.",
    "Detection by vLLM's Prometheus metrics. No metrics series was retained "
    "(not scored, not failed).",
    "Repeatability. Each cell ran once.",
]


def _cell_title(index: int, arm: str, trigger: str) -> str:
    return f"Cell {index} · {arm} arm · {trigger}"


def build_timeline(result_dir: Path) -> dict[str, Any]:
    """Verify ``result_dir`` with ``replay`` and return a normalized model."""
    manifest_path = result_dir / MANIFEST_NAME
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as error:
        raise ReplayError("cannot read replay manifest") from error
    verified_lines = replay(result_dir)
    try:
        manifest_unchanged = manifest_path.read_bytes() == manifest_bytes
    except OSError as error:
        raise ReplayError("replay manifest disappeared after verification") from error
    if not manifest_unchanged:
        raise ReplayError("replay manifest changed during verification")
    manifest = json.loads(manifest_bytes)
    cells = []
    for label, (index, arm, trigger) in CELLS.items():
        entry = manifest["files"][label]
        path = result_dir / entry["path"]
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise ReplayError(f"{label}: evidence disappeared after replay") from error
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ReplayError(f"{label}: evidence changed after replay")
        record = json.loads(raw)
        progress = record["progress"]
        offsets = [float(value) for value in progress["progress_offsets_seconds"]]
        completed = float(progress["completed_offset_seconds"])
        release = record["release_offset_seconds"]
        cell: dict[str, Any] = {
            "label": label,
            "index": index,
            "arm": arm,
            "trigger": trigger,
            "title": _cell_title(index, arm, trigger),
            "progress_offsets": offsets,
            "completed": completed,
            "release": None if release is None else float(release),
            "gap": None,
            "bounded": [],
            "untimed": [],
        }
        if record["stalled_before_release"]:
            before = record["progress_count_at_release"]
            if not 0 < before < len(offsets):
                raise ReplayError(f"{label}: release does not split progress")
            gap_start, gap_end = offsets[before - 1], offsets[before]
            cell["gap"] = {"start": gap_start, "end": gap_end}
            window = {"start": gap_start, "end": cell["release"]}
            cell["bounded"] = [
                {
                    **window,
                    "name": f"GET /health → {record['health_during_stall']}",
                    "kind": "health",
                },
                {**window, "name": f"Stack: {STACK_SHAPE}", "kind": "stack"},
            ]
        if record["dropped_batch_count"]:
            cell["untimed"].append(
                f"{record['dropped_batch_count']} KV-event batches dropped"
            )
        if cell["release"] is not None and cell["release"] >= completed:
            cell["untimed"].append("Request had completed before the release")
        cells.append(cell)
    return {
        "case_id": manifest["case_id"],
        "title": manifest["title"],
        "source_links": manifest["source_links"],
        "evidence_sha256": {
            label: manifest["files"][label]["sha256"] for label in CELLS
        },
        "verified": verified_lines,
        "not_established": NOT_ESTABLISHED,
        "cells": cells,
    }


def to_fact_timeline(model: dict[str, Any]) -> dict[str, Any]:
    """Adapt the verified R3 replay to the same fact contract as native bundles."""
    lanes = []
    for cell in model["cells"]:

        def fact(
            kind: str, name: str, start: float | None, end: float | None = None
        ) -> dict[str, Any]:
            return {"kind": kind, "name": name, "start": start, "end": end}

        progress = [
            fact("instant", "progress event", t) for t in cell["progress_offsets"]
        ]
        progress.append(fact("instant", "request completed", cell["completed"]))
        if cell["gap"] is not None:
            progress.append(
                fact(
                    "observed_gap",
                    "no token progress observed between samples",
                    cell["gap"]["start"],
                    cell["gap"]["end"],
                )
            )
        control = []
        if cell["release"] is not None:
            control.append(fact("instant", "consumer released", cell["release"]))
        control.extend(fact("untimed", name, None) for name in cell["untimed"])
        health = [
            fact("possible_window", item["name"], item["start"], item["end"])
            for item in cell["bounded"]
            if item["kind"] == "health"
        ]
        stack = [
            fact("possible_window", item["name"], item["start"], item["end"])
            for item in cell["bounded"]
            if item["kind"] == "stack"
        ]
        lanes.append(
            {
                "id": cell["index"],
                "title": (
                    f"Independent run {cell['index']} · {cell['arm']}/{cell['trigger']}"
                ),
                "origin": "request observer start",
                "tracks": [
                    {"id": 1, "title": "token progress", "facts": progress},
                    {"id": 2, "title": "fault control", "facts": control},
                    {"id": 3, "title": "possible /health sample time", "facts": health},
                    {"id": 4, "title": "possible stack sample time", "facts": stack},
                ],
            }
        )
    facts = {
        "schema_version": 1,
        "title": "#53859: KV-event backpressure (four independent runs)",
        "source_kind": "verified_r3_replay_v1",
        "lanes": lanes,
        "notes": model["not_established"]
        + [
            "The four independent runs are not synchronized to a common clock.",
            "Evidence digests: " + ", ".join(model["evidence_sha256"].values()),
        ],
    }
    validate_timeline(facts)
    return facts


def to_trace_events(model: dict[str, Any]) -> dict[str, Any]:
    """Return Chrome Trace Event JSON through the case-neutral renderer."""
    return to_trace(to_fact_timeline(model))


# ---------------------------------------------------------------- HTML view

_LEFT, _RIGHT, _ROW, _TOP = 190, 940, 104, 44


def _x(seconds: float, axis_max: float) -> float:
    return _LEFT + (_RIGHT - _LEFT) * seconds / axis_max


def _svg(model: dict[str, Any]) -> str:
    axis_max = max(cell["completed"] for cell in model["cells"]) * 1.04
    height = _TOP + _ROW * len(model["cells"]) + 36
    esc = html.escape
    parts = [
        f'<svg viewBox="0 0 960 {height}" role="img" '
        'aria-label="Per-cell timeline of token progress, release and '
        'bounded observations">',
        '<defs><pattern id="hatch" width="6" height="6" '
        'patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
        '<rect width="6" height="6" class="gap-fill"/>'
        '<line x1="0" y1="0" x2="0" y2="6" class="gap-line"/></pattern></defs>',
    ]
    step = 1 if axis_max <= 14 else 2
    tick = 0
    while tick <= axis_max:
        x = _x(tick, axis_max)
        for row in range(len(model["cells"])):
            top = _TOP + row * _ROW
            parts.append(
                f'<line x1="{x:.1f}" y1="{top + 10}" x2="{x:.1f}" '
                f'y2="{top + 88}" class="grid"/>'
            )
        parts.append(f'<text x="{x:.1f}" y="{_TOP - 14}" class="tick">{tick} s</text>')
        tick += step
    for row, cell in enumerate(model["cells"]):
        top = _TOP + row * _ROW
        lane_progress, lane_obs = top + 30, top + 66
        parts.append(
            f'<text x="12" y="{top + 26}" class="cell-title">'
            f"{esc(cell['title'])}</text>"
        )
        parts.append(
            f'<text x="12" y="{top + 44}" class="cell-sub">'
            "independent run</text>"
            f'<text x="12" y="{top + 59}" class="cell-sub">'
            "own t=0: request start</text>"
        )
        parts.append(
            f'<line x1="{_LEFT}" y1="{lane_progress}" x2="{_RIGHT}" '
            f'y2="{lane_progress}" class="lane"/>'
        )
        if cell["gap"] is not None:
            gap = cell["gap"]
            x0, x1 = _x(gap["start"], axis_max), _x(gap["end"], axis_max)
            parts.append(
                f'<rect x="{x0:.1f}" y="{lane_progress - 12}" '
                f'width="{x1 - x0:.1f}" height="24" rx="3" fill="url(#hatch)" '
                'class="gap-box"><title>No token progress observed between '
                f"two recorded progress events ({gap['start']:.2f}–"
                f"{gap['end']:.2f} s)"
                "</title></rect>"
                f'<text x="{(x0 + x1) / 2:.1f}" y="{lane_progress + 4}" '
                'class="gap-label">no token progress observed · '
                f"{gap['end'] - gap['start']:.2f} s</text>"
                f'<text x="{(x0 + x1) / 2:.1f}" y="{lane_progress - 18}" '
                'class="gap-endpoints">between progress events: '
                f"{gap['start']:.2f} → {gap['end']:.2f} s</text>"
            )
        for offset in cell["progress_offsets"]:
            x = _x(offset, axis_max)
            parts.append(
                f'<line x1="{x:.1f}" y1="{lane_progress - 8}" x2="{x:.1f}" '
                f'y2="{lane_progress + 8}" class="progress"/>'
            )
        done = _x(cell["completed"], axis_max)
        parts.append(
            f'<circle cx="{done:.1f}" cy="{lane_progress}" r="5" class="done">'
            f"<title>Request completed at {cell['completed']:.2f} s</title>"
            "</circle>"
        )
        if cell["release"] is not None:
            x = _x(cell["release"], axis_max)
            # Keep the label inside the plot: anchor it left of the marker
            # when the marker sits near the right edge.
            near_edge = x > _RIGHT - 190
            label_x, anchor = (x - 5, "end") if near_edge else (x + 5, "start")
            parts.append(
                f'<line x1="{x:.1f}" y1="{top + 12}" x2="{x:.1f}" '
                f'y2="{lane_obs + 12}" class="release"/>'
                f'<text x="{label_x:.1f}" y="{top + 16}" text-anchor="{anchor}" '
                f'class="release-label">'
                f"consumer released · {cell['release']:.2f} s</text>"
            )
        for slot, item in enumerate(cell["bounded"]):
            x0 = _x(item["start"], axis_max)
            x1 = _x(item["end"], axis_max)
            y = lane_obs - 10 + slot * 14
            parts.append(
                f'<path d="M {x0:.1f} {y + 11} V {y} H {x1:.1f} V {y + 11}" '
                f'class="bounded {item["kind"]}"><title>{esc(item["name"])}'
                f" — {BOUNDED}</title></path>"
                f'<text x="{x0 + 6:.1f}" y="{y + 9}" class="bounded-label">'
                f"{esc(item['name'])} · possible time window</text>"
            )
    parts.append(
        f'<text x="{_RIGHT}" y="{height - 10}" class="axis-note">'
        "seconds since the request observer started · one run per cell</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


_CSS = """
:root{--bg:#f7f6f2;--panel:#ffffff;--ink:#1d1d1b;--muted:#6b6a64;--line:#dedcd4;
--progress:#2b2b28;--gap:#f3c969;--gap-ink:#8a5a00;--health:#3f9b62;
--stack:#5b6fd6;--release:#b0413e;--done:#1d1d1b;--pass:#2f7d4f;--warn:#9a6700}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--bg:#161614;--panel:#1f1f1c;--ink:#ecebe6;--muted:#a09e96;--line:#34332f;
--progress:#dddcd6;--gap:#6b5214;--gap-ink:#f3c969;--health:#58b87c;
--stack:#8c9cf0;--release:#ef7a74;--done:#ecebe6;--pass:#6fcf97;--warn:#e0b04c}}
:root[data-theme="dark"]{--bg:#161614;--panel:#1f1f1c;--ink:#ecebe6;
--muted:#a09e96;--line:#34332f;--progress:#dddcd6;--gap:#6b5214;
--gap-ink:#f3c969;--health:#58b87c;--stack:#8c9cf0;--release:#ef7a74;
--done:#ecebe6;--pass:#6fcf97;--warn:#e0b04c}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1040px;margin:0 auto;padding:32px 16px 48px}
h1{font-size:1.45rem;margin:0 0 4px;letter-spacing:-.01em}
.lede{color:var(--muted);margin:0 0 20px}
.lede strong{color:var(--ink)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
padding:16px}
.plot{overflow-x:auto}
svg{width:100%;min-width:760px;height:auto;display:block}
.grid{stroke:var(--line);stroke-width:1}
.tick,.axis-note{fill:var(--muted);font-size:11px}
.tick{text-anchor:middle}.axis-note{text-anchor:end}
.cell-title{fill:var(--ink);font-size:13px;font-weight:600}
.cell-sub{fill:var(--muted);font-size:11.5px}
.lane{stroke:var(--line);stroke-width:1}
.progress{stroke:var(--progress);stroke-width:1.2}
.done{fill:var(--done)}
.gap-fill{fill:var(--gap);opacity:.35}.gap-line{stroke:var(--gap);stroke-width:2}
.gap-box{stroke:var(--gap-ink);stroke-width:1}
.gap-label{fill:var(--gap-ink);font-size:12px;font-weight:600;text-anchor:middle}
.gap-endpoints{fill:var(--gap-ink);font-size:10.5px;text-anchor:middle}
.release{stroke:var(--release);stroke-width:1.5;stroke-dasharray:4 3}
.release-label{fill:var(--release);font-size:11.5px}
.bounded{fill:none;stroke-width:1.5;stroke-dasharray:4 3}
.observed{fill:none;stroke:var(--gap-ink);stroke-width:2}
.analysis-window{fill:none;stroke:var(--muted);stroke-width:1;stroke-dasharray:1 4}
.bounded.health{stroke:var(--health)}.bounded.stack{stroke:var(--stack)}
.bounded-label{fill:var(--ink);font-size:10.5px}
.untimed{fill:var(--warn);font-size:11.5px;font-weight:600}
.plot-summary{margin:0 0 10px;color:var(--ink);font-size:13px}
.plot-note{margin:10px 0 0;color:var(--muted);font-size:13px}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;margin:12px 0 0;padding:0;
list-style:none;color:var(--muted);font-size:13px}
.legend span{display:inline-block;width:14px;height:10px;margin-right:6px;
vertical-align:-1px;border-radius:2px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}
@media (max-width:720px){.cols{grid-template-columns:1fr}}
h2{font-size:1rem;margin:0 0 8px}
ul.facts{margin:0;padding-left:18px}ul.facts li{margin:4px 0}
.ok{color:var(--pass);font-weight:600}.no{color:var(--warn);font-weight:600}
.legend .dot{background:var(--done);border-radius:50%;width:10px}
footer{margin-top:16px;color:var(--muted);font-size:13px}
code{font-size:.92em}a{color:inherit}
"""


def render_html(model: dict[str, Any]) -> str:
    esc = html.escape
    verified = "".join(
        f"<li><span class=ok>✓</span> {esc(line)}</li>"
        for line in model["verified"]
        if not line.startswith(("CASE:", "BOUNDARY:"))
    )
    missing = "".join(
        f"<li><span class=no>?</span> {esc(line)}</li>"
        for line in model["not_established"]
    )
    links = " · ".join(
        f'<a href="{esc(url)}">{esc(url.rsplit("/", 2)[-2])}'
        f" #{esc(url.rsplit('/', 1)[-1])}</a>"
        for url in model["source_links"]
    )
    digests = "".join(
        f"<li><code>{esc(label)}</code> {esc(digest[:16])}…</li>"
        for label, digest in model["evidence_sha256"].items()
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Failure Path Timeline</title><style>{_CSS}</style></head>
<body><main>
<h1>#53859 · four independent failure-path runs</h1>
<p class="lede">Case <code>{esc(model["case_id"])}</code>. In the base arm,
<strong>no token progress was observed between 0.31 and 10.38 s; one
<code>/health</code> probe inside that window returned 2xx</strong>. The fix arm
completed under the same trigger but dropped event batches. Everything drawn
here comes from a bundle that passed the fail-closed replay check.</p>
<div class="card"><p class="plot-summary"><strong>Four independent runs; each
starts its own clock at its request observer.</strong> Cell 3: 10.07 s without
observed token progress (0.31 → 10.38 s). Scroll the detailed timeline →</p>
<div class="plot">{_svg(model)}</div>
<p class="plot-note">Untimed, Cell 4: {esc("; ".join(model["cells"][3]["untimed"]))}.
In Cell 3, consumer release was a harness control action; progress resumed
later in that run. The graphic alone does not prove causation.</p>
<ul class="legend">
<li><span style="background:var(--progress)"></span>progress event</li>
<li><span style="background:var(--gap);opacity:.6"></span>no progress observed</li>
<li><span style="border-top:2px dashed var(--release);height:0"></span>release</li>
<li><span style="border-top:2px dashed var(--health);height:0"></span>
/health possible time</li>
<li><span style="border-top:2px dashed var(--stack);height:0"></span>
stack possible time</li>
<li><span class="dot"></span>completed</li>
</ul></div>
<div class="cols">
<section class="card"><h2>Verified by replay</h2><ul class="facts">{verified}</ul>
</section>
<section class="card"><h2>Not established, or not drawn</h2>
<ul class="facts">{missing}</ul></section>
</div>
<footer><p>Events are placed in time order only; no arrow or edge asserts
causation. Dashed brackets mark possible sample times, not continuous states.
Each cell is an independent run aligned to its own request observer start;
the trace's process IDs are synthetic lane ordinals. The same facts are in
<code>{TRACE_NAME}</code>
for <a href="https://ui.perfetto.dev">ui.perfetto.dev</a>.</p>
<p>Sources: {links}. Evidence files (SHA-256 prefix):</p>
<ul class="facts">{digests}</ul></footer>
</main></body></html>
"""


def render_fact_html(model: dict[str, Any]) -> str:
    """Small case-neutral SVG view; uncertainty is a bracket, not a state bar."""
    validate_timeline(model)
    timed = [
        fact
        for lane in model["lanes"]
        for track in lane["tracks"]
        for fact in track["facts"]
        if fact["kind"] != "untimed"
    ]
    axis_max = (
        max(
            (fact["end"] if fact["end"] is not None else fact["start"])
            for fact in timed
        )
        if timed
        else 1.0
    )
    axis_max = max(axis_max, 0.001)
    rows: list[str] = []
    y = 40
    for lane in model["lanes"]:
        lane_title = html.escape(lane["title"])
        rows.append(f'<text x="10" y="{y}" class="cell-title">{lane_title}</text>')
        y += 26
        for track in lane["tracks"]:
            track_title = html.escape(track["title"])
            rows.append(
                f'<text x="10" y="{y + 4}" class="cell-sub">{track_title}</text>'
                f'<line x1="190" y1="{y}" x2="940" y2="{y}" class="lane"/>'
            )
            for fact in track["facts"]:
                if fact["kind"] == "untimed":
                    continue
                x0 = 190 + 750 * fact["start"] / axis_max
                label = html.escape(fact["name"])
                if fact["kind"] == "instant":
                    rows.append(
                        f'<circle cx="{x0:.1f}" cy="{y}" r="4" class="done">'
                        f"<title>{label}: {fact['start']:.3f}s</title></circle>"
                    )
                else:
                    x1 = 190 + 750 * fact["end"] / axis_max
                    css = {
                        "possible_window": "bounded",
                        "observed_gap": "observed",
                        "analysis_window": "analysis-window",
                    }[fact["kind"]]
                    rows.append(
                        f'<path d="M {x0:.1f} {y + 7} V {y - 7} H {x1:.1f} '
                        f'V {y + 7}" class="{css}">'
                        f"<title>{label}: {fact['kind']}, "
                        f"{fact['start']:.3f}–{fact['end']:.3f}s</title></path>"
                    )
            y += 40
        y += 12
    notes = "".join(f"<li>{html.escape(note)}</li>" for note in model["notes"])
    untimed = "".join(
        f"<li>{html.escape(fact['name'])}</li>"
        for lane in model["lanes"]
        for track in lane["tracks"]
        for fact in track["facts"]
        if fact["kind"] == "untimed"
    )
    ticks = "".join(
        f'<line x1="{190 + 150 * index}" y1="28" '
        f'x2="{190 + 150 * index}" y2="{y - 15}" class="grid"/>'
        f'<text x="{190 + 150 * index}" y="19" class="tick">'
        f"{axis_max * index / 5:.3g} s</text>"
        for index in range(6)
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Verified evidence timeline</title><style>{_CSS}</style></head><body><main>
<h1>{html.escape(model["title"])}</h1>
<p class="lede">Verified source: {html.escape(model["source_kind"])}.
Dots are exact sample times; solid brackets show observed gaps; thin dotted
brackets mark analysis windows; dashed brackets bound possible sample times.
The axes use each lane's stated origin, not a shared wall clock.</p>
<div class="card plot"><svg viewBox="0 0 960 {y + 25}" role="img"
aria-label="Verified evidence timeline">{ticks}{"".join(rows)}</svg></div>
<div class="cols"><section class="card"><h2>Untimed claims</h2>
<ul class="facts">{untimed}</ul></section>
<section class="card"><h2>Limits</h2><ul class="facts">{notes}</ul></section></div>
<footer>Time order is not causation. Synthetic lanes are not OS processes.</footer>
</main></body></html>"""


def write_timeline(result_dir: Path, out_dir: Path) -> list[Path]:
    if out_dir.exists():
        raise ReplayError("output directory already exists; choose a new directory")
    if (result_dir / MANIFEST_NAME).is_file():
        model = build_timeline(result_dir)
        trace = to_trace_events(model)
        page = render_html(model)
    else:
        try:
            facts = build_bundle_timeline(result_dir)
        except BundleError as error:
            raise ReplayError(f"native bundle verification failed: {error}") from error
        trace = to_trace(facts)
        page = render_fact_html(facts)
    try:
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".timeline-", dir=out_dir.parent))
    except OSError as error:
        raise ReplayError("cannot prepare timeline output") from error
    published = False
    try:
        (temporary / TRACE_NAME).write_bytes(
            (json.dumps(trace, indent=1, sort_keys=True) + "\n").encode("utf-8")
        )
        (temporary / HTML_NAME).write_bytes(page.encode("utf-8"))
        if out_dir.exists():
            raise ReplayError("output directory appeared during rendering")
        temporary.rename(out_dir)
        published = True
    except OSError as error:
        raise ReplayError("cannot write complete timeline output") from error
    finally:
        if not published:
            for name in (TRACE_NAME, HTML_NAME):
                (temporary / name).unlink(missing_ok=True)
            temporary.rmdir()
    return [out_dir / TRACE_NAME, out_dir / HTML_NAME]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a replay-verified result as a failure-path timeline."
    )
    parser.add_argument("result_dir", type=Path)
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="new output directory"
    )
    args = parser.parse_args(argv)
    try:
        paths = write_timeline(args.result_dir.resolve(), args.out_dir)
    except ReplayError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
