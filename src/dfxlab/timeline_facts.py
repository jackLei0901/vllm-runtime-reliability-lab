"""Closed, case-neutral facts for rendering *verified* evidence as a trace.

Adapters own verification and provenance. This module never reads raw evidence,
infers a cause, or turns an unknown observation time into an exact event.
"""

from __future__ import annotations

import math
from typing import Any


class TimelineError(ValueError):
    """A timeline adapter supplied an invalid or over-specific fact."""


EVENT_KINDS = {
    "instant",
    "observed_gap",
    "analysis_window",
    "possible_window",
    "untimed",
}


def _seconds(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise TimelineError(f"invalid {name}")
    return float(value)


def validate_timeline(model: dict[str, Any]) -> None:
    """Require a small closed shape; unknown time may not masquerade as an instant."""
    if set(model) != {"schema_version", "title", "source_kind", "lanes", "notes"}:
        raise TimelineError("timeline shape changed")
    if model["schema_version"] != 1 or not all(
        isinstance(model[key], str) and model[key] for key in ("title", "source_kind")
    ):
        raise TimelineError("timeline identity invalid")
    if not isinstance(model["notes"], list) or not all(
        isinstance(note, str) for note in model["notes"]
    ):
        raise TimelineError("timeline notes invalid")
    if not isinstance(model["lanes"], list) or not model["lanes"]:
        raise TimelineError("timeline has no lanes")
    lane_ids: set[int] = set()
    for lane in model["lanes"]:
        if set(lane) != {"id", "title", "origin", "tracks"}:
            raise TimelineError("lane shape changed")
        pid = lane["id"]
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            raise TimelineError("lane id invalid")
        if pid in lane_ids:
            raise TimelineError("duplicate lane id")
        lane_ids.add(pid)
        if not all(
            isinstance(lane[key], str) and lane[key] for key in ("title", "origin")
        ):
            raise TimelineError("lane label invalid")
        if not isinstance(lane["tracks"], list) or not lane["tracks"]:
            raise TimelineError("lane has no tracks")
        track_ids: set[int] = set()
        for track in lane["tracks"]:
            if set(track) != {"id", "title", "facts"}:
                raise TimelineError("track shape changed")
            tid = track["id"]
            if not isinstance(tid, int) or isinstance(tid, bool) or tid <= 0:
                raise TimelineError("track id invalid")
            if tid in track_ids:
                raise TimelineError("duplicate track id")
            track_ids.add(tid)
            if not isinstance(track["title"], str) or not track["title"]:
                raise TimelineError("track title invalid")
            if not isinstance(track["facts"], list):
                raise TimelineError("track facts invalid")
            for fact in track["facts"]:
                if set(fact) != {"kind", "name", "start", "end"}:
                    raise TimelineError("fact shape changed")
                kind = fact["kind"]
                if (
                    kind not in EVENT_KINDS
                    or not isinstance(fact["name"], str)
                    or not fact["name"]
                ):
                    raise TimelineError("fact label invalid")
                if kind == "untimed":
                    if fact["start"] is not None or fact["end"] is not None:
                        raise TimelineError("untimed fact has a time")
                elif kind == "instant":
                    _seconds(fact["start"], "instant")
                    if fact["end"] is not None:
                        raise TimelineError("instant has a duration")
                else:
                    start = _seconds(fact["start"], "start")
                    end = _seconds(fact["end"], "end")
                    if end <= start:
                        raise TimelineError("interval is empty")


def to_trace(model: dict[str, Any]) -> dict[str, Any]:
    """Render normalized facts as Chrome Trace Event JSON (synthetic lanes)."""
    validate_timeline(model)
    events: list[dict[str, Any]] = []
    untimed: list[dict[str, Any]] = []
    for lane in model["lanes"]:
        pid = lane["id"]
        events.append(
            {
                "ph": "M",
                "pid": pid,
                "name": "process_name",
                "args": {"name": lane["title"]},
            }
        )
        events.append(
            {
                "ph": "M",
                "pid": pid,
                "name": "process_sort_index",
                "args": {"sort_index": pid},
            }
        )
        for track in lane["tracks"]:
            tid = track["id"]
            if any(fact["kind"] != "untimed" for fact in track["facts"]):
                events.append(
                    {
                        "ph": "M",
                        "pid": pid,
                        "tid": tid,
                        "name": "thread_name",
                        "args": {"name": track["title"]},
                    }
                )
            for fact in track["facts"]:
                kind = fact["kind"]
                if kind == "untimed":
                    untimed.append({"lane": pid, "track": tid, "name": fact["name"]})
                    continue
                event: dict[str, Any] = {
                    "ph": "I" if kind == "instant" else "X",
                    "pid": pid,
                    "tid": tid,
                    "name": fact["name"],
                    "ts": round(fact["start"] * 1_000_000),
                    "args": {"fact_kind": kind},
                }
                if kind == "instant":
                    event["s"] = "t"
                else:
                    event["dur"] = round((fact["end"] - fact["start"]) * 1_000_000)
                    if event["dur"] <= 0:
                        raise TimelineError("interval below trace resolution")
                    if kind == "possible_window":
                        event["name"] = (
                            "Possible time of one sample (not a state): " + fact["name"]
                        )
                        event["args"]["window_only"] = True
                events.append(event)
    return {
        "traceEvents": events,
        "displayTimeUnit": "ms",
        "otherData": {
            "source_kind": model["source_kind"],
            "title": model["title"],
            "lane_identity": "synthetic ordinals; independent runs are not OS PIDs",
            "lane_origins": {
                str(lane["id"]): lane["origin"] for lane in model["lanes"]
            },
            "untimed_facts": untimed,
            "notes": model["notes"],
        },
    }
