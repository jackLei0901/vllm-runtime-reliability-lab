"""Project a verified v0.2 incident bundle onto case-neutral timeline facts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from dfxlab.bundle import BundleError
from dfxlab.timeline_facts import validate_timeline
from dfxlab.verify_bundle import verify_bundle


def build_bundle_timeline(bundle_dir: Path) -> dict[str, Any]:
    """Render only validated, publishable observations; omit target identities."""
    summary = verify_bundle(bundle_dir)
    try:
        raw = (bundle_dir / "observations.json").read_bytes()
    except OSError as error:
        raise BundleError("observations disappeared after verification") from error
    if (
        hashlib.sha256(raw).hexdigest()
        != summary["sources"]["observations.json"]["sha256"]
    ):
        raise BundleError("observations changed after verification")
    observations = json.loads(raw)
    collection = observations["intervals"]["collection"]
    origin, collection_end = collection["start_ns"], collection["end_ns"]
    evaluation = observations["intervals"]["evaluation"]

    def in_collection(timestamp: int) -> bool:
        return origin <= timestamp <= collection_end

    def out_of_window(name: str, timestamps: list[int]) -> list[dict[str, Any]]:
        before = sum(timestamp < origin for timestamp in timestamps)
        after = sum(timestamp > collection_end for timestamp in timestamps)
        return [
            untimed(f"{count} {name} before collection window")
            for count in (before,)
            if count
        ] + [
            untimed(f"{count} {name} after collection window")
            for count in (after,)
            if count
        ]

    def offset(value: int) -> float:
        return (value - origin) / 1_000_000_000

    def instant(name: str, timestamp: int) -> dict[str, Any]:
        return {
            "kind": "instant",
            "name": name,
            "start": offset(timestamp),
            "end": None,
        }

    def untimed(name: str) -> dict[str, Any]:
        return {"kind": "untimed", "name": name, "start": None, "end": None}

    process_samples = observations["process"]["samples"]
    process = [
        instant(
            "process sampled: "
            + ("alive" if sample["alive"] else "not alive")
            + (" (stale)" if not sample["fresh"] else ""),
            sample["monotonic_ns"],
        )
        for sample in process_samples
        if in_collection(sample["monotonic_ns"])
    ]
    process += out_of_window(
        "process samples", [item["monotonic_ns"] for item in process_samples]
    )
    health_samples = observations["health"]["samples"]
    health = [
        instant(
            "health sampled: "
            + ("ok" if sample["ok"] else "not ok")
            + (" (stale)" if not sample["fresh"] else ""),
            sample["monotonic_ns"],
        )
        for sample in health_samples
        if in_collection(sample["monotonic_ns"])
    ]
    health += out_of_window(
        "health samples", [item["monotonic_ns"] for item in health_samples]
    )
    producers = observations["producer_inputs"]
    decision_kind = producers["decision_source"]
    decision = producers[decision_kind]
    decision_state = summary["producers"]["decision"]
    decision_start = decision["evaluation_start_ns"]
    decision_end = decision["evaluation_end_ns"]
    if not decision["producer_available"]:
        progress = [untimed("decision producer unavailable")]
        if decision_kind == "client_request":
            started = decision["request_started_ns"]
            if in_collection(started):
                progress.append(instant("request started", started))
            else:
                progress += out_of_window("request starts", [started])
            if decision["request_completed_ns"] is not None:
                completed = decision["request_completed_ns"]
                if in_collection(completed):
                    progress.append(instant("request completed", completed))
                else:
                    progress += out_of_window("request completions", [completed])
    elif decision_kind == "server_counter":
        progress = []
        previous_fresh: float | None = None
        fresh_samples = []
        for sample in decision["samples"]:
            if not in_collection(sample["monotonic_ns"]):
                continue
            if sample["fresh"]:
                fresh_samples.append(sample)
                label = (
                    "counter sampled (baseline)"
                    if previous_fresh is None
                    else f"counter change {sample['value'] - previous_fresh:+g}"
                )
                previous_fresh = sample["value"]
            else:
                label = "counter sampled (stale; not evaluated)"
            progress.append(instant(label, sample["monotonic_ns"]))
        progress += out_of_window(
            "counter samples", [item["monotonic_ns"] for item in decision["samples"]]
        )
        if decision_state == {"state": "flat", "reason": "equal_samples"}:
            progress.append(
                {
                    "kind": "observed_gap",
                    "name": "counter did not increase across fresh samples",
                    "start": offset(fresh_samples[0]["monotonic_ns"]),
                    "end": offset(fresh_samples[-1]["monotonic_ns"]),
                }
            )
        if not progress:
            progress.append(untimed("decision producer available; no counter samples"))
    else:
        chunks = decision["chunks"]
        progress = [
            instant("request chunk: " + chunk["kind"], chunk["monotonic_ns"])
            for chunk in chunks
            if in_collection(chunk["monotonic_ns"])
        ]
        progress += out_of_window(
            "request chunks", [item["monotonic_ns"] for item in chunks]
        )
        in_evaluation = [
            item
            for item in chunks
            if decision_start <= item["monotonic_ns"] <= decision_end
        ]
        started = decision["request_started_ns"]
        completed = decision["request_completed_ns"]
        open_start = max(decision_start, started)
        open_end = (
            min(decision_end, completed) if completed is not None else decision_end
        )
        if started > decision_start or (
            completed is not None and completed <= decision_end
        ):
            progress.append(untimed("request not open for the full evaluation window"))
        if not any(item["kind"] == "content" for item in in_evaluation):
            label = (
                "no request chunks observed (producer available)"
                if not in_evaluation
                else "no content chunks observed (producer available)"
            )
            if open_end > open_start:
                progress.append(
                    {
                        "kind": "observed_gap",
                        "name": label,
                        "start": offset(open_start),
                        "end": offset(open_end),
                    }
                )
        if started is not None:
            if in_collection(started):
                progress.append(instant("request started", started))
            else:
                progress += out_of_window("request starts", [started])
        if completed is not None:
            if in_collection(completed):
                progress.append(instant("request completed", completed))
            else:
                progress += out_of_window("request completions", [completed])
    progress.append(
        untimed(
            f"decision producer state: {decision_state['state']} "
            f"({decision_state['reason']})"
        )
    )
    verdict = summary["verdict"]["verdict"]
    tracks = [
        {"id": 1, "title": "process samples", "facts": process},
        {"id": 2, "title": "health samples", "facts": health},
        {"id": 3, "title": decision_kind + " observations", "facts": progress},
        {
            "id": 4,
            "title": "evaluation and claims",
            "facts": [
                {
                    "kind": "analysis_window",
                    "name": "evaluation window (not fault duration)",
                    "start": offset(evaluation["start_ns"]),
                    "end": offset(evaluation["end_ns"]),
                },
                untimed("verified verdict: " + verdict),
                untimed("stack producer: " + summary["stack"]["state"]),
            ],
        },
    ]
    model = {
        "schema_version": 1,
        "title": "v0.2 incident evidence",
        "source_kind": "verified_incident_bundle_v1",
        "lanes": [
            {
                "id": 1,
                "title": "one verified incident (synthetic lane)",
                "origin": "collection start (monotonic offsets)",
                "tracks": tracks,
            }
        ],
        "notes": [
            "The verifier recomputed the verdict before projection.",
            "The evaluation interval is not a measured fault duration.",
            "An observed chunk gap alone is not a fault or a verdict; "
            "demand and freshness still govern the verifier.",
            "No PID, endpoint identity, request hash or raw stack is exported.",
        ],
    }
    validate_timeline(model)
    return model
