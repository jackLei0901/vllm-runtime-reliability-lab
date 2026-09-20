from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from dfxlab.progress import derive_verdict, evaluate_demand, evaluate_producer

BUNDLE_SCHEMA = "incident-evidence-bundle-v1"
OBSERVATION_SCHEMA = "incident-observations-v1"


class BundleError(ValueError):
    """A native evidence bundle is incomplete or internally inconsistent."""


def _evaluation_bounds(observations: dict[str, Any]) -> tuple[int, int]:
    interval = observations["intervals"]["evaluation"]
    start = interval["start_ns"]
    end = interval["end_ns"]
    if not isinstance(start, int) or not isinstance(end, int) or end <= start:
        raise BundleError("invalid evaluation interval")
    return start, end


def _process_facts(observations: dict[str, Any]) -> tuple[bool, bool, bool]:
    target = observations["targets"]["process"]
    supplied = target["supplied"]
    if not supplied:
        if observations["process"]["samples"]:
            raise BundleError("unsupplied process has samples")
        return False, False, False
    start, end = _evaluation_bounds(observations)
    samples = [
        sample
        for sample in observations["process"]["samples"]
        if sample["fresh"] and start <= sample["monotonic_ns"] <= end
    ]
    if (
        len(samples) < 2
        or samples[0]["monotonic_ns"] > start
        or samples[-1]["monotonic_ns"] < end
    ):
        raise BundleError("process samples do not span evaluation interval")
    expected_identity = target["start_identity"]["value"]
    alive = all(sample["alive"] for sample in samples)
    stable = alive and all(
        sample["start_identity_value"] == expected_identity for sample in samples
    )
    return True, stable, alive


def _health_state(observations: dict[str, Any]) -> str:
    if not observations["targets"]["endpoint"]["supplied"]:
        if observations["health"]["samples"]:
            raise BundleError("unsupplied endpoint has health samples")
        return "unavailable"
    start, end = _evaluation_bounds(observations)
    fresh = [sample for sample in observations["health"]["samples"] if sample["fresh"]]
    evaluated = [sample for sample in fresh if start <= sample["monotonic_ns"] <= end]
    healthy_before = any(
        sample["ok"] for sample in fresh if sample["monotonic_ns"] <= end
    )
    threshold = observations["health"]["failure_threshold"]
    trailing_failures = 0
    for sample in reversed(evaluated):
        if sample["ok"]:
            break
        trailing_failures += 1
    if healthy_before and trailing_failures >= threshold:
        return "lost"
    if (
        len(evaluated) >= 2
        and evaluated[0]["monotonic_ns"] <= start
        and evaluated[-1]["monotonic_ns"] >= end
        and all(sample["ok"] and sample["error_kind"] is None for sample in evaluated)
    ):
        return "ok"
    return "insufficient_evidence"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if os.name != "nt":
        path.chmod(0o600)


def _evaluation_input(observations: dict[str, Any]) -> dict[str, Any]:
    process_supplied, identity_stable, process_alive = _process_facts(observations)
    endpoint_supplied = observations["targets"]["endpoint"]["supplied"]
    producers = observations["producer_inputs"]
    decision_source = producers["decision_source"]
    corroborating_source = producers["corroborating_source"]
    decision = evaluate_producer(decision_source, producers[decision_source])
    if corroborating_source is None:
        corroborating = {"state": "producer_missing", "reason": "not_configured"}
    else:
        corroborating = evaluate_producer(
            corroborating_source, producers[corroborating_source]
        )
    demand = evaluate_demand(
        decision_source, observations["demand_inputs"][decision_source]
    )
    verdict_input = {
        "process_supplied": process_supplied,
        "process_identity_stable": identity_stable,
        "process_alive_throughout": process_alive,
        "endpoint_supplied": endpoint_supplied,
        "health_state": _health_state(observations),
        "decision_source": decision_source,
        "decision_state": decision["state"],
        "corroborating_state": corroborating["state"],
        "demand_state": demand["state"],
        "stack_state": observations["stack"]["state"],
    }
    verdict = derive_verdict(verdict_input)
    verdict["decision_source"] = decision_source
    return {
        "decision": decision,
        "corroborating": corroborating,
        "demand": demand,
        "verdict": verdict,
    }


def expected_summary(
    observations: dict[str, Any], observations_sha256: str
) -> dict[str, Any]:
    derived = _evaluation_input(observations)
    return {
        "schema_version": BUNDLE_SCHEMA,
        "targets": observations["targets"],
        "intervals": observations["intervals"],
        "producers": {
            "decision_source": observations["producer_inputs"]["decision_source"],
            "corroborating_source": observations["producer_inputs"][
                "corroborating_source"
            ],
            "decision": derived["decision"],
            "corroborating": derived["corroborating"],
            "client_request_sha256": observations["producer_inputs"][
                "client_request_sha256"
            ],
        },
        "demand": derived["demand"],
        "stack": observations["stack"],
        "verdict": derived["verdict"],
        "sources": {
            "observations.json": {"sha256": observations_sha256},
        },
    }


def write_bundle(output_dir: Path, observations: dict[str, Any]) -> dict[str, Any]:
    """Write observations first, then a summary derived only from those bytes."""

    if observations.get("schema_version") != OBSERVATION_SCHEMA:
        raise BundleError("unknown observation schema")
    output_dir.mkdir(parents=True, exist_ok=True)
    unexpected = {
        path.name
        for path in output_dir.iterdir()
        if path.name not in {"summary.json", "observations.json", "private"}
    }
    if unexpected:
        raise BundleError("output directory contains unexpected public entries")
    observations_path = output_dir / "observations.json"
    _atomic_json(observations_path, observations)
    summary = expected_summary(observations, sha256_file(observations_path))
    _atomic_json(output_dir / "summary.json", summary)
    return summary
