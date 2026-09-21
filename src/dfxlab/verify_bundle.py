from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from dfxlab.bundle import (
    BUNDLE_SCHEMA,
    OBSERVATION_SCHEMA,
    BundleError,
    expected_summary,
    sha256_file,
)
from dfxlab.progress import (
    ProgressContractError,
    evaluate_demand,
    evaluate_producer,
)
from dfxlab.replay import ReplayError, replay
from dfxlab.stacks import STACK_ERROR_KINDS

SUMMARY_KEYS = {
    "schema_version",
    "targets",
    "intervals",
    "producers",
    "demand",
    "stack",
    "verdict",
    "sources",
}
OBSERVATION_KEYS = {
    "schema_version",
    "targets",
    "intervals",
    "process",
    "health",
    "producer_inputs",
    "demand_inputs",
    "stack",
}
FORBIDDEN_KEYS = {
    "base_url",
    "hostname",
    "prompt",
    "completion",
    "token_ids",
    "headers",
    "authorization",
    "credential",
    "command",
    "resolved_path",
    "raw_stack",
    "environment",
}


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BundleError("duplicate JSON key")
        result[key] = value
    return result


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BundleError(f"cannot read {path.name}") from exc
    if not isinstance(value, dict):
        raise BundleError(f"{path.name} root is not an object")
    return value


def _closed(value: Any, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise BundleError(f"{name} shape changed")
    return value


def _privacy_scan(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key.lower() in FORBIDDEN_KEYS:
                raise BundleError(f"privacy boundary crossed by {key}")
            _privacy_scan(nested)
    elif isinstance(value, list):
        for nested in value:
            _privacy_scan(nested)


def _strict_times(samples: list[dict[str, Any]], name: str) -> None:
    times = [sample["monotonic_ns"] for sample in samples]
    if not all(
        isinstance(value, int) and not isinstance(value, bool) for value in times
    ):
        raise BundleError(f"{name} time type")
    if any(
        current <= previous for previous, current in zip(times, times[1:], strict=False)
    ):
        raise BundleError(f"{name} order")


def _validate_observations(value: dict[str, Any]) -> None:
    targets = _closed(value["targets"], {"process", "endpoint", "relation"}, "targets")
    process_target = _closed(
        targets["process"],
        {"supplied", "pid", "start_identity", "identity_source"},
        "process target",
    )
    endpoint_target = _closed(
        targets["endpoint"],
        {"supplied", "endpoint_id", "identity_source"},
        "endpoint target",
    )
    if targets["relation"] != "operator_asserted_same_incident":
        raise BundleError("unknown target relation")
    if not isinstance(process_target["supplied"], bool) or not isinstance(
        endpoint_target["supplied"], bool
    ):
        raise BundleError("target supplied type")
    if process_target["identity_source"] != "operator_supplied":
        raise BundleError("unknown process identity source")
    if endpoint_target["identity_source"] != "operator_supplied":
        raise BundleError("unknown endpoint identity source")
    if process_target["supplied"]:
        if not isinstance(process_target["pid"], int) or process_target["pid"] <= 0:
            raise BundleError("invalid process target")
        identity = _closed(
            process_target["start_identity"], {"kind", "value"}, "start identity"
        )
        if identity["kind"] not in {"linux_start_ticks", "windows_creation_time"}:
            raise BundleError("unknown process identity kind")
        if not isinstance(identity["value"], str) or not identity["value"]:
            raise BundleError("invalid process identity value")
    elif (
        process_target["pid"] is not None
        or process_target["start_identity"] is not None
    ):
        raise BundleError("unsupplied process retains identity")
    if endpoint_target["supplied"]:
        if not isinstance(endpoint_target["endpoint_id"], str) or not re.fullmatch(
            r"sha256:[0-9a-f]{12}", endpoint_target["endpoint_id"]
        ):
            raise BundleError("invalid endpoint id")
    elif endpoint_target["endpoint_id"] is not None:
        raise BundleError("unsupplied endpoint retains identity")

    intervals = _closed(value["intervals"], {"collection", "evaluation"}, "intervals")
    collection = _closed(
        intervals["collection"], {"start_ns", "end_ns"}, "collection interval"
    )
    evaluation = _closed(
        intervals["evaluation"],
        {"start_ns", "end_ns", "minimum_duration_ns"},
        "evaluation interval",
    )
    numbers = [
        collection["start_ns"],
        collection["end_ns"],
        evaluation["start_ns"],
        evaluation["end_ns"],
        evaluation["minimum_duration_ns"],
    ]
    if not all(
        isinstance(item, int) and not isinstance(item, bool) for item in numbers
    ):
        raise BundleError("interval type")
    if min(numbers) < 0 or evaluation["minimum_duration_ns"] <= 0:
        raise BundleError("interval value")
    if not (
        collection["start_ns"]
        <= evaluation["start_ns"]
        < evaluation["end_ns"]
        <= collection["end_ns"]
    ):
        raise BundleError("interval containment")
    if (
        evaluation["end_ns"] - evaluation["start_ns"]
        < evaluation["minimum_duration_ns"]
    ):
        raise BundleError("evaluation duration")

    process = _closed(value["process"], {"samples"}, "process")
    if not isinstance(process["samples"], list):
        raise BundleError("process samples type")
    for sample in process["samples"]:
        _closed(
            sample,
            {"monotonic_ns", "fresh", "alive", "start_identity_value"},
            "process sample",
        )
        if not isinstance(sample["fresh"], bool) or not isinstance(
            sample["alive"], bool
        ):
            raise BundleError("process sample type")
        identity = sample["start_identity_value"]
        if identity is not None and not isinstance(identity, str):
            raise BundleError("process sample identity type")
    _strict_times(process["samples"], "process samples")

    health = _closed(value["health"], {"failure_threshold", "samples"}, "health")
    if (
        not isinstance(health["failure_threshold"], int)
        or health["failure_threshold"] <= 0
    ):
        raise BundleError("health threshold")
    if not isinstance(health["samples"], list):
        raise BundleError("health samples type")
    for sample in health["samples"]:
        _closed(
            sample,
            {"monotonic_ns", "fresh", "ok", "status", "error_kind"},
            "health sample",
        )
        if not isinstance(sample["fresh"], bool) or not isinstance(sample["ok"], bool):
            raise BundleError("health sample type")
        if sample["status"] is not None and not isinstance(sample["status"], int):
            raise BundleError("health status type")
        if sample["error_kind"] not in {None, "timeout", "connection", "os_error"}:
            raise BundleError("health error kind")
        if sample["error_kind"] is None:
            if sample["status"] is None or sample["ok"] != (
                200 <= sample["status"] < 300
            ):
                raise BundleError("health status inconsistent")
        elif sample["ok"] or sample["status"] is not None:
            raise BundleError("health transport error inconsistent")
    _strict_times(health["samples"], "health samples")

    producers = _closed(
        value["producer_inputs"],
        {
            "decision_source",
            "corroborating_source",
            "client_request_sha256",
            "server_counter",
            "client_request",
        },
        "producer inputs",
    )
    if producers["decision_source"] not in {"server_counter", "client_request"}:
        raise BundleError("decision source")
    if producers["corroborating_source"] not in {
        None,
        "server_counter",
        "client_request",
    }:
        raise BundleError("corroborating source")
    digest = producers["client_request_sha256"]
    if digest is not None and not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise BundleError("client request digest")
    if (
        producers["decision_source"] == "client_request"
        or producers["corroborating_source"] == "client_request"
    ) and digest is None:
        raise BundleError("configured client request lacks digest")
    demands = _closed(
        value["demand_inputs"],
        {"server_counter", "client_request"},
        "demand inputs",
    )
    for kind in ("server_counter", "client_request"):
        evaluate_producer(kind, producers[kind])
        evaluate_demand(kind, demands[kind])

    stack = _closed(value["stack"], {"state", "producer"}, "stack")
    if stack["state"] == "disabled":
        if stack["producer"] is not None:
            raise BundleError("disabled stack has producer")
    elif stack["state"] in {"produced", "unavailable"}:
        producer = _closed(
            stack["producer"],
            {
                "sampler_name",
                "sampler_version",
                "binary_sha256",
                "platform",
                "yama_ptrace_scope",
                "exit_status",
                "output_produced",
                "raw_output_sha256",
                "error_kind",
            },
            "stack producer",
        )
        if producer["sampler_name"] != "py-spy":
            raise BundleError("unknown stack sampler")
        for key in ("sampler_version", "binary_sha256", "raw_output_sha256"):
            item = producer[key]
            if item is not None and not isinstance(item, str):
                raise BundleError("stack producer type")
        for key in ("binary_sha256", "raw_output_sha256"):
            item = producer[key]
            if item is not None and not re.fullmatch(r"[0-9a-f]{64}", item):
                raise BundleError("stack digest")
        if producer["platform"] not in {"Linux", "Windows", "Darwin"}:
            raise BundleError("stack platform")
        if producer["yama_ptrace_scope"] is not None and not isinstance(
            producer["yama_ptrace_scope"], int
        ):
            raise BundleError("stack attach context")
        if producer["exit_status"] is not None and not isinstance(
            producer["exit_status"], int
        ):
            raise BundleError("stack exit status")
        if not isinstance(producer["output_produced"], bool):
            raise BundleError("stack output type")
        if producer["error_kind"] not in STACK_ERROR_KINDS | {None}:
            raise BundleError("stack error kind")
        if stack["state"] == "produced" and (
            not producer["output_produced"] or producer["error_kind"] is not None
        ):
            raise BundleError("inconsistent produced stack")
        if stack["state"] == "unavailable" and producer["error_kind"] is None:
            raise BundleError("unavailable stack lacks error")
    else:
        raise BundleError("stack state")


def verify_bundle(bundle_dir: Path) -> dict[str, Any]:
    if not bundle_dir.is_dir():
        raise BundleError("bundle directory missing")
    public_entries = {
        path.name for path in bundle_dir.iterdir() if path.name != "private"
    }
    if public_entries != {"summary.json", "observations.json"}:
        raise BundleError("publishable file set changed")
    if (bundle_dir / "private").exists() and not (bundle_dir / "private").is_dir():
        raise BundleError("private entry is not a directory")

    summary = _closed(_load(bundle_dir / "summary.json"), SUMMARY_KEYS, "summary")
    observations = _closed(
        _load(bundle_dir / "observations.json"),
        OBSERVATION_KEYS,
        "observations",
    )
    if summary["schema_version"] != BUNDLE_SCHEMA:
        raise BundleError("unknown bundle schema")
    if observations["schema_version"] != OBSERVATION_SCHEMA:
        raise BundleError("unknown observation schema")
    try:
        _validate_observations(observations)
    except (KeyError, TypeError, ProgressContractError) as exc:
        raise BundleError("observation structure invalid") from exc
    _privacy_scan(summary)
    _privacy_scan(observations)

    digest = sha256_file(bundle_dir / "observations.json")
    try:
        expected = expected_summary(observations, digest)
    except (KeyError, TypeError, ProgressContractError) as exc:
        raise BundleError("evidence cannot be recomputed") from exc
    if summary != expected:
        if summary.get("sources") != expected["sources"]:
            raise BundleError("source digest mismatch")
        raise BundleError("stored claims differ from recomputed evidence")
    return summary


def project_legacy_r3(case_dir: Path) -> dict[str, Any]:
    """Project only facts present in the immutable public R3 result."""

    case_dir = case_dir.resolve()
    try:
        replay(case_dir)
    except ReplayError as exc:
        raise BundleError(f"legacy R3 digest mismatch or invalid shape: {exc}") from exc
    manifest = _load(case_dir / "replay-manifest.json")
    expected_keys = {
        "case_id",
        "files",
        "kind",
        "schema_version",
        "source_links",
        "title",
    }
    _closed(manifest, expected_keys, "legacy manifest")
    cells: dict[str, dict[str, Any]] = {}
    for arm in ("base_pause", "fix_pause"):
        entry = manifest["files"][arm]
        _closed(entry, {"path", "sha256"}, f"legacy {arm} source")
        path = case_dir / entry["path"]
        if sha256_file(path) != entry["sha256"]:
            raise BundleError(f"legacy {arm} digest mismatch")
        cell = _load(path)
        cells[arm] = {
            "classification": cell.get("classification"),
            "stalled_before_release": cell.get("stalled_before_release"),
            "health_during_stall": cell.get("health_during_stall"),
            "progress_count_at_release": cell.get("progress_count_at_release"),
            "native_v0_2_verdict": None,
        }
    return {
        "schema_version": "legacy-r3-v0.2-projection-v1",
        "case_id": manifest["case_id"],
        "native_bundle": False,
        "unavailable": [
            "process_start_identity",
            "repeated_health_samples",
            "server_counter_samples",
        ],
        "cells": cells,
    }
