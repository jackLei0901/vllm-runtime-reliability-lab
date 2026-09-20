from __future__ import annotations

import math
from typing import Any

PRODUCER_STATES = {
    "progressing",
    "flat",
    "producer_missing",
    "insufficient_evidence",
}
DEMAND_STATES = {"present", "absent", "insufficient_evidence"}
HEALTH_STATES = {"ok", "lost", "unavailable", "insufficient_evidence"}
STACK_STATES = {"disabled", "produced", "unavailable"}
DECISION_SCOPES = {
    "server_counter": "service",
    "client_request": "request",
}

SERVER_PRODUCER_KEYS = {
    "producer_available",
    "evaluation_start_ns",
    "evaluation_end_ns",
    "minimum_span_ns",
    "minimum_fresh_samples",
    "samples",
}
CLIENT_PRODUCER_KEYS = {
    "producer_available",
    "evaluation_start_ns",
    "evaluation_end_ns",
    "request_started_ns",
    "request_completed_ns",
    "chunks",
}
SERVER_DEMAND_KEYS = {
    "evaluation_start_ns",
    "evaluation_end_ns",
    "minimum_fresh_samples",
    "samples",
}
CLIENT_DEMAND_KEYS = {
    "evaluation_start_ns",
    "evaluation_end_ns",
    "request_started_ns",
    "request_completed_ns",
}
VERDICT_KEYS = {
    "process_supplied",
    "process_identity_stable",
    "process_alive_throughout",
    "endpoint_supplied",
    "health_state",
    "decision_source",
    "decision_state",
    "corroborating_state",
    "demand_state",
    "stack_state",
}


class ProgressContractError(ValueError):
    """An input cannot be evaluated under the closed v0.2 contract."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ProgressContractError(code)


def _closed(value: Any, expected: set[str], path: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{path}_not_object")
    _require(set(value) == expected, f"{path}_shape")
    return value


def _integer(value: Any, path: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{path}_type")
    return value


def _number(value: Any, path: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value),
        f"{path}_type",
    )
    return float(value)


def _boolean(value: Any, path: str) -> bool:
    _require(isinstance(value, bool), f"{path}_type")
    return value


def _interval(value: dict[str, Any]) -> tuple[int, int]:
    start = _integer(value["evaluation_start_ns"], "evaluation_start_ns")
    end = _integer(value["evaluation_end_ns"], "evaluation_end_ns")
    _require(start >= 0 and end > start, "evaluation_interval")
    return start, end


def _ordered_times(items: list[dict[str, Any]], path: str) -> list[int]:
    times = [_integer(item["monotonic_ns"], f"{path}_monotonic_ns") for item in items]
    _require(
        all(
            current > previous
            for previous, current in zip(times, times[1:], strict=False)
        ),
        f"{path}_order",
    )
    return times


def evaluate_producer(kind: str, value: dict[str, Any]) -> dict[str, str]:
    """Evaluate one producer directly from a frozen matrix input shape."""

    _require(kind in DECISION_SCOPES, "producer_kind")
    if kind == "server_counter":
        return _evaluate_server_counter(value)
    return _evaluate_client_request(value)


def _evaluate_server_counter(value: dict[str, Any]) -> dict[str, str]:
    value = _closed(value, SERVER_PRODUCER_KEYS, "server_producer")
    start, end = _interval(value)
    available = _boolean(value["producer_available"], "producer_available")
    minimum_span = _integer(value["minimum_span_ns"], "minimum_span_ns")
    minimum_samples = _integer(value["minimum_fresh_samples"], "minimum_fresh_samples")
    _require(minimum_span > 0, "minimum_span_ns_value")
    _require(minimum_samples >= 2, "minimum_fresh_samples_value")
    samples = value["samples"]
    _require(isinstance(samples, list), "server_samples_type")

    normalized: list[tuple[int, bool, float]] = []
    for sample in samples:
        sample = _closed(sample, {"monotonic_ns", "fresh", "value"}, "server_sample")
        monotonic_ns = _integer(sample["monotonic_ns"], "sample_monotonic_ns")
        _require(start <= monotonic_ns <= end, "server_sample_outside_interval")
        fresh = _boolean(sample["fresh"], "sample_fresh")
        counter = _number(sample["value"], "sample_value")
        _require(counter >= 0, "sample_value_negative")
        normalized.append((monotonic_ns, fresh, counter))
    _ordered_times([{"monotonic_ns": item[0]} for item in normalized], "server_samples")

    if not available:
        _require(not samples, "missing_producer_has_samples")
        return {"state": "producer_missing", "reason": "unavailable"}

    fresh_samples = [sample for sample in normalized if sample[1]]
    if len(fresh_samples) < minimum_samples:
        return {"state": "insufficient_evidence", "reason": "sample_count"}
    if fresh_samples[-1][0] - fresh_samples[0][0] < minimum_span:
        return {"state": "insufficient_evidence", "reason": "sample_span"}

    values = [sample[2] for sample in fresh_samples]
    adjacent_values = zip(values, values[1:], strict=False)
    if any(current < previous for previous, current in adjacent_values):
        return {"state": "insufficient_evidence", "reason": "counter_reset"}
    adjacent_values = zip(values, values[1:], strict=False)
    if any(current > previous for previous, current in adjacent_values):
        return {"state": "progressing", "reason": "positive_delta"}
    return {"state": "flat", "reason": "equal_samples"}


def _evaluate_client_request(value: dict[str, Any]) -> dict[str, str]:
    value = _closed(value, CLIENT_PRODUCER_KEYS, "client_producer")
    start, end = _interval(value)
    available = _boolean(value["producer_available"], "producer_available")
    request_started = _integer(value["request_started_ns"], "request_started_ns")
    completed = value["request_completed_ns"]
    if completed is not None:
        completed = _integer(completed, "request_completed_ns")
        _require(completed >= request_started, "request_completion_order")

    chunks = value["chunks"]
    _require(isinstance(chunks, list), "client_chunks_type")
    normalized: list[tuple[int, str]] = []
    for chunk in chunks:
        chunk = _closed(chunk, {"monotonic_ns", "kind"}, "client_chunk")
        monotonic_ns = _integer(chunk["monotonic_ns"], "chunk_monotonic_ns")
        chunk_kind = chunk["kind"]
        _require(
            chunk_kind in {"content", "role_only", "usage_only", "empty"},
            "chunk_kind",
        )
        normalized.append((monotonic_ns, chunk_kind))
    _ordered_times([{"monotonic_ns": item[0]} for item in normalized], "client_chunks")

    if not available:
        _require(not chunks, "missing_producer_has_chunks")
        return {"state": "producer_missing", "reason": "unavailable"}

    in_interval = [item for item in normalized if start <= item[0] <= end]
    if any(chunk_kind == "content" for _, chunk_kind in in_interval):
        return {"state": "progressing", "reason": "content_chunk"}
    if request_started > start or (completed is not None and completed <= end):
        return {
            "state": "insufficient_evidence",
            "reason": "request_not_open_for_interval",
        }
    return {"state": "flat", "reason": "no_content_while_open"}


def evaluate_demand(kind: str, value: dict[str, Any]) -> dict[str, str]:
    """Evaluate admitted-work evidence from the selected producer's raw shape."""

    _require(kind in DECISION_SCOPES, "demand_kind")
    if kind == "server_counter":
        return _evaluate_server_demand(value)
    return _evaluate_client_demand(value)


def _evaluate_server_demand(value: dict[str, Any]) -> dict[str, str]:
    value = _closed(value, SERVER_DEMAND_KEYS, "server_demand")
    start, end = _interval(value)
    minimum_samples = _integer(value["minimum_fresh_samples"], "minimum_fresh_samples")
    _require(minimum_samples >= 2, "minimum_fresh_samples_value")
    samples = value["samples"]
    _require(isinstance(samples, list), "demand_samples_type")

    normalized: list[tuple[int, bool, float, float]] = []
    for sample in samples:
        sample = _closed(
            sample,
            {"monotonic_ns", "fresh", "running", "waiting"},
            "demand_sample",
        )
        monotonic_ns = _integer(sample["monotonic_ns"], "sample_monotonic_ns")
        _require(start <= monotonic_ns <= end, "demand_sample_outside_interval")
        fresh = _boolean(sample["fresh"], "sample_fresh")
        running = _number(sample["running"], "sample_running")
        waiting = _number(sample["waiting"], "sample_waiting")
        _require(running >= 0 and waiting >= 0, "demand_value_negative")
        normalized.append((monotonic_ns, fresh, running, waiting))
    _ordered_times([{"monotonic_ns": item[0]} for item in normalized], "demand_samples")

    fresh_samples = [sample for sample in normalized if sample[1]]
    if len(fresh_samples) < minimum_samples:
        return {"state": "insufficient_evidence", "reason": "sample_count"}
    if fresh_samples[0][0] > start or fresh_samples[-1][0] < end:
        return {"state": "insufficient_evidence", "reason": "sample_span"}

    work_visible = [running + waiting > 0 for _, _, running, waiting in fresh_samples]
    if all(work_visible):
        return {"state": "present", "reason": "work_visible"}
    if not any(work_visible):
        return {"state": "absent", "reason": "no_work_visible"}
    return {
        "state": "insufficient_evidence",
        "reason": "demand_not_continuous",
    }


def _evaluate_client_demand(value: dict[str, Any]) -> dict[str, str]:
    value = _closed(value, CLIENT_DEMAND_KEYS, "client_demand")
    start, end = _interval(value)
    request_started = _integer(value["request_started_ns"], "request_started_ns")
    completed = value["request_completed_ns"]
    if completed is not None:
        completed = _integer(completed, "request_completed_ns")
        _require(completed >= request_started, "request_completion_order")
    if request_started <= start and (completed is None or completed > end):
        return {"state": "present", "reason": "request_open_for_interval"}
    return {
        "state": "insufficient_evidence",
        "reason": "request_not_open_for_interval",
    }


def derive_verdict(value: dict[str, Any]) -> dict[str, str | bool]:
    """Apply the frozen precedence directly to a verdict fixture input."""

    value = _closed(value, VERDICT_KEYS, "verdict")
    process_supplied = _boolean(value["process_supplied"], "process_supplied")
    identity_stable = _boolean(
        value["process_identity_stable"], "process_identity_stable"
    )
    process_alive = _boolean(
        value["process_alive_throughout"], "process_alive_throughout"
    )
    endpoint_supplied = _boolean(value["endpoint_supplied"], "endpoint_supplied")
    health_state = value["health_state"]
    decision_source = value["decision_source"]
    decision_state = value["decision_state"]
    corroborating_state = value["corroborating_state"]
    demand_state = value["demand_state"]
    stack_state = value["stack_state"]

    _require(health_state in HEALTH_STATES, "health_state")
    _require(decision_source in DECISION_SCOPES, "decision_source")
    _require(decision_state in PRODUCER_STATES, "decision_state")
    _require(corroborating_state in PRODUCER_STATES, "corroborating_state")
    _require(demand_state in DEMAND_STATES, "demand_state")
    _require(stack_state in STACK_STATES, "stack_state")
    if not process_supplied:
        _require(not identity_stable and not process_alive, "unsupplied_process_state")
    if not endpoint_supplied:
        _require(health_state == "unavailable", "unsupplied_endpoint_health")

    comparable = {"progressing", "flat"}
    conflict = (
        decision_state in comparable
        and corroborating_state in comparable
        and decision_state != corroborating_state
    )

    if process_supplied and (not identity_stable or not process_alive):
        verdict = "process_missing"
    elif endpoint_supplied and health_state == "lost":
        verdict = "health_lost"
    elif decision_state == "progressing":
        verdict = "progress_observed"
    elif (
        process_supplied
        and identity_stable
        and process_alive
        and endpoint_supplied
        and health_state == "ok"
        and decision_state == "flat"
        and demand_state == "present"
    ):
        verdict = "alive_health_ok_no_progress"
    else:
        verdict = "undetermined"

    return {
        "verdict": verdict,
        "progress_scope": DECISION_SCOPES[decision_source],
        "producer_conflict": conflict,
    }
