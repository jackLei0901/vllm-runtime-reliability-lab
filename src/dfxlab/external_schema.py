from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from enum import Enum
from typing import Any

SCHEMA_VERSION = "external-runtime-observation-v1"
COMPONENT_TYPE = "external_observer"
COLLECTOR_ERROR_KINDS = {
    "health_timeout",
    "health_connection",
    "health_os_error",
    "metrics_timeout",
    "metrics_connection",
    "metrics_os_error",
    "metrics_http_status",
    "metrics_malformed_metrics",
    "gpu_timeout",
    "gpu_error",
    "gpu_malformed",
}


class TriggerKind(str, Enum):
    PROCESS_EXIT = "process_exit"
    HEALTH_LOST = "health_lost"
    KV_PRESSURE = "kv_pressure"
    PREEMPTION_STORM = "preemption_storm"


class WriterErrorKind(str, Enum):
    VALIDATION = "validation"
    SIZE_LIMIT = "size_limit"
    IO = "io"


@dataclass(frozen=True, slots=True)
class RuntimeInfo:
    python_version: str
    platform: str
    torch_version: str | None
    vllm_version: str | None
    gpu_count: int
    gpu_models: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "torch_version": self.torch_version,
            "vllm_version": self.vllm_version,
            "gpu_count": self.gpu_count,
            "gpu_models": list(self.gpu_models),
        }


@dataclass(frozen=True, slots=True)
class HealthObservation:
    ok: bool
    status: int | None

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "status": self.status}


@dataclass(frozen=True, slots=True)
class ProcessObservation:
    tracked: bool
    alive: bool | None
    rss_bytes: int | None = None
    vms_bytes: int | None = None
    threads: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracked": self.tracked,
            "alive": self.alive,
            "rss_bytes": self.rss_bytes,
            "vms_bytes": self.vms_bytes,
            "threads": self.threads,
        }


@dataclass(frozen=True, slots=True)
class MetricsObservation:
    kv_cache_usage: float | None = None
    preemptions_total: float | None = None
    running_requests: float | None = None
    waiting_requests: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kv_cache_usage": self.kv_cache_usage,
            "preemptions_total": self.preemptions_total,
            "running_requests": self.running_requests,
            "waiting_requests": self.waiting_requests,
        }


@dataclass(frozen=True, slots=True)
class GpuAggregate:
    device_count: int
    memory_used_mib: int | None = None
    memory_total_mib: int | None = None
    utilization_percent_mean: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_count": self.device_count,
            "memory_used_mib": self.memory_used_mib,
            "memory_total_mib": self.memory_total_mib,
            "utilization_percent_mean": self.utilization_percent_mean,
        }


@dataclass(frozen=True, slots=True)
class ExternalObservation:
    sequence: int
    observed_at: str
    monotonic_ns: int
    health: HealthObservation
    process: ProcessObservation
    metrics: MetricsObservation
    gpu: GpuAggregate
    sampled_sources: tuple[str, ...] = ()
    collector_errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "observed_at": self.observed_at,
            "monotonic_ns": self.monotonic_ns,
            "health": self.health.to_dict(),
            "process": self.process.to_dict(),
            "metrics": self.metrics.to_dict(),
            "gpu": self.gpu.to_dict(),
            "sampled_sources": list(self.sampled_sources),
            "collector_errors": list(self.collector_errors),
        }


@dataclass(frozen=True, slots=True)
class ExternalTriggerContext:
    kind: TriggerKind
    observed_at: str
    observation_sequence: int
    internal_kind: str = "unknown"
    internal_stage: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "observed_at": self.observed_at,
            "observation_sequence": self.observation_sequence,
            "internal_kind": self.internal_kind,
            "internal_stage": self.internal_stage,
        }


@dataclass(frozen=True, slots=True)
class RecorderHealth:
    events_appended_total: int
    events_overwritten_total: int
    events_dropped_total: int
    retained_sequence_start: int | None
    retained_sequence_end: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "events_appended_total": self.events_appended_total,
            "events_overwritten_total": self.events_overwritten_total,
            "events_dropped_total": self.events_dropped_total,
            "retained_sequence_start": self.retained_sequence_start,
            "retained_sequence_end": self.retained_sequence_end,
        }


@dataclass(frozen=True, slots=True)
class WriterHealth:
    artifacts_written_total: int
    artifacts_failed_total: int
    last_error_kind: WriterErrorKind | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifacts_written_total": self.artifacts_written_total,
            "artifacts_failed_total": self.artifacts_failed_total,
            "last_error_kind": (
                self.last_error_kind.value if self.last_error_kind else None
            ),
        }


@dataclass(frozen=True, slots=True)
class ExternalIncidentArtifact:
    created_at: str
    incident_id: str
    trigger: ExternalTriggerContext
    runtime: RuntimeInfo
    history: tuple[ExternalObservation, ...]
    recorder: RecorderHealth
    writer: WriterHealth
    schema_version: str = SCHEMA_VERSION
    component_type: str = COMPONENT_TYPE

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "component_type": self.component_type,
            "created_at": self.created_at,
            "incident_id": self.incident_id,
            "trigger": self.trigger.to_dict(),
            "runtime": self.runtime.to_dict(),
            "history": [sample.to_dict() for sample in self.history],
            "recorder": self.recorder.to_dict(),
            "writer": self.writer.to_dict(),
        }
        validate_external_artifact(payload)
        return payload


def ephemeral_fingerprint(key: bytes, value: str) -> str:
    digest = hmac.new(key, value.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"hmac-sha256:{digest[:24]}"


def _exact_keys(value: Any, expected: set[str], path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    actual = set(value)
    if actual != expected:
        unknown = sorted(actual - expected)
        missing = sorted(expected - actual)
        raise ValueError(f"{path} keys differ; unknown={unknown}, missing={missing}")
    return value


def _nullable_number(value: Any, path: str) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, (int, float))
    ):
        raise ValueError(f"{path} must be a number or null")


def _nullable_int(value: Any, path: str) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        raise ValueError(f"{path} must be an integer or null")


def validate_external_artifact(payload: dict[str, Any]) -> None:
    """Validate the closed v1 contract without adding a runtime dependency."""
    root = _exact_keys(
        payload,
        {
            "schema_version",
            "component_type",
            "created_at",
            "incident_id",
            "trigger",
            "runtime",
            "history",
            "recorder",
            "writer",
        },
        "$",
    )
    if root["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    if root["component_type"] != COMPONENT_TYPE:
        raise ValueError("unsupported component_type")
    if not isinstance(root["created_at"], str) or not isinstance(
        root["incident_id"], str
    ):
        raise ValueError("created_at and incident_id must be strings")

    trigger = _exact_keys(
        root["trigger"],
        {
            "kind",
            "observed_at",
            "observation_sequence",
            "internal_kind",
            "internal_stage",
        },
        "$.trigger",
    )
    if trigger["kind"] not in {item.value for item in TriggerKind}:
        raise ValueError("invalid trigger kind")
    if trigger["internal_kind"] != "unknown" or trigger["internal_stage"] != "unknown":
        raise ValueError("external artifacts cannot claim an internal cause or stage")
    if not isinstance(trigger["observation_sequence"], int):
        raise ValueError("trigger observation_sequence must be an integer")

    runtime = _exact_keys(
        root["runtime"],
        {
            "python_version",
            "platform",
            "torch_version",
            "vllm_version",
            "gpu_count",
            "gpu_models",
        },
        "$.runtime",
    )
    if not isinstance(runtime["gpu_count"], int) or not isinstance(
        runtime["gpu_models"], list
    ):
        raise ValueError("invalid runtime GPU fields")
    if not all(isinstance(item, str) for item in runtime["gpu_models"]):
        raise ValueError("gpu_models must contain strings")

    history = root["history"]
    if not isinstance(history, list):
        raise ValueError("history must be an array")
    for index, item in enumerate(history):
        prefix = f"$.history[{index}]"
        observation = _exact_keys(
            item,
            {
                "sequence",
                "observed_at",
                "monotonic_ns",
                "health",
                "process",
                "metrics",
                "gpu",
                "sampled_sources",
                "collector_errors",
            },
            prefix,
        )
        if not isinstance(observation["sequence"], int) or not isinstance(
            observation["monotonic_ns"], int
        ):
            raise ValueError(f"{prefix} sequence fields must be integers")
        health = _exact_keys(
            observation["health"], {"ok", "status"}, f"{prefix}.health"
        )
        if not isinstance(health["ok"], bool):
            raise ValueError(f"{prefix}.health.ok must be boolean")
        _nullable_int(health["status"], f"{prefix}.health.status")
        process = _exact_keys(
            observation["process"],
            {"tracked", "alive", "rss_bytes", "vms_bytes", "threads"},
            f"{prefix}.process",
        )
        if not isinstance(process["tracked"], bool):
            raise ValueError(f"{prefix}.process.tracked must be boolean")
        if process["alive"] is not None and not isinstance(process["alive"], bool):
            raise ValueError(f"{prefix}.process.alive must be boolean or null")
        for name in ("rss_bytes", "vms_bytes", "threads"):
            _nullable_int(process[name], f"{prefix}.process.{name}")
        metrics = _exact_keys(
            observation["metrics"],
            {
                "kv_cache_usage",
                "preemptions_total",
                "running_requests",
                "waiting_requests",
            },
            f"{prefix}.metrics",
        )
        for name, value in metrics.items():
            _nullable_number(value, f"{prefix}.metrics.{name}")
        gpu = _exact_keys(
            observation["gpu"],
            {
                "device_count",
                "memory_used_mib",
                "memory_total_mib",
                "utilization_percent_mean",
            },
            f"{prefix}.gpu",
        )
        if not isinstance(gpu["device_count"], int):
            raise ValueError(f"{prefix}.gpu.device_count must be an integer")
        _nullable_int(gpu["memory_used_mib"], f"{prefix}.gpu.memory_used_mib")
        _nullable_int(gpu["memory_total_mib"], f"{prefix}.gpu.memory_total_mib")
        _nullable_number(
            gpu["utilization_percent_mean"], f"{prefix}.gpu.utilization_percent_mean"
        )
        sampled_sources = observation["sampled_sources"]
        if not isinstance(sampled_sources, list) or not all(
            source in {"health", "metrics", "process", "gpu"}
            for source in sampled_sources
        ):
            raise ValueError(f"{prefix}.sampled_sources contains unknown values")
        if len(sampled_sources) != len(set(sampled_sources)):
            raise ValueError(f"{prefix}.sampled_sources contains duplicates")
        if not isinstance(observation["collector_errors"], list) or not all(
            isinstance(error, str) for error in observation["collector_errors"]
        ):
            raise ValueError(f"{prefix}.collector_errors must contain strings")
        unknown_errors = set(observation["collector_errors"]) - COLLECTOR_ERROR_KINDS
        if unknown_errors:
            raise ValueError(f"{prefix}.collector_errors contains unknown values")

    recorder = _exact_keys(
        root["recorder"],
        {
            "events_appended_total",
            "events_overwritten_total",
            "events_dropped_total",
            "retained_sequence_start",
            "retained_sequence_end",
        },
        "$.recorder",
    )
    for name in (
        "events_appended_total",
        "events_overwritten_total",
        "events_dropped_total",
    ):
        if not isinstance(recorder[name], int):
            raise ValueError(f"$.recorder.{name} must be an integer")
    _nullable_int(
        recorder["retained_sequence_start"], "$.recorder.retained_sequence_start"
    )
    _nullable_int(recorder["retained_sequence_end"], "$.recorder.retained_sequence_end")

    writer = _exact_keys(
        root["writer"],
        {"artifacts_written_total", "artifacts_failed_total", "last_error_kind"},
        "$.writer",
    )
    for name in ("artifacts_written_total", "artifacts_failed_total"):
        if not isinstance(writer[name], int):
            raise ValueError(f"$.writer.{name} must be an integer")
    allowed_errors = {item.value for item in WriterErrorKind}
    if (
        writer["last_error_kind"] is not None
        and writer["last_error_kind"] not in allowed_errors
    ):
        raise ValueError("invalid writer error kind")
