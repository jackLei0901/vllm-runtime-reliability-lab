from __future__ import annotations

import json
import secrets
import time
from collections import deque
from pathlib import Path
from typing import Protocol

from dfxlab.collectors import CadencedCollector, runtime_allowlist
from dfxlab.external_schema import (
    ExternalIncidentArtifact,
    ExternalObservation,
    ExternalTriggerContext,
    RecorderHealth,
    RuntimeInfo,
    TriggerKind,
    ephemeral_fingerprint,
)
from dfxlab.external_writer import IncidentWriter, WriteOutcome
from dfxlab.schema import atomic_write_private_json, utc_now


class ObservationCollector(Protocol):
    def collect(self, sequence: int) -> ExternalObservation: ...


class IncidentRecorder:
    def __init__(
        self,
        base_url: str,
        output_dir: Path,
        pid: int | None = None,
        sample_interval: float = 1.0,
        history_size: int = 300,
        timeout: float = 1.0,
        health_interval: float = 1.0,
        metrics_interval: float = 1.0,
        process_interval: float = 1.0,
        gpu_interval: float = 5.0,
        kv_threshold: float = 0.95,
        unhealthy_samples: int = 3,
        preemption_delta: float = 20.0,
        incident_cooldown: float = 60.0,
        private_raw_timeline: bool = False,
        collector: ObservationCollector | None = None,
        writer: IncidentWriter | None = None,
        runtime: RuntimeInfo | None = None,
    ) -> None:
        if sample_interval <= 0 or history_size <= 0:
            raise ValueError("sample_interval and history_size must be positive")
        self.output_dir = output_dir
        self.sample_interval = sample_interval
        self.kv_threshold = kv_threshold
        self.unhealthy_samples = unhealthy_samples
        self.preemption_delta = preemption_delta
        self.incident_cooldown = incident_cooldown
        self.private_raw_timeline = private_raw_timeline
        self.history: deque[ExternalObservation] = deque(maxlen=history_size)
        self.collector = collector or CadencedCollector(
            base_url=base_url,
            pid=pid,
            timeout=timeout,
            health_interval=health_interval,
            metrics_interval=metrics_interval,
            process_interval=process_interval,
            gpu_interval=gpu_interval,
        )
        self.writer = writer or IncidentWriter(output_dir)
        self.runtime = runtime or runtime_allowlist()
        self._identity_key = secrets.token_bytes(32)
        self._healthy_once = False
        self._consecutive_unhealthy = 0
        self._last_preemptions: float | None = None
        self._last_capture_by_reason: dict[TriggerKind, float] = {}
        self._events_appended_total = 0
        self._events_overwritten_total = 0
        self._events_dropped_total = 0
        self._incident_serial = 0
        self._warned_writer_errors: set[str] = set()

    def classify(self, sample: ExternalObservation) -> TriggerKind | None:
        if sample.process.tracked and sample.process.alive is False:
            return TriggerKind.PROCESS_EXIT

        if "health" in sample.sampled_sources:
            if sample.health.ok:
                self._healthy_once = True
                self._consecutive_unhealthy = 0
            elif self._healthy_once:
                self._consecutive_unhealthy += 1
                if self._consecutive_unhealthy >= self.unhealthy_samples:
                    return TriggerKind.HEALTH_LOST

        usage = sample.metrics.kv_cache_usage
        if usage is not None and usage >= self.kv_threshold:
            return TriggerKind.KV_PRESSURE

        current = sample.metrics.preemptions_total
        if current is not None:
            previous = self._last_preemptions
            self._last_preemptions = current
            if previous is not None and current - previous >= self.preemption_delta:
                return TriggerKind.PREEMPTION_STORM
        return None

    def append(self, sample: ExternalObservation) -> None:
        if len(self.history) == self.history.maxlen:
            self._events_overwritten_total += 1
        self.history.append(sample)
        self._events_appended_total += 1

    def recorder_health(self) -> RecorderHealth:
        first = self.history[0].sequence if self.history else None
        last = self.history[-1].sequence if self.history else None
        return RecorderHealth(
            events_appended_total=self._events_appended_total,
            events_overwritten_total=self._events_overwritten_total,
            events_dropped_total=self._events_dropped_total,
            retained_sequence_start=first,
            retained_sequence_end=last,
        )

    def capture(self, reason: TriggerKind, sample: ExternalObservation) -> WriteOutcome:
        self._incident_serial += 1
        incident_id = ephemeral_fingerprint(
            self._identity_key,
            f"{self._incident_serial}:{sample.sequence}:{sample.monotonic_ns}",
        )
        artifact = ExternalIncidentArtifact(
            created_at=utc_now(),
            incident_id=incident_id,
            trigger=ExternalTriggerContext(
                kind=reason,
                observed_at=sample.observed_at,
                observation_sequence=sample.sequence,
            ),
            runtime=self.runtime,
            history=tuple(self.history),
            recorder=self.recorder_health(),
            writer=self.writer.health(),
        )
        outcome = self.writer.try_write(artifact)
        if not outcome.ok and outcome.error_kind is not None:
            error = outcome.error_kind.value
            if error not in self._warned_writer_errors:
                print(f"incident writer failed open: {error}")
                self._warned_writer_errors.add(error)
        return outcome

    def run(self, duration: float | None = None, stop_on_incident: bool = False) -> int:
        start = time.monotonic()
        sequence = 0
        captured: list[str] = []
        timeline = None
        try:
            if self.private_raw_timeline:
                try:
                    timeline = (self.output_dir / "timeline.private.jsonl").open(
                        "a", encoding="utf-8"
                    )
                except OSError:
                    print("private timeline unavailable; continuing without it")
            while duration is None or time.monotonic() - start < duration:
                try:
                    sample = self.collector.collect(sequence)
                except Exception as exc:
                    self._events_dropped_total += 1
                    print(f"collector sample dropped: {type(exc).__name__}")
                    sequence += 1
                    time.sleep(self.sample_interval)
                    continue
                self.append(sample)
                if timeline is not None:
                    try:
                        timeline.write(json.dumps(sample.to_dict()) + "\n")
                        timeline.flush()
                    except OSError:
                        timeline.close()
                        timeline = None
                        print("private timeline disabled after write failure")
                reason = self.classify(sample)
                if reason is not None:
                    now = time.monotonic()
                    last_capture = self._last_capture_by_reason.get(reason)
                    if (
                        last_capture is None
                        or now - last_capture >= self.incident_cooldown
                    ):
                        outcome = self.capture(reason, sample)
                        self._last_capture_by_reason[reason] = now
                        if outcome.ok and outcome.path is not None:
                            captured.append(str(outcome.path))
                            print(f"captured incident: {outcome.path}")
                        if stop_on_incident:
                            break
                sequence += 1
                time.sleep(self.sample_interval)
        except KeyboardInterrupt:
            print("recording stopped by user")
        finally:
            if timeline is not None:
                timeline.close()

        try:
            atomic_write_private_json(
                self.output_dir / "run-summary.private.json",
                {
                    "schema_version": 1,
                    "finished_at": utc_now(),
                    "recorder": self.recorder_health().to_dict(),
                    "writer": self.writer.health().to_dict(),
                    "incident_files": captured,
                    "private_raw_timeline": self.private_raw_timeline,
                },
            )
        except OSError:
            print("private run summary write failed")
        return 0
