import tempfile
import unittest
from pathlib import Path

from dfxlab.external_schema import (
    ExternalObservation,
    GpuAggregate,
    HealthObservation,
    MetricsObservation,
    ProcessObservation,
    RuntimeInfo,
    TriggerKind,
)
from dfxlab.recorder import IncidentRecorder

RUNTIME = RuntimeInfo("3.12", "Linux", None, None, 0)


class OneSampleCollector:
    def collect(self, sequence: int) -> ExternalObservation:
        return sample(sequence)


def sample(
    sequence: int = 0,
    *,
    healthy: bool = True,
    alive: bool = True,
    kv_usage: float | None = None,
    preemptions: float | None = None,
) -> ExternalObservation:
    return ExternalObservation(
        sequence=sequence,
        observed_at="2026-01-01T00:00:00+00:00",
        monotonic_ns=sequence + 1,
        health=HealthObservation(healthy, 200 if healthy else None),
        process=ProcessObservation(True, alive),
        metrics=MetricsObservation(
            kv_cache_usage=kv_usage,
            preemptions_total=preemptions,
        ),
        gpu=GpuAggregate(0),
        sampled_sources=("health", "metrics", "process", "gpu"),
    )


class RecorderTest(unittest.TestCase):
    def make_recorder(self, directory: Path) -> IncidentRecorder:
        return IncidentRecorder(
            "http://127.0.0.1:8000",
            directory,
            pid=1,
            history_size=2,
            unhealthy_samples=2,
            preemption_delta=5,
            runtime=RUNTIME,
        )

    def test_bounded_history_accounts_for_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = self.make_recorder(Path(tmp))
            for index in range(3):
                recorder.append(sample(index))
            health = recorder.recorder_health()
            self.assertEqual(len(recorder.history), 2)
            self.assertEqual(health.events_appended_total, 3)
            self.assertEqual(health.events_overwritten_total, 1)
            self.assertEqual(health.retained_sequence_start, 1)
            self.assertEqual(health.retained_sequence_end, 2)

    def test_capture_writes_closed_external_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = self.make_recorder(Path(tmp))
            current = sample(1, alive=False)
            recorder.append(current)
            outcome = recorder.capture(TriggerKind.PROCESS_EXIT, current)
            self.assertTrue(outcome.ok)
            self.assertIsNotNone(outcome.path)
            self.assertTrue(outcome.path.exists())

    def test_process_exit_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = self.make_recorder(Path(tmp))
            self.assertEqual(
                recorder.classify(sample(alive=False)), TriggerKind.PROCESS_EXIT
            )

    def test_health_must_have_been_healthy_before_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = self.make_recorder(Path(tmp))
            self.assertIsNone(recorder.classify(sample(healthy=False)))
            self.assertIsNone(recorder.classify(sample(healthy=True)))
            self.assertIsNone(recorder.classify(sample(healthy=False)))
            self.assertEqual(
                recorder.classify(sample(healthy=False)), TriggerKind.HEALTH_LOST
            )

    def test_cached_unhealthy_value_is_not_counted_twice(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = self.make_recorder(Path(tmp))
            self.assertIsNone(recorder.classify(sample(healthy=True)))
            fresh_failure = sample(healthy=False)
            self.assertIsNone(recorder.classify(fresh_failure))
            cached_failure = ExternalObservation(
                sequence=2,
                observed_at=fresh_failure.observed_at,
                monotonic_ns=2,
                health=fresh_failure.health,
                process=fresh_failure.process,
                metrics=fresh_failure.metrics,
                gpu=fresh_failure.gpu,
                sampled_sources=("metrics",),
            )
            self.assertIsNone(recorder.classify(cached_failure))
            self.assertEqual(
                recorder.classify(sample(healthy=False)), TriggerKind.HEALTH_LOST
            )

    def test_pressure_and_preemption_classification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = self.make_recorder(Path(tmp))
            self.assertEqual(
                recorder.classify(sample(kv_usage=0.98)), TriggerKind.KV_PRESSURE
            )
            self.assertIsNone(recorder.classify(sample(preemptions=1.0)))
            self.assertEqual(
                recorder.classify(sample(preemptions=7.0)),
                TriggerKind.PREEMPTION_STORM,
            )

    def test_unavailable_output_directory_does_not_block_collection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            blocking_file = Path(tmp) / "not-a-directory"
            blocking_file.write_text("occupied", encoding="utf-8")
            recorder = IncidentRecorder(
                "http://127.0.0.1:8000",
                blocking_file / "artifacts",
                sample_interval=0.001,
                private_raw_timeline=True,
                collector=OneSampleCollector(),
                runtime=RUNTIME,
            )
            self.assertEqual(recorder.run(duration=0.01), 0)
            self.assertGreater(recorder.recorder_health().events_appended_total, 0)


if __name__ == "__main__":
    unittest.main()
