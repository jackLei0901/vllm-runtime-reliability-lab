import json
import unittest
from pathlib import Path

try:
    from jsonschema import Draft202012Validator, FormatChecker
except ImportError:  # The project declares jsonschema in its dev extra.
    Draft202012Validator = None
    FormatChecker = None

from dfxlab.external_schema import (
    ExternalIncidentArtifact,
    ExternalObservation,
    ExternalTriggerContext,
    GpuAggregate,
    HealthObservation,
    MetricsObservation,
    ProcessObservation,
    RecorderHealth,
    RuntimeInfo,
    TriggerKind,
    WriterHealth,
    ephemeral_fingerprint,
    validate_external_artifact,
)


def artifact() -> ExternalIncidentArtifact:
    observation = ExternalObservation(
        sequence=7,
        observed_at="2026-01-01T00:00:00+00:00",
        monotonic_ns=123,
        health=HealthObservation(False, 503),
        process=ProcessObservation(True, True, 1024, 2048, 4),
        metrics=MetricsObservation(0.96, 2.0, 3.0, 1.0),
        gpu=GpuAggregate(1, 20000, 24000, 91.0),
        sampled_sources=("health", "metrics", "process", "gpu"),
        collector_errors=(),
    )
    return ExternalIncidentArtifact(
        created_at="2026-01-01T00:00:00+00:00",
        incident_id=ephemeral_fingerprint(b"key-a", "incident-1"),
        trigger=ExternalTriggerContext(
            TriggerKind.HEALTH_LOST, observation.observed_at, observation.sequence
        ),
        runtime=RuntimeInfo("3.12.3", "Linux", "2.13", "0.27", 1, ("GPU",)),
        history=(observation,),
        recorder=RecorderHealth(8, 1, 0, 1, 7),
        writer=WriterHealth(0, 0, None),
    )


@unittest.skipIf(Draft202012Validator is None, "jsonschema dev extra is unavailable")
class ExternalSchemaTest(unittest.TestCase):
    def test_valid_artifact(self) -> None:
        payload = artifact().to_dict()
        validate_external_artifact(payload)

    def test_unknown_top_level_field_is_rejected(self) -> None:
        payload = artifact().to_dict()
        payload["prompt"] = "secret"
        with self.assertRaisesRegex(ValueError, "unknown"):
            validate_external_artifact(payload)

    def test_unknown_nested_field_is_rejected(self) -> None:
        payload = artifact().to_dict()
        payload["history"][0]["metrics"]["request_id"] = 1
        with self.assertRaisesRegex(ValueError, "unknown"):
            validate_external_artifact(payload)

    def test_external_trigger_cannot_claim_internal_cause(self) -> None:
        payload = artifact().to_dict()
        payload["trigger"]["internal_kind"] = "cuda_oom"
        with self.assertRaisesRegex(ValueError, "cannot claim"):
            validate_external_artifact(payload)

    def test_ephemeral_keys_do_not_correlate_across_processes(self) -> None:
        self.assertNotEqual(
            ephemeral_fingerprint(b"process-a", "same"),
            ephemeral_fingerprint(b"process-b", "same"),
        )

    def test_privacy_canaries_are_absent(self) -> None:
        encoded = json.dumps(artifact().to_dict())
        for canary in ("prompt_token_ids", "messages", "authorization", "api_key"):
            self.assertNotIn(canary, encoded)

    def test_published_json_schema_is_closed(self) -> None:
        schema_path = (
            Path(__file__).resolve().parents[1]
            / "schema"
            / "external-runtime-observation-v1.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertFalse(schema["additionalProperties"])
        self.assertFalse(schema["$defs"]["observation"]["additionalProperties"])

    def test_example_conforms_to_published_json_schema(self) -> None:
        root = Path(__file__).resolve().parents[1]
        schema = json.loads(
            (root / "schema" / "external-runtime-observation-v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
        example = json.loads(
            (root / "examples" / "external-runtime-observation-v1.json").read_text(
                encoding="utf-8"
            )
        )
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(example)


if __name__ == "__main__":
    unittest.main()
