from __future__ import annotations

import hashlib
import json
import re
from typing import Any

CAPTURE_SCHEMA = "native-evidence-experiment-v0"
ATTRIBUTION_SCHEMA = "native-attribution-v0"

STAGE_OUTCOMES = {
    "not_requested": {"disabled"},
    "coordinator": {"capture_occupied", "rate_limited"},
    "preflight": {"unsupported", "binary_missing", "feature_disabled"},
    "execution": {
        "produced",
        "timeout",
        "output_budget_exceeded",
        "permission_denied",
        "empty_output",
        "execution_failed",
    },
}
EXECUTION_DOMAINS = {"python", "native", "mixed", "unknown"}
GIL_STATES = {"holding", "waiting", "dropping", "unknown"}
BLOCKED_IN = {"queue_wait", "communicator_destruction", "unknown"}
COVERAGE = {"thread_state", "gil_state", "lifecycle_stage"}
PAIRING_RESULTS = {
    "interchangeable",
    "not_scorable",
    "producer_disagreement",
    "target_not_stable",
}
LIFECYCLE_STAGES = {
    "dump_responder_active",
    "dump_responder_stopped",
    "communicator_destroy_started",
    "communicator_destroy_completed",
    "peer_dump_request_observed",
    "dump_completed",
}
LIFECYCLE_ORDER = (
    ("dump_responder_active", "dump_responder_stopped"),
    ("dump_responder_stopped", "communicator_destroy_started"),
    ("communicator_destroy_started", "communicator_destroy_completed"),
    ("peer_dump_request_observed", "dump_completed"),
)

_DIGEST = re.compile(r"[0-9a-f]{64}")
_TOKEN = re.compile(r"[a-z0-9][a-z0-9_.:+-]{0,127}")


class NativeEvidenceError(ValueError):
    """Raised when an experimental native-evidence contract fails closed."""


def _closed(value: Any, keys: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise NativeEvidenceError(f"{name} shape changed")
    return value


def _bounded_token(value: Any, name: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not _TOKEN.fullmatch(value):
        raise NativeEvidenceError(f"invalid {name}")
    return value


def _digest(value: Any, name: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise NativeEvidenceError(f"invalid {name}")
    return value


def validate_capture(capture: dict[str, Any]) -> dict[str, Any]:
    """Validate the experimental envelope before any producer normalization."""

    _closed(
        capture,
        {"schema_version", "subject", "window", "producer", "outcome"},
        "capture",
    )
    if capture["schema_version"] != CAPTURE_SCHEMA:
        raise NativeEvidenceError("unknown capture schema")

    subject = _closed(
        capture["subject"],
        {
            "process_identity_kind",
            "process_identity_value",
            "post_capture_identity_value",
            "declared_role",
            "declared_rank",
        },
        "subject",
    )
    if subject["process_identity_kind"] not in {
        "linux_proc_start_ticks",
        "windows_creation_time",
    }:
        raise NativeEvidenceError("unknown process identity kind")
    _bounded_token(subject["process_identity_value"], "process identity")
    _bounded_token(
        subject["post_capture_identity_value"],
        "post-capture process identity",
        nullable=True,
    )
    _bounded_token(subject["declared_role"], "declared role")
    rank = subject["declared_rank"]
    if rank is not None and (
        not isinstance(rank, int) or isinstance(rank, bool) or rank < 0
    ):
        raise NativeEvidenceError("invalid declared rank")

    window = _closed(
        capture["window"],
        {
            "start_monotonic_ns",
            "end_monotonic_ns",
            "producer_timeout_ns",
            "coordinator_timeout_ns",
            "max_output_bytes",
        },
        "window",
    )
    numbers = list(window.values())
    if not all(
        isinstance(item, int) and not isinstance(item, bool) for item in numbers
    ):
        raise NativeEvidenceError("window values must be integers")
    if (
        window["start_monotonic_ns"] < 0
        or window["end_monotonic_ns"] < window["start_monotonic_ns"]
        or window["producer_timeout_ns"] <= 0
        or window["coordinator_timeout_ns"] < window["producer_timeout_ns"]
        or window["max_output_bytes"] <= 0
    ):
        raise NativeEvidenceError("invalid capture window")
    if (
        window["end_monotonic_ns"] - window["start_monotonic_ns"]
        > window["coordinator_timeout_ns"]
    ):
        raise NativeEvidenceError("capture exceeded timeout")

    producer = _closed(
        capture["producer"],
        {
            "kind",
            "implementation_name",
            "implementation_version",
            "binary_sha256",
            "platform",
        },
        "producer",
    )
    if producer["kind"] not in {
        "stack_snapshot",
        "flight_recorder",
        "nccl_ras",
        "lifecycle_stage_flags",
    }:
        raise NativeEvidenceError("unknown producer kind")
    _bounded_token(producer["implementation_name"], "implementation name")
    _bounded_token(
        producer["implementation_version"],
        "implementation version",
        nullable=True,
    )
    _digest(producer["binary_sha256"], "producer binary digest", nullable=True)
    if producer["platform"] not in {"linux", "windows", "darwin"}:
        raise NativeEvidenceError("unknown producer platform")

    outcome = _closed(
        capture["outcome"],
        {"attempt_stage", "outcome_code", "raw_output_sha256"},
        "outcome",
    )
    stage = outcome["attempt_stage"]
    code = outcome["outcome_code"]
    if stage not in STAGE_OUTCOMES or code not in STAGE_OUTCOMES[stage]:
        raise NativeEvidenceError("invalid attempt-stage/outcome pair")
    raw_digest = _digest(
        outcome["raw_output_sha256"], "raw output digest", nullable=True
    )
    if stage != "execution" and raw_digest is not None:
        raise NativeEvidenceError("non-execution outcome has raw output")
    if stage == "execution" and code == "produced" and raw_digest is None:
        raise NativeEvidenceError("produced outcome lacks raw output")
    return capture


def validate_observation(
    observation: dict[str, Any], producer_kind: str
) -> dict[str, Any]:
    _closed(
        observation,
        {
            "thread_ref",
            "execution_domain",
            "gil_state",
            "ordered_frame_classes",
            "lifecycle_facts",
        },
        "normalized observation",
    )
    _bounded_token(observation["thread_ref"], "thread reference")
    if observation["execution_domain"] not in EXECUTION_DOMAINS:
        raise NativeEvidenceError("unknown execution domain")
    if observation["gil_state"] not in GIL_STATES:
        raise NativeEvidenceError("unknown GIL state")
    frames = observation["ordered_frame_classes"]
    if not isinstance(frames, list) or len(frames) > 128:
        raise NativeEvidenceError("invalid ordered_frame_classes")
    for value in frames:
        _bounded_token(value, "ordered_frame_classes")
    facts = observation["lifecycle_facts"]
    if not isinstance(facts, list) or len(facts) > len(LIFECYCLE_STAGES):
        raise NativeEvidenceError("invalid lifecycle facts")
    stage_sequences: dict[str, int] = {}
    sequence_values = set()
    for fact in facts:
        _closed(
            fact,
            {"component", "stage", "logical_sequence", "observed"},
            "lifecycle fact",
        )
        if fact["component"] != "process_group_nccl":
            raise NativeEvidenceError("unknown lifecycle component")
        if fact["stage"] not in LIFECYCLE_STAGES:
            raise NativeEvidenceError("unknown lifecycle stage")
        sequence = fact["logical_sequence"]
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
            raise NativeEvidenceError("invalid lifecycle sequence")
        if fact["observed"] is not True:
            raise NativeEvidenceError("invalid lifecycle observation")
        if fact["stage"] in stage_sequences or sequence in sequence_values:
            raise NativeEvidenceError("duplicate lifecycle stage or sequence")
        stage_sequences[fact["stage"]] = sequence
        sequence_values.add(sequence)
    for before, after in LIFECYCLE_ORDER:
        if (
            before in stage_sequences
            and after in stage_sequences
            and stage_sequences[before] >= stage_sequences[after]
        ):
            raise NativeEvidenceError("invalid lifecycle order")
    if producer_kind not in {
        "stack_snapshot",
        "flight_recorder",
        "nccl_ras",
        "lifecycle_stage_flags",
    }:
        raise NativeEvidenceError("unknown observation producer kind")
    if producer_kind != "lifecycle_stage_flags" and facts:
        raise NativeEvidenceError(
            "non-lifecycle observation carries lifecycle provenance"
        )
    if producer_kind != "stack_snapshot" and (
        frames
        or observation["execution_domain"] != "unknown"
        or observation["gil_state"] != "unknown"
    ):
        raise NativeEvidenceError("non-stack observation carries stack provenance")
    return observation


def _ordered_subsequence(required: list[str], observed: list[str]) -> bool:
    cursor = 0
    for item in observed:
        if cursor < len(required) and item == required[cursor]:
            cursor += 1
    return cursor == len(required)


def _rule_matches(
    rule: dict[str, Any],
    *,
    capture: dict[str, Any],
    observation: dict[str, Any],
    target: dict[str, Any],
) -> bool:
    _closed(rule, {"rule_set_id", "applies_to", "requires", "forbids", "emits"}, "rule")
    _bounded_token(rule["rule_set_id"], "rule-set id")
    applies = _closed(
        rule["applies_to"],
        {
            "producer_kind",
            "platform",
            "vllm_versions",
            "pytorch_versions",
            "pytorch_backends",
            "nccl_versions",
            "topologies",
        },
        "rule applies_to",
    )
    if applies["producer_kind"] not in {
        "stack_snapshot",
        "flight_recorder",
        "nccl_ras",
        "lifecycle_stage_flags",
    }:
        raise NativeEvidenceError("unknown rule producer kind")
    if applies["platform"] not in {"linux", "windows", "darwin"}:
        raise NativeEvidenceError("unknown rule platform")
    for name in (
        "vllm_versions",
        "pytorch_versions",
        "pytorch_backends",
        "nccl_versions",
        "topologies",
    ):
        values = applies[name]
        if not isinstance(values, list) or not values:
            raise NativeEvidenceError(f"empty rule constraint: {name}")
        for value in values:
            _bounded_token(value, f"rule {name}")
    if applies["producer_kind"] != capture["producer"]["kind"]:
        return False
    if applies["platform"] != capture["producer"]["platform"]:
        return False
    target_keys = {
        "vllm_versions": "vllm_version",
        "pytorch_versions": "pytorch_version",
        "pytorch_backends": "pytorch_backend",
        "nccl_versions": "nccl_version",
        "topologies": "topology",
    }
    for rule_key, target_key in target_keys.items():
        if target.get(target_key) not in applies[rule_key]:
            return False

    requires = _closed(
        rule["requires"],
        {"ordered_frame_classes", "lifecycle_stages"},
        "rule requires",
    )
    forbids = _closed(rule["forbids"], {"lifecycle_stages"}, "rule forbids")
    for values in (
        requires["ordered_frame_classes"],
        requires["lifecycle_stages"],
        forbids["lifecycle_stages"],
    ):
        if not isinstance(values, list):
            raise NativeEvidenceError("rule predicate is not a list")
        for value in values:
            _bounded_token(value, "rule predicate")
    frame_requirements = requires["ordered_frame_classes"]
    lifecycle_requirements = requires["lifecycle_stages"]
    if not frame_requirements and not lifecycle_requirements:
        raise NativeEvidenceError("rule has no required predicate")
    if frame_requirements and lifecycle_requirements:
        raise NativeEvidenceError("single-producer rule mixes provenance")
    if frame_requirements and applies["producer_kind"] != "stack_snapshot":
        raise NativeEvidenceError("non-stack rule requires stack frames")
    if lifecycle_requirements and applies["producer_kind"] != "lifecycle_stage_flags":
        raise NativeEvidenceError("non-lifecycle rule requires lifecycle facts")
    emits = _closed(rule["emits"], {"blocked_in"}, "rule emits")
    if emits["blocked_in"] not in BLOCKED_IN - {"unknown"}:
        raise NativeEvidenceError("rule emits an unadmitted attribution")
    if not _ordered_subsequence(
        requires["ordered_frame_classes"], observation["ordered_frame_classes"]
    ):
        return False
    stages = {
        fact["stage"] for fact in observation["lifecycle_facts"] if fact["observed"]
    }
    return set(requires["lifecycle_stages"]) <= stages and not (
        set(forbids["lifecycle_stages"]) & stages
    )


def _binding_digest(subject: dict[str, Any]) -> str:
    identity = {
        "kind": subject["process_identity_kind"],
        "value": subject["process_identity_value"],
        "role": subject["declared_role"],
        "rank": subject["declared_rank"],
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_target_runtime(target: dict[str, Any]) -> dict[str, Any]:
    _closed(
        target,
        {
            "vllm_version",
            "pytorch_version",
            "pytorch_backend",
            "nccl_version",
            "topology",
        },
        "target runtime",
    )
    for name, value in target.items():
        _bounded_token(value, f"target {name}")
    return target


def validate_attribution(value: dict[str, Any]) -> dict[str, Any]:
    _closed(
        value,
        {
            "attribution_schema_version",
            "subject_binding_digest",
            "binding_status",
            "capture_attempt_stage",
            "capture_outcome",
            "blocked_in",
            "execution_domain",
            "gil_state",
            "rule_set_id",
            "rule_match",
            "coverage",
            "raw_output_sha256",
        },
        "attribution",
    )
    if value["attribution_schema_version"] != ATTRIBUTION_SCHEMA:
        raise NativeEvidenceError("unknown attribution schema")
    _digest(value["subject_binding_digest"], "subject binding digest")
    if value["binding_status"] not in {"valid", "invalid"}:
        raise NativeEvidenceError("unknown binding status")
    stage = value["capture_attempt_stage"]
    outcome = value["capture_outcome"]
    if stage not in STAGE_OUTCOMES or outcome not in STAGE_OUTCOMES[stage]:
        raise NativeEvidenceError("invalid attribution stage/outcome")
    if value["blocked_in"] not in BLOCKED_IN:
        raise NativeEvidenceError("unknown blocked_in")
    if value["execution_domain"] not in EXECUTION_DOMAINS:
        raise NativeEvidenceError("unknown attribution execution domain")
    if value["gil_state"] not in GIL_STATES:
        raise NativeEvidenceError("unknown attribution GIL state")
    if value["rule_match"] not in {"exact", "unmatched"}:
        raise NativeEvidenceError("unknown rule match")
    if value["rule_match"] == "exact":
        _bounded_token(value["rule_set_id"], "matched rule-set id")
        if value["blocked_in"] == "unknown":
            raise NativeEvidenceError("exact rule emitted unknown")
    elif value["rule_set_id"] is not None or value["blocked_in"] != "unknown":
        raise NativeEvidenceError("unmatched attribution carries a claim")
    coverage = value["coverage"]
    if (
        not isinstance(coverage, list)
        or len(coverage) != len(set(coverage))
        or not set(coverage) <= COVERAGE
    ):
        raise NativeEvidenceError("invalid attribution coverage")
    raw_digest = _digest(
        value["raw_output_sha256"], "attribution raw digest", nullable=True
    )
    usable = (
        stage == "execution"
        and outcome == "produced"
        and value["binding_status"] == "valid"
    )
    if outcome == "produced" and raw_digest is None:
        raise NativeEvidenceError("produced attribution lacks raw digest")
    if not usable and (
        value["blocked_in"] != "unknown"
        or value["execution_domain"] != "unknown"
        or value["gil_state"] != "unknown"
        or value["rule_match"] != "unmatched"
        or coverage
    ):
        raise NativeEvidenceError("unusable attribution carries normalized facts")
    return value


def evaluate_attribution(
    capture: dict[str, Any],
    observation: dict[str, Any] | None,
    target: dict[str, Any],
    rules: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate closed facts without consulting producer implementation identity."""

    validate_capture(capture)
    validate_target_runtime(target)
    subject = capture["subject"]
    outcome = capture["outcome"]
    binding_valid = (
        subject["post_capture_identity_value"] is not None
        and subject["post_capture_identity_value"] == subject["process_identity_value"]
    )
    usable = (
        outcome["attempt_stage"] == "execution"
        and outcome["outcome_code"] == "produced"
        and binding_valid
    )
    if usable:
        if observation is None:
            raise NativeEvidenceError("produced capture lacks normalized observation")
        validate_observation(observation, capture["producer"]["kind"])
    elif observation is not None:
        raise NativeEvidenceError("unusable capture has normalized observation")

    matches = []
    if usable and observation is not None:
        matches = [
            rule
            for rule in rules
            if _rule_matches(
                rule, capture=capture, observation=observation, target=target
            )
        ]
        if len(matches) > 1:
            raise NativeEvidenceError("multiple attribution rules matched")

    match = matches[0] if matches else None
    blocked_in = match["emits"]["blocked_in"] if match else "unknown"
    coverage: list[str] = []
    if usable and observation is not None:
        if observation["ordered_frame_classes"]:
            coverage.append("thread_state")
        if observation["gil_state"] != "unknown":
            coverage.append("gil_state")
        if observation["lifecycle_facts"]:
            coverage.append("lifecycle_stage")
    if not set(coverage) <= COVERAGE:
        raise AssertionError("internal coverage vocabulary error")
    return validate_attribution(
        {
            "attribution_schema_version": ATTRIBUTION_SCHEMA,
            "subject_binding_digest": _binding_digest(subject),
            "binding_status": "valid" if binding_valid else "invalid",
            "capture_attempt_stage": outcome["attempt_stage"],
            "capture_outcome": outcome["outcome_code"],
            "blocked_in": blocked_in,
            "execution_domain": (
                observation["execution_domain"] if usable and observation else "unknown"
            ),
            "gil_state": observation["gil_state"]
            if usable and observation
            else "unknown",
            "rule_set_id": match["rule_set_id"] if match else None,
            "rule_match": "exact" if match else "unmatched",
            "coverage": coverage,
            "raw_output_sha256": outcome["raw_output_sha256"],
        }
    )


def compare_capture_triplet(
    first: dict[str, Any], candidate: dict[str, Any], repeat: dict[str, Any]
) -> dict[str, Any]:
    """Compare A/B/A2 at rule-match level, never by raw frame equality."""

    signature_keys = ("rule_set_id", "rule_match", "blocked_in")
    values = (first, candidate, repeat)
    for value in values:
        validate_attribution(value)
    usable = all(
        value["binding_status"] == "valid"
        and value["capture_attempt_stage"] == "execution"
        and value["capture_outcome"] == "produced"
        for value in values
    )
    if not usable:
        return {
            "pairing_result": "not_scorable",
            "not_scorable_reason": "unusable_capture",
            "stability_control_passed": False,
            "frame_sequence_compared": False,
            "coverage_equal": set(first["coverage"]) == set(candidate["coverage"]),
        }
    same_subject = (
        len(
            {
                first["subject_binding_digest"],
                candidate["subject_binding_digest"],
                repeat["subject_binding_digest"],
            }
        )
        == 1
    )
    stable = same_subject and all(first[key] == repeat[key] for key in signature_keys)
    if not stable:
        result = "target_not_stable"
        reason = None
    elif all(value["rule_match"] == "unmatched" for value in values):
        result = "not_scorable"
        reason = "no_admitted_rule"
    elif all(first[key] == candidate[key] for key in signature_keys):
        result = "interchangeable"
        reason = None
    else:
        result = "producer_disagreement"
        reason = None
    if result not in PAIRING_RESULTS:
        raise AssertionError("internal pairing vocabulary error")
    return {
        "pairing_result": result,
        "not_scorable_reason": reason,
        "stability_control_passed": stable,
        "frame_sequence_compared": False,
        "coverage_equal": set(first["coverage"]) == set(candidate["coverage"]),
    }
