from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path

from dfxlab.native_evidence import (
    NOT_SCORABLE_REASONS,
    NativeEvidenceError,
    compare_capture_triplet,
    evaluate_attribution,
    validate_capture,
    validate_observation,
)

DIGEST = hashlib.sha256(b"raw stack").hexdigest()


def capture() -> dict:
    return {
        "schema_version": "native-evidence-experiment-v0",
        "subject": {
            "process_identity_kind": "linux_proc_start_ticks",
            "process_identity_value": "12345",
            "post_capture_identity_value": "12345",
            "declared_role": "engine_core",
            "declared_rank": None,
        },
        "window": {
            "start_monotonic_ns": 100,
            "end_monotonic_ns": 150,
            "producer_timeout_ns": 100,
            "coordinator_timeout_ns": 200,
            "max_output_bytes": 65536,
        },
        "producer": {
            "kind": "stack_snapshot",
            "implementation_name": "pystack",
            "implementation_version": "1.7.1",
            "binary_sha256": hashlib.sha256(b"pystack").hexdigest(),
            "platform": "linux",
        },
        "outcome": {
            "attempt_stage": "execution",
            "outcome_code": "produced",
            "raw_output_sha256": DIGEST,
        },
    }


def observation() -> dict:
    return {
        "thread_ref": "thread-1",
        "execution_domain": "mixed",
        "gil_state": "waiting",
        "ordered_frame_classes": [
            "python:publisher",
            "python:queue-put",
            "native:condition-wait",
        ],
        "lifecycle_facts": [],
    }


def target() -> dict:
    return {
        "vllm_version": "0.11.0.dev",
        "pytorch_version": "2.13.0",
        "pytorch_backend": "none",
        "nccl_version": "none",
        "topology": "single-process",
    }


def rule() -> dict:
    return {
        "rule_set_id": "fixture-queue-wait-v1",
        "applies_to": {
            "producer_kind": "stack_snapshot",
            "platform": "linux",
            "vllm_versions": ["0.11.0.dev"],
            "pytorch_versions": ["2.13.0"],
            "pytorch_backends": ["none"],
            "nccl_versions": ["none"],
            "topologies": ["single-process"],
        },
        "requires": {
            "ordered_frame_classes": [
                "python:queue-put",
                "native:condition-wait",
            ],
            "lifecycle_stages": [],
        },
        "forbids": {"lifecycle_stages": []},
        "emits": {"blocked_in": "queue_wait"},
    }


class NativeCaptureContractTest(unittest.TestCase):
    def test_valid_execution_capture(self) -> None:
        value = capture()
        self.assertIs(validate_capture(value), value)

    def test_stage_outcome_matrix_is_closed(self) -> None:
        value = capture()
        value["outcome"] = {
            "attempt_stage": "preflight",
            "outcome_code": "permission_denied",
            "raw_output_sha256": None,
        }
        with self.assertRaisesRegex(NativeEvidenceError, "stage/outcome"):
            validate_capture(value)

    def test_not_requested_requires_no_raw_output(self) -> None:
        value = capture()
        value["outcome"] = {
            "attempt_stage": "not_requested",
            "outcome_code": "disabled",
            "raw_output_sha256": DIGEST,
        }
        with self.assertRaisesRegex(NativeEvidenceError, "non-execution"):
            validate_capture(value)

    def test_produced_requires_digest(self) -> None:
        value = capture()
        value["outcome"]["raw_output_sha256"] = None
        with self.assertRaisesRegex(NativeEvidenceError, "lacks raw output"):
            validate_capture(value)

    def test_missing_post_capture_identity_invalidates_binding(self) -> None:
        value = capture()
        value["subject"]["post_capture_identity_value"] = None
        result = evaluate_attribution(value, None, target(), [rule()])
        self.assertEqual("invalid", result["binding_status"])
        self.assertEqual("unknown", result["blocked_in"])


class NativeAttributionTest(unittest.TestCase):
    def test_published_stage_b_rule_keeps_legacy_stack_shape(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "native-evidence-capability"
            / "stage_b_queue_wait_rule.json"
        )
        raw = path.read_bytes()
        self.assertEqual(
            "9c2612d3171e150c9a8fdd910ca3772315915af18bdbead582244330284c958f",
            hashlib.sha256(raw).hexdigest(),
        )
        published_rule = json.loads(raw)
        runtime = {
            "vllm_version": "0.1.dev1+g9935dfceb",
            "pytorch_version": "2.13.0+cu130",
            "pytorch_backend": "nccl",
            "nccl_version": "2.29.7",
            "topology": "single_gpu",
        }
        result = evaluate_attribution(
            capture(), observation(), runtime, [published_rule]
        )
        self.assertEqual("exact", result["rule_match"])

    def test_exact_rule_emits_closed_attribution(self) -> None:
        result = evaluate_attribution(capture(), observation(), target(), [rule()])
        self.assertEqual("queue_wait", result["blocked_in"])
        self.assertEqual("exact", result["rule_match"])
        self.assertEqual(
            ["thread_state", "gil_state"],
            result["coverage"],
        )

    def test_implementation_identity_is_non_decisional(self) -> None:
        first = evaluate_attribution(capture(), observation(), target(), [rule()])
        other = capture()
        other["producer"]["implementation_name"] = "py-spy"
        other["producer"]["implementation_version"] = "0.4.1"
        second = evaluate_attribution(other, observation(), target(), [rule()])
        for key in ("blocked_in", "rule_set_id", "rule_match"):
            self.assertEqual(first[key], second[key])

    def test_out_of_range_version_is_unknown(self) -> None:
        runtime = target()
        runtime["vllm_version"] = "0.12.0"
        result = evaluate_attribution(capture(), observation(), runtime, [rule()])
        self.assertEqual("unknown", result["blocked_in"])
        self.assertEqual("unmatched", result["rule_match"])

    def test_unmatched_frames_are_unknown(self) -> None:
        facts = observation()
        facts["ordered_frame_classes"] = ["python:unrelated"]
        result = evaluate_attribution(capture(), facts, target(), [rule()])
        self.assertEqual("unknown", result["blocked_in"])

    def test_pid_reuse_invalidates_attribution(self) -> None:
        value = capture()
        value["subject"]["post_capture_identity_value"] = "54321"
        result = evaluate_attribution(value, None, target(), [rule()])
        self.assertEqual("invalid", result["binding_status"])
        self.assertEqual("unknown", result["blocked_in"])

    def test_unusable_capture_cannot_smuggle_observation(self) -> None:
        value = capture()
        value["outcome"] = {
            "attempt_stage": "preflight",
            "outcome_code": "unsupported",
            "raw_output_sha256": None,
        }
        value["subject"]["post_capture_identity_value"] = None
        with self.assertRaisesRegex(NativeEvidenceError, "unusable capture"):
            evaluate_attribution(value, observation(), target(), [rule()])

    def test_invalid_lifecycle_order_fails_closed(self) -> None:
        facts = observation()
        facts["ordered_frame_classes"] = []
        facts["execution_domain"] = "unknown"
        facts["gil_state"] = "unknown"
        facts["lifecycle_facts"] = [
            {
                "component": "process_group_nccl",
                "stage": "communicator_destroy_completed",
                "logical_sequence": 1,
                "observed": True,
            },
            {
                "component": "process_group_nccl",
                "stage": "communicator_destroy_started",
                "logical_sequence": 2,
                "observed": True,
            },
        ]
        with self.assertRaisesRegex(NativeEvidenceError, "lifecycle order"):
            validate_observation(facts, "lifecycle_stage_flags")

    def test_fix_arm_destroy_before_stop_request_is_valid(self) -> None:
        facts = observation()
        facts["ordered_frame_classes"] = []
        facts["execution_domain"] = "unknown"
        facts["gil_state"] = "unknown"
        facts["lifecycle_facts"] = [
            {
                "component": "process_group_nccl",
                "stage": stage,
                "logical_sequence": sequence,
                "observed": True,
            }
            for sequence, stage in enumerate(
                (
                    "dump_responder_active",
                    "communicator_destroy_started",
                    "communicator_destroy_completed",
                    "dump_responder_stop_requested",
                )
            )
        ]
        self.assertIs(validate_observation(facts, "lifecycle_stage_flags"), facts)

    def test_unpatched_order_is_versioned_rule_predicate(self) -> None:
        value = capture()
        value["producer"]["kind"] = "lifecycle_stage_flags"
        facts = observation()
        facts["ordered_frame_classes"] = []
        facts["execution_domain"] = "unknown"
        facts["gil_state"] = "unknown"
        facts["lifecycle_facts"] = [
            {
                "component": "process_group_nccl",
                "stage": stage,
                "logical_sequence": sequence,
                "observed": True,
            }
            for sequence, stage in enumerate(
                ("dump_responder_stop_requested", "communicator_destroy_started")
            )
        ]
        base_rule = rule()
        base_rule["applies_to"]["producer_kind"] = "lifecycle_stage_flags"
        base_rule["applies_to"]["pytorch_versions"] = ["2.13.0"]
        base_rule["applies_to"]["pytorch_source_revisions"] = ["a" * 40]
        base_rule["requires"] = {
            "ordered_frame_classes": [],
            "lifecycle_stages": [
                "dump_responder_stop_requested",
                "communicator_destroy_started",
            ],
            "lifecycle_order": [
                ["dump_responder_stop_requested", "communicator_destroy_started"]
            ],
        }
        base_rule["emits"]["blocked_in"] = "communicator_destruction"
        base_target = target()
        base_target["pytorch_source_revision"] = "a" * 40
        result = evaluate_attribution(value, facts, base_target, [base_rule])
        self.assertEqual("exact", result["rule_match"])
        reversed_facts = copy.deepcopy(facts)
        reversed_facts["lifecycle_facts"][0]["logical_sequence"] = 2
        reversed_facts["lifecycle_facts"][1]["logical_sequence"] = 1
        result = evaluate_attribution(value, reversed_facts, base_target, [base_rule])
        self.assertEqual("unmatched", result["rule_match"])
        other_version = copy.deepcopy(base_target)
        other_version["pytorch_version"] = "2.15.0a0+fix"
        result = evaluate_attribution(value, facts, other_version, [base_rule])
        self.assertEqual("unmatched", result["rule_match"])
        other_revision = copy.deepcopy(base_target)
        other_revision["pytorch_source_revision"] = "b" * 40
        result = evaluate_attribution(value, facts, other_revision, [base_rule])
        self.assertEqual("unmatched", result["rule_match"])

    def test_lifecycle_rule_rejects_order_outside_required_stages(self) -> None:
        value = capture()
        value["producer"]["kind"] = "lifecycle_stage_flags"
        facts = observation()
        facts["ordered_frame_classes"] = []
        facts["execution_domain"] = "unknown"
        facts["gil_state"] = "unknown"
        facts["lifecycle_facts"] = [
            {
                "component": "process_group_nccl",
                "stage": "dump_responder_stop_requested",
                "logical_sequence": 1,
                "observed": True,
            }
        ]
        bad = rule()
        bad["applies_to"]["producer_kind"] = "lifecycle_stage_flags"
        bad["applies_to"]["pytorch_source_revisions"] = ["a" * 40]
        bad["requires"] = {
            "ordered_frame_classes": [],
            "lifecycle_stages": ["dump_responder_stop_requested"],
            "lifecycle_order": [
                ["dump_responder_stop_requested", "communicator_destroy_started"]
            ],
        }
        runtime = target()
        runtime["pytorch_source_revision"] = "a" * 40
        with self.assertRaisesRegex(NativeEvidenceError, "rule lifecycle order"):
            evaluate_attribution(value, facts, runtime, [bad])

    def test_lifecycle_rule_requires_exact_source_revision(self) -> None:
        value = capture()
        value["producer"]["kind"] = "lifecycle_stage_flags"
        facts = observation()
        facts["ordered_frame_classes"] = []
        facts["execution_domain"] = "unknown"
        facts["gil_state"] = "unknown"
        facts["lifecycle_facts"] = [
            {
                "component": "process_group_nccl",
                "stage": "dump_responder_stop_requested",
                "logical_sequence": 1,
                "observed": True,
            }
        ]
        lifecycle_rule = rule()
        lifecycle_rule["applies_to"]["producer_kind"] = "lifecycle_stage_flags"
        lifecycle_rule["requires"] = {
            "ordered_frame_classes": [],
            "lifecycle_stages": ["dump_responder_stop_requested"],
            "lifecycle_order": [],
        }
        with self.assertRaisesRegex(NativeEvidenceError, "rule applies_to shape"):
            evaluate_attribution(value, facts, target(), [lifecycle_rule])

    def test_ambiguous_rules_fail_closed(self) -> None:
        duplicate = copy.deepcopy(rule())
        duplicate["rule_set_id"] = "fixture-queue-wait-v2"
        with self.assertRaisesRegex(NativeEvidenceError, "multiple"):
            evaluate_attribution(
                capture(), observation(), target(), [rule(), duplicate]
            )

    def test_rule_without_required_predicate_fails_closed(self) -> None:
        empty = rule()
        empty["requires"] = {
            "ordered_frame_classes": [],
            "lifecycle_stages": [],
        }
        with self.assertRaisesRegex(NativeEvidenceError, "no required predicate"):
            evaluate_attribution(capture(), observation(), target(), [empty])

    def test_single_producer_rule_cannot_mix_provenance(self) -> None:
        mixed = rule()
        mixed["requires"]["lifecycle_stages"] = ["dump_responder_stop_requested"]
        with self.assertRaisesRegex(NativeEvidenceError, "mixes provenance"):
            evaluate_attribution(capture(), observation(), target(), [mixed])

    def test_stack_rule_rejects_dead_lifecycle_forbid(self) -> None:
        dead_constraint = rule()
        dead_constraint["forbids"]["lifecycle_stages"] = ["dump_completed"]
        with self.assertRaisesRegex(NativeEvidenceError, "dead lifecycle forbids"):
            evaluate_attribution(capture(), observation(), target(), [dead_constraint])

    def test_stack_observation_cannot_launder_lifecycle_facts(self) -> None:
        facts = observation()
        facts["lifecycle_facts"] = [
            {
                "component": "process_group_nccl",
                "stage": "dump_responder_stop_requested",
                "logical_sequence": 1,
                "observed": True,
            }
        ]
        with self.assertRaisesRegex(NativeEvidenceError, "lifecycle provenance"):
            evaluate_attribution(capture(), facts, target(), [rule()])


class ProducerPairingTest(unittest.TestCase):
    def attribution(self) -> dict:
        return evaluate_attribution(capture(), observation(), target(), [rule()])

    def test_frame_and_coverage_difference_does_not_break_pair(self) -> None:
        first = self.attribution()
        candidate = copy.deepcopy(first)
        candidate["coverage"] = ["thread_state"]
        repeat = copy.deepcopy(first)
        result = compare_capture_triplet(first, candidate, repeat)
        self.assertEqual("interchangeable", result["pairing_result"])
        self.assertFalse(result["frame_sequence_compared"])
        self.assertFalse(result["coverage_equal"])

    def test_same_tool_instability_stops_comparison(self) -> None:
        first = self.attribution()
        candidate = copy.deepcopy(first)
        repeat = copy.deepcopy(first)
        repeat["blocked_in"] = "unknown"
        repeat["rule_match"] = "unmatched"
        repeat["rule_set_id"] = None
        result = compare_capture_triplet(first, candidate, repeat)
        self.assertEqual("target_not_stable", result["pairing_result"])
        self.assertFalse(result["stability_control_passed"])

    def test_cross_tool_rule_disagreement_is_explicit(self) -> None:
        first = self.attribution()
        candidate = copy.deepcopy(first)
        candidate["blocked_in"] = "unknown"
        candidate["rule_match"] = "unmatched"
        candidate["rule_set_id"] = None
        result = compare_capture_triplet(first, candidate, copy.deepcopy(first))
        self.assertEqual("producer_disagreement", result["pairing_result"])

    def test_different_subject_stops_comparison(self) -> None:
        first = self.attribution()
        candidate = copy.deepcopy(first)
        candidate["subject_binding_digest"] = hashlib.sha256(b"other").hexdigest()
        result = compare_capture_triplet(first, candidate, copy.deepcopy(first))
        self.assertEqual("target_not_stable", result["pairing_result"])

    def test_unusable_attribution_cannot_enter_pair(self) -> None:
        first = self.attribution()
        timed_out = capture()
        timed_out["outcome"]["outcome_code"] = "timeout"
        candidate = evaluate_attribution(timed_out, None, target(), [rule()])
        result = compare_capture_triplet(first, candidate, copy.deepcopy(first))
        self.assertEqual("not_scorable", result["pairing_result"])
        self.assertEqual("unusable_capture", result["not_scorable_reason"])

    def test_three_unmatched_attributions_are_not_scorable(self) -> None:
        unmatched = evaluate_attribution(capture(), observation(), target(), [])
        result = compare_capture_triplet(
            unmatched, copy.deepcopy(unmatched), copy.deepcopy(unmatched)
        )
        self.assertEqual("not_scorable", result["pairing_result"])
        self.assertEqual("no_admitted_rule", result["not_scorable_reason"])

    def test_not_scorable_reason_vocabulary_is_closed(self) -> None:
        self.assertEqual(
            {"no_admitted_rule", "unusable_capture"},
            NOT_SCORABLE_REASONS,
        )


if __name__ == "__main__":
    unittest.main()
