from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "tests" / "fixtures" / "collect_verify_v0_2" / "cases.json"

PRODUCER_STATES = {
    "progressing",
    "flat",
    "producer_missing",
    "insufficient_evidence",
}
DEMAND_STATES = {"present", "absent", "insufficient_evidence"}
VERDICTS = {
    "progress_observed",
    "alive_health_ok_no_progress",
    "process_missing",
    "health_lost",
    "undetermined",
}
HEALTH_STATES = {"ok", "lost", "unavailable", "insufficient_evidence"}
STACK_STATES = {"disabled", "produced", "unavailable"}
DECISION_SCOPES = {
    "server_counter": "service",
    "client_request": "request",
}

REQUIRED_PRODUCER_CASES = {
    "server-counter-rising",
    "server-counter-flat",
    "server-counter-producer-missing",
    "server-counter-one-sample",
    "server-counter-reset",
    "server-counter-cached-values-do-not-count",
    "client-content-progress",
    "client-non-content-chunks-remain-flat",
    "client-completed-before-interval-end",
}
REQUIRED_DEMAND_CASES = {
    "server-demand-present",
    "server-demand-absent",
    "server-demand-contradictory",
    "client-demand-present",
    "client-demand-ended",
    "client-demand-started-after-interval",
    "server-demand-cached-values-do-not-count",
}
REQUIRED_VERDICT_CASES = {
    "idle-flat-is-undetermined",
    "service-no-progress-with-demand",
    "request-no-progress-despite-global-progress",
    "decision-progress-observed",
    "service-progress-despite-stuck-request",
    "missing-decision-producer-is-undetermined",
    "insufficient-decision-evidence-is-undetermined",
    "pid-reuse-is-process-missing",
    "process-exit-is-process-missing",
    "health-loss-precedes-progress",
    "pid-only-survival-is-undetermined",
    "pid-only-exit-is-process-missing",
    "stack-unavailable-does-not-change-verdict",
    "health-evidence-insufficient-for-no-progress",
    "endpoint-never-healthy-is-undetermined",
    "endpoint-only-flat-is-undetermined",
}


def load_cases() -> dict[str, Any]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def ids(cases: list[dict[str, Any]]) -> set[str]:
    return {case["id"] for case in cases}


class CollectVerifyContractCasesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = load_cases()

    def test_root_shape_is_closed(self) -> None:
        self.assertEqual(
            set(self.payload),
            {
                "schema_version",
                "case_count",
                "producer_cases",
                "demand_cases",
                "verdict_cases",
            },
        )
        self.assertEqual(
            self.payload["schema_version"], "collect-verify-contract-cases-v1"
        )
        self.assertEqual(
            self.payload["case_count"],
            sum(
                len(self.payload[name])
                for name in ("producer_cases", "demand_cases", "verdict_cases")
            ),
        )

    def test_case_ids_are_unique_across_the_matrix(self) -> None:
        all_cases = [
            *self.payload["producer_cases"],
            *self.payload["demand_cases"],
            *self.payload["verdict_cases"],
        ]
        case_ids = [case["id"] for case in all_cases]
        self.assertEqual(len(case_ids), len(set(case_ids)))

    def test_required_boundary_cases_are_present(self) -> None:
        self.assertEqual(ids(self.payload["producer_cases"]), REQUIRED_PRODUCER_CASES)
        self.assertEqual(ids(self.payload["demand_cases"]), REQUIRED_DEMAND_CASES)
        self.assertEqual(ids(self.payload["verdict_cases"]), REQUIRED_VERDICT_CASES)

    def test_producer_case_shapes_and_states_are_closed(self) -> None:
        for case in self.payload["producer_cases"]:
            with self.subTest(case=case["id"]):
                self.assertEqual(set(case), {"id", "kind", "input", "expected"})
                self.assertIn(case["kind"], DECISION_SCOPES)
                self.assertEqual(set(case["expected"]), {"state", "reason"})
                self.assertIn(case["expected"]["state"], PRODUCER_STATES)
                self.assertIsInstance(case["expected"]["reason"], str)
                self.assertTrue(case["expected"]["reason"])

    def test_server_counter_samples_have_closed_shape(self) -> None:
        for case in self.payload["producer_cases"]:
            if case["kind"] != "server_counter":
                continue
            with self.subTest(case=case["id"]):
                value = case["input"]
                self.assertEqual(
                    set(value),
                    {
                        "producer_available",
                        "evaluation_start_ns",
                        "evaluation_end_ns",
                        "minimum_span_ns",
                        "minimum_fresh_samples",
                        "samples",
                    },
                )
                for sample in value["samples"]:
                    self.assertEqual(set(sample), {"monotonic_ns", "fresh", "value"})

    def test_client_request_inputs_have_closed_shape(self) -> None:
        for case in self.payload["producer_cases"]:
            if case["kind"] != "client_request":
                continue
            with self.subTest(case=case["id"]):
                value = case["input"]
                self.assertEqual(
                    set(value),
                    {
                        "producer_available",
                        "evaluation_start_ns",
                        "evaluation_end_ns",
                        "request_started_ns",
                        "request_completed_ns",
                        "chunks",
                    },
                )
                for chunk in value["chunks"]:
                    self.assertEqual(set(chunk), {"monotonic_ns", "kind"})
                    self.assertIn(
                        chunk["kind"], {"content", "role_only", "usage_only", "empty"}
                    )

    def test_demand_case_shapes_and_states_are_closed(self) -> None:
        for case in self.payload["demand_cases"]:
            with self.subTest(case=case["id"]):
                self.assertEqual(set(case), {"id", "kind", "input", "expected"})
                self.assertIn(case["kind"], DECISION_SCOPES)
                self.assertEqual(set(case["expected"]), {"state", "reason"})
                self.assertIn(case["expected"]["state"], DEMAND_STATES)

    def test_demand_inputs_have_closed_shape(self) -> None:
        for case in self.payload["demand_cases"]:
            with self.subTest(case=case["id"]):
                value = case["input"]
                if case["kind"] == "server_counter":
                    self.assertEqual(
                        set(value),
                        {
                            "evaluation_start_ns",
                            "evaluation_end_ns",
                            "minimum_fresh_samples",
                            "samples",
                        },
                    )
                    for sample in value["samples"]:
                        self.assertEqual(
                            set(sample),
                            {"monotonic_ns", "fresh", "running", "waiting"},
                        )
                else:
                    self.assertEqual(
                        set(value),
                        {
                            "evaluation_start_ns",
                            "evaluation_end_ns",
                            "request_started_ns",
                            "request_completed_ns",
                        },
                    )

    def test_verdict_case_shape_and_vocabulary_are_closed(self) -> None:
        expected_input_keys = {
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
        for case in self.payload["verdict_cases"]:
            with self.subTest(case=case["id"]):
                self.assertEqual(set(case), {"id", "input", "expected"})
                value = case["input"]
                expected = case["expected"]
                self.assertEqual(set(value), expected_input_keys)
                self.assertEqual(
                    set(expected),
                    {"verdict", "progress_scope", "producer_conflict"},
                )
                self.assertIn(value["health_state"], HEALTH_STATES)
                self.assertIn(value["decision_source"], DECISION_SCOPES)
                self.assertIn(value["decision_state"], PRODUCER_STATES)
                self.assertIn(value["corroborating_state"], PRODUCER_STATES)
                self.assertIn(value["demand_state"], DEMAND_STATES)
                self.assertIn(value["stack_state"], STACK_STATES)
                self.assertIn(expected["verdict"], VERDICTS)
                self.assertEqual(
                    expected["progress_scope"],
                    DECISION_SCOPES[value["decision_source"]],
                )

    def test_conflict_flag_is_derived_from_present_producers(self) -> None:
        comparable = {"progressing", "flat"}
        for case in self.payload["verdict_cases"]:
            with self.subTest(case=case["id"]):
                value = case["input"]
                conflict = (
                    value["decision_state"] in comparable
                    and value["corroborating_state"] in comparable
                    and value["decision_state"] != value["corroborating_state"]
                )
                self.assertEqual(case["expected"]["producer_conflict"], conflict)

    def test_positive_verdicts_have_their_required_evidence(self) -> None:
        for case in self.payload["verdict_cases"]:
            with self.subTest(case=case["id"]):
                value = case["input"]
                verdict = case["expected"]["verdict"]
                if verdict == "process_missing":
                    self.assertTrue(value["process_supplied"])
                    self.assertTrue(
                        not value["process_identity_stable"]
                        or not value["process_alive_throughout"]
                    )
                elif verdict == "health_lost":
                    self.assertEqual(value["health_state"], "lost")
                    self.assertTrue(value["process_identity_stable"])
                    self.assertTrue(value["process_alive_throughout"])
                elif verdict == "progress_observed":
                    self.assertEqual(value["decision_state"], "progressing")
                    self.assertNotEqual(value["health_state"], "lost")
                    if value["process_supplied"]:
                        self.assertTrue(value["process_identity_stable"])
                        self.assertTrue(value["process_alive_throughout"])
                elif verdict == "alive_health_ok_no_progress":
                    self.assertTrue(value["process_supplied"])
                    self.assertTrue(value["process_identity_stable"])
                    self.assertTrue(value["process_alive_throughout"])
                    self.assertTrue(value["endpoint_supplied"])
                    self.assertEqual(value["health_state"], "ok")
                    self.assertEqual(value["decision_state"], "flat")
                    self.assertEqual(value["demand_state"], "present")


if __name__ == "__main__":
    unittest.main()
