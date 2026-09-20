from __future__ import annotations

import json
import unittest
from pathlib import Path

from dfxlab.progress import (
    ProgressContractError,
    derive_verdict,
    evaluate_demand,
    evaluate_producer,
)

ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "tests" / "fixtures" / "collect_verify_v0_2" / "cases.json"


class ProgressContractImplementationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))

    def test_producer_vectors_without_adapter(self) -> None:
        for case in self.cases["producer_cases"]:
            with self.subTest(case=case["id"]):
                self.assertEqual(
                    evaluate_producer(case["kind"], case["input"]),
                    case["expected"],
                )

    def test_demand_vectors_without_adapter(self) -> None:
        for case in self.cases["demand_cases"]:
            with self.subTest(case=case["id"]):
                self.assertEqual(
                    evaluate_demand(case["kind"], case["input"]),
                    case["expected"],
                )

    def test_verdict_vectors_without_adapter(self) -> None:
        for case in self.cases["verdict_cases"]:
            with self.subTest(case=case["id"]):
                self.assertEqual(derive_verdict(case["input"]), case["expected"])

    def test_unknown_fields_fail_closed(self) -> None:
        value = dict(self.cases["verdict_cases"][0]["input"])
        value["unexpected"] = True
        with self.assertRaisesRegex(ProgressContractError, "verdict_shape"):
            derive_verdict(value)

    def test_unknown_producer_kind_fails_closed(self) -> None:
        with self.assertRaisesRegex(ProgressContractError, "producer_kind"):
            evaluate_producer("merged", {})

    def test_invalid_interval_fails_closed(self) -> None:
        value = dict(self.cases["producer_cases"][0]["input"])
        value["evaluation_end_ns"] = value["evaluation_start_ns"]
        with self.assertRaisesRegex(ProgressContractError, "evaluation_interval"):
            evaluate_producer("server_counter", value)


if __name__ == "__main__":
    unittest.main()
