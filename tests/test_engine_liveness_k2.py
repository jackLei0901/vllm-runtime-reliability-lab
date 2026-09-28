from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "engine-liveness-contract"
    / "k2_nested_shutdown_budget.py"
)
SPEC = importlib.util.spec_from_file_location("engine_liveness_k2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
k2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(k2)


def valid_cells() -> list[dict]:
    return [
        {
            "cell": "control",
            "worker_handlers_ready": True,
            "engine_sigterm_s": 0.01,
            "inner_start_s": 0.02,
            "inner_schedule_completed": True,
            "inner_sigterm_step_reached": True,
            "engine_exitcode": 0,
        },
        {
            "cell": "nested_x1",
            "worker_handlers_ready": True,
            "engine_sigterm_s": 0.01,
            "inner_start_s": 0.02,
            "inner_schedule_completed": False,
            "engine_sigkilled_by_outer": True,
        },
    ]


class K2ClassificationTest(unittest.TestCase):
    def test_valid_slow_worker_budget_shortfall(self) -> None:
        self.assertEqual(
            k2.classify_cells(valid_cells()),
            "c2_tested_nested_budget_shortfall",
        )

    def test_missing_inner_start_is_apparatus_failure(self) -> None:
        cells = valid_cells()
        cells[1]["inner_start_s"] = None
        self.assertEqual(k2.classify_cells(cells), "apparatus_failed")

    def test_unready_worker_is_apparatus_failure(self) -> None:
        cells = valid_cells()
        cells[1]["worker_handlers_ready"] = False
        self.assertEqual(k2.classify_cells(cells), "apparatus_failed")

    def test_control_must_complete_and_reach_sigterm(self) -> None:
        cells = valid_cells()
        cells[0]["inner_sigterm_step_reached"] = False
        self.assertEqual(k2.classify_cells(cells), "apparatus_failed")

    def test_nested_completion_is_not_shortfall(self) -> None:
        cells = valid_cells()
        cells[1]["inner_schedule_completed"] = True
        cells[1]["engine_sigkilled_by_outer"] = False
        self.assertEqual(k2.classify_cells(cells), "c2_not_reproduced")


if __name__ == "__main__":
    unittest.main()
