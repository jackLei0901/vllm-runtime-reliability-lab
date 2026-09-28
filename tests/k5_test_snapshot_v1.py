"""CPU-only checks for the K5 activation and closed scoring rules."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1] / "experiments" / "engine-liveness-contract"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


plugin = load("llr_k5_worker_hold_test", ROOT / "k5-worker-hold-plugin" /
              "llr_k5_worker_hold.py")
gate = load("k5_health_stall_gate_test", ROOT / "k5_health_stall_gate.py")


def samples(before: int = 200, after: int = 200) -> list[dict]:
    return [{"t": float(t), "status": before if t <= 26 else after}
            for t in range(2, 49, 2)]


class TestK5Activation(unittest.TestCase):
    def test_only_armed_rank_zero_with_work_is_eligible(self):
        args = {"armed": True, "claimed": False, "rank": 0,
                "tp": 2, "expected_tp": 2, "scheduled_tokens": 1}
        self.assertTrue(plugin.eligible(**args))
        for change in ({"armed": False}, {"claimed": True}, {"rank": 1},
                       {"tp": 1}, {"scheduled_tokens": 0}):
            self.assertFalse(plugin.eligible(**(args | change)))

    def test_marker_needs_live_descendant_and_start_identity(self):
        marker = {"schema": "llr-k5-worker-hold-v1", "pid": 42,
                  "start_ticks": 123, "rank": 0, "tp": 2,
                  "scheduled_tokens_positive": True, "monotonic_ns": 500}
        with patch.object(gate, "_proc_stat", return_value=(20, 123)), \
             patch.object(gate, "_descends_from", return_value=True):
            self.assertTrue(gate._check_marker(marker, tp=2, root_pid=10,
                                               root_ticks=90))
        with patch.object(gate, "_proc_stat", return_value=(20, 124)), \
             patch.object(gate, "_descends_from", return_value=True):
            self.assertFalse(gate._check_marker(marker, tp=2, root_pid=10,
                                                root_ticks=90))
        self.assertFalse(gate._check_marker(None, tp=2, root_pid=10,
                                            root_ticks=90))


class TestK5Scoring(unittest.TestCase):
    def test_tp1_positive(self):
        self.assertEqual(gate.score(
            "tp1", baseline_ok=True, marker_ok=True, samples=samples(),
            completion={"status": 200, "valid_response": True},
            released=True, timeout_logged=False,
            post_health=200), "tp1_no_timeout_health_200")

    def test_tp2_positive_requires_timeout_log(self):
        expected = "tp2_timeout_health_503"
        self.assertEqual(gate.score(
            "tp2", baseline_ok=True, marker_ok=True,
            samples=samples(after=503), completion={"status": 500},
            released=False, timeout_logged=True, post_health=503), expected)
        self.assertEqual(gate.score(
            "tp2", baseline_ok=True, marker_ok=True,
            samples=samples(after=503), completion={"status": 500},
            released=False, timeout_logged=False, post_health=503),
            "unscored_timeout_attribution")

    def test_missing_witness_cannot_pass(self):
        self.assertEqual(gate.score(
            "tp1", baseline_ok=True, marker_ok=False, samples=samples(),
            completion={"status": 200}, released=True, timeout_logged=False,
            post_health=200), "unscored_apparatus")

    def test_tp1_early_unhealthy_is_contradiction(self):
        self.assertEqual(gate.score(
            "tp1", baseline_ok=True, marker_ok=True,
            samples=samples(after=503), completion={"status": 200},
            released=True, timeout_logged=False, post_health=200),
            "contradicted_tp1_unhealthy")

    def test_tp2_early_unhealthy_is_contradiction(self):
        self.assertEqual(gate.score(
            "tp2", baseline_ok=True, marker_ok=True,
            samples=samples(before=503, after=503), completion={},
            released=False, timeout_logged=True, post_health=503),
            "contradicted_early_unhealthy")

    def test_tp2_stays_healthy_is_contradiction(self):
        self.assertEqual(gate.score(
            "tp2", baseline_ok=True, marker_ok=True, samples=samples(),
            completion={}, released=False, timeout_logged=False,
            post_health=200), "contradicted_no_timeout")

    def test_missing_window_is_unscored(self):
        self.assertEqual(gate.score(
            "tp2", baseline_ok=True, marker_ok=True,
            samples=[{"t": 3.0, "status": 200}], completion={},
            released=False, timeout_logged=False, post_health=None),
            "unscored_window")


if __name__ == "__main__":
    unittest.main()
