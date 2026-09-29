import importlib.util
import unittest
from pathlib import Path


def _load_probe():
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments/pr54553-fatal-shutdown/probe.py"
    )
    spec = importlib.util.spec_from_file_location("pr54553_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(case, code, *, timeout=False, elapsed=1.0):
    markers = ["source_verified", "outer_path_entered", "teardown_entered"]
    if case != "clean_quick":
        markers.append("engine_dead_sent")
    if case != "fatal_hold":
        markers.append("teardown_completed")
    return {
        "case": case,
        "exit_code": code,
        "parent_timeout": timeout,
        "elapsed_after_teardown_s": elapsed,
        "markers": markers,
    }


def _cells():
    base = {
        "fatal_hold": _row("fatal_hold", -9, timeout=True, elapsed=4.0),
        "fatal_quick": _row("fatal_quick", 1),
        "clean_quick": _row("clean_quick", 0),
    }
    patch = {
        **base,
        "fatal_hold": _row("fatal_hold", 1, elapsed=1.0),
    }
    return base, patch


class ProbeScoringTests(unittest.TestCase):
    def test_expected_contrast(self):
        base, patch = _cells()
        self.assertEqual(_load_probe().score(base, patch), "supported")

    def test_patch_still_hangs_is_refuted(self):
        base, patch = _cells()
        patch["fatal_hold"] = base["fatal_hold"]
        self.assertEqual(_load_probe().score(base, patch), "refuted")

    def test_clean_forced_exit_is_refuted(self):
        base, patch = _cells()
        patch["clean_quick"] = _row("clean_quick", 1)
        self.assertEqual(_load_probe().score(base, patch), "refuted")

    def test_missing_entry_witness_is_unscored(self):
        base, patch = _cells()
        patch["fatal_hold"]["markers"].remove("teardown_entered")
        self.assertEqual(_load_probe().score(base, patch), "unscored")

    def test_missing_completion_witness_is_unscored(self):
        base, patch = _cells()
        patch["clean_quick"]["markers"].remove("teardown_completed")
        self.assertEqual(_load_probe().score(base, patch), "unscored")

    def test_premature_fatal_exit_is_refuted(self):
        base, patch = _cells()
        patch["fatal_hold"]["elapsed_after_teardown_s"] = 0.2
        self.assertEqual(_load_probe().score(base, patch), "refuted")
