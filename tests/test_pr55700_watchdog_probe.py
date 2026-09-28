from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "pr55700-watchdog-metrics"
    / "probe.py"
)
SPEC = importlib.util.spec_from_file_location("pr55700_probe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)

EXPOSITION = """\
# HELP vllm:watchdog_timeouts_total Feed-timeout events detected by the watchdog.
# TYPE vllm:watchdog_timeouts_total counter
vllm:watchdog_timeouts_total{engine="0",model_name="m",watchdog="engine_0"} 1.0
vllm:watchdog_timeouts_total{engine="0",model_name="m",watchdog="worker_0"} 0.0
vllm:watchdog_timeouts_created{engine="0",model_name="m",watchdog="worker_0"} 1.7e9
vllm:watchdog_recoveries_total{engine="0",model_name="m",watchdog="worker_0"} 5.0
vllm:num_requests_running{engine="0",model_name="m"} 0.0
"""


def good_episode(names=("engine_0",), witness=2.0, during=0.0, post=1.0) -> dict:
    """A clean hold: series present at zero, witness at 2 s, export after release."""
    present = dict.fromkeys(names, 0.0)
    return {
        "held_rank": 0,
        "hold_entered": True,
        "request_ok": True,
        "released": True,
        "post_release_requests_ok": True,
        "witness_s": dict.fromkeys(names, witness),
        "baseline": dict(present),
        "samples": [[float(t), dict.fromkeys(names, during)] for t in range(10)],
        "post_release": dict.fromkeys(names, post),
    }


def outcome(result: dict) -> tuple[str, str]:
    return result["outcome"], result["reason"]


class ParseTests(unittest.TestCase):
    def test_counts_timeout_totals_and_keeps_zero_series(self) -> None:
        self.assertEqual(
            probe.parse_timeouts(EXPOSITION), {"engine_0": 1.0, "worker_0": 0.0}
        )

    def test_absent_series_is_absent_not_zero(self) -> None:
        self.assertNotIn("engine_0", probe.parse_timeouts("vllm:other 1.0\n"))


class ControlTests(unittest.TestCase):
    def test_required_series_must_exist_after_warmup(self) -> None:
        result = probe.evaluate_control({}, {}, True, False, ["engine_0"])
        self.assertEqual(result["missing_series"], ["engine_0"])
        self.assertFalse(result["ok"])

    def test_change_while_idle_fails(self) -> None:
        warm, idle = {"engine_0": 0.0}, {"engine_0": 1.0}
        self.assertFalse(
            probe.evaluate_control(warm, idle, True, False, ["engine_0"])["ok"]
        )

    def test_clean_control_passes(self) -> None:
        warm = {"engine_0": 0.0}
        self.assertTrue(
            probe.evaluate_control(warm, dict(warm), True, False, ["engine_0"])["ok"]
        )


class ContinuingTests(unittest.TestCase):
    def test_clean_hold_is_supported(self) -> None:
        self.assertEqual(
            outcome(probe.score_continuing(good_episode(), "engine_0")),
            ("supported", "counter_at_baseline_during_hold"),
        )

    def test_rise_during_hold_is_refuted(self) -> None:
        episode = good_episode()
        episode["samples"][5][1]["engine_0"] = 1.0
        self.assertEqual(
            probe.score_continuing(episode, "engine_0")["outcome"], "refuted"
        )

    def test_mutations_are_unscored(self) -> None:
        cases = {
            "hold_not_entered": lambda e: e.update(hold_entered=False),
            "held_request_failed": lambda e: e.update(request_ok=False),
            "release_not_observed": lambda e: e.update(released=False),
            "post_release_requests_failed": lambda e: e.update(
                post_release_requests_ok=False
            ),
            "baseline_scrape_failed": lambda e: e.update(baseline=None),
            "post_release_scrape_failed": lambda e: e.update(post_release=None),
            "no_witness": lambda e: e["witness_s"].update(engine_0=None),
            "series_absent_at_baseline": lambda e: e["baseline"].pop("engine_0"),
            "export_control_failed": lambda e: e["post_release"].update(engine_0=0.0),
        }
        for reason, mutate in cases.items():
            with self.subTest(reason=reason):
                episode = copy.deepcopy(good_episode())
                mutate(episode)
                self.assertEqual(
                    outcome(probe.score_continuing(episode, "engine_0")),
                    ("unscored", reason),
                )

    def test_one_absent_series_during_hold_is_unscored(self) -> None:
        episode = good_episode()
        episode["samples"][6][1] = {}
        self.assertEqual(
            outcome(probe.score_continuing(episode, "engine_0")),
            ("unscored", "series_absent_during_hold"),
        )

    def test_one_failed_post_witness_scrape_is_unscored(self) -> None:
        episode = good_episode()
        episode["samples"][6][1] = None
        self.assertEqual(
            outcome(probe.score_continuing(episode, "engine_0")),
            ("unscored", "scrape_failed_during_hold"),
        )

    def test_failed_scrape_before_witness_is_not_scored(self) -> None:
        episode = good_episode()
        episode["samples"][0][1] = None
        self.assertEqual(
            probe.score_continuing(episode, "engine_0")["outcome"], "supported"
        )


class RankTests(unittest.TestCase):
    def rank_episode(self) -> dict:
        episode = good_episode(names=("worker_1",), post=0.0)
        episode["baseline"] = {}
        episode["samples"] = [[float(t), {}] for t in range(10)]
        episode["post_release"] = {}
        return episode

    def test_absent_non_output_series_is_supported_with_export_control(self) -> None:
        self.assertEqual(
            outcome(probe.score_rank(self.rank_episode(), "worker_1", True)),
            ("supported", "non_output_rank_series_absent"),
        )

    def test_export_control_required(self) -> None:
        self.assertEqual(
            outcome(probe.score_rank(self.rank_episode(), "worker_1", False)),
            ("unscored", "export_control_failed"),
        )

    def test_present_series_without_rise_is_unscored(self) -> None:
        episode = self.rank_episode()
        episode["post_release"] = {"worker_1": 0.0}
        self.assertEqual(
            outcome(probe.score_rank(episode, "worker_1", True)),
            ("unscored", "series_present_without_rise"),
        )

    def test_rise_refutes(self) -> None:
        episode = self.rank_episode()
        episode["post_release"] = {"worker_1": 1.0}
        self.assertEqual(
            probe.score_rank(episode, "worker_1", True)["outcome"], "refuted"
        )

    def test_failed_scrape_cannot_look_like_absence(self) -> None:
        episode = self.rank_episode()
        episode["samples"][6][1] = None
        self.assertEqual(
            outcome(probe.score_rank(episode, "worker_1", True)),
            ("unscored", "scrape_failed_during_hold"),
        )


class ScoreTests(unittest.TestCase):
    def test_control_failure_overrides_every_outcome(self) -> None:
        outcomes = probe.score(1, [good_episode()], control_ok=False)
        self.assertEqual(
            outcome(outcomes["engine_core_continuing_hold"]),
            ("unscored", "control_failed"),
        )

    def test_tp2_scores_three_questions(self) -> None:
        names = ("engine_0", "worker_0", "worker_1")
        first = good_episode(names=names)
        first["witness_s"]["engine_0"] = None
        second = RankTests().rank_episode()
        outcomes = probe.score(2, [first, second], control_ok=True)
        self.assertEqual(
            {key: value["outcome"] for key, value in outcomes.items()},
            {
                "engine_core_continuing_hold": "unscored",
                "worker_continuing_hold": "supported",
                "non_output_rank_exported": "supported",
            },
        )

    def test_identity_file_list_covers_pr_runtime_files(self) -> None:
        self.assertEqual(len(probe.IDENTITY_FILES), 13)
        self.assertTrue(all(p.startswith("vllm/") for p in probe.IDENTITY_FILES))


class ServerConfigTests(unittest.TestCase):
    def test_cpu_memory_reservation_is_fixed_for_startup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                model="m", port=8055, tp=1, watchdog_timeout=15, check_interval=1
            )
            with mock.patch.object(probe.subprocess, "Popen") as popen:
                server = probe.Server(args, Path(tmp), Path(tmp) / "control")
            try:
                command = popen.call_args.args[0]
                index = command.index("--gpu-memory-utilization")
                self.assertEqual(command[index + 1], "0.5")
                self.assertIn("--enforce-eager", command)
            finally:
                server.log.close()


class FailClosedTests(unittest.TestCase):
    def test_failed_git_status_is_not_a_clean_tree(self) -> None:
        def fake_run(command, **kwargs):
            if "rev-parse" in command:
                return subprocess.CompletedProcess(command, 0, probe.PR_HEAD, "")
            return subprocess.CompletedProcess(command, 128, "", "fatal")

        with mock.patch.object(probe.subprocess, "run", fake_run):
            identity = probe.check_identity(Path("unused"))
        self.assertTrue(identity["source_head_matches_pin"])
        self.assertFalse(identity["source_status_ok"])
        self.assertFalse(identity["source_tracked_clean"])
        self.assertFalse(identity["verified"])

    def test_reused_work_dir_is_refused_and_left_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "work"
            (work / "control").mkdir(parents=True)
            evidence = work / "control" / "earlier.txt"
            evidence.write_text("keep\n")
            done = subprocess.run(
                [
                    sys.executable, str(SCRIPT), "--tp", "1",
                    "--vllm-src", str(Path(tmp) / "missing-src"),
                    "--work-dir", str(work),
                ],
                capture_output=True, text=True, timeout=120,
            )  # fmt: skip
            self.assertEqual(done.returncode, 2)
            self.assertIn("--work-dir must be new or empty", done.stderr)
            self.assertEqual(evidence.read_text(), "keep\n")
            self.assertFalse((work / "result.json").exists())

    def test_unverified_identity_writes_receipt_without_launching(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "work"
            done = subprocess.run(
                [
                    sys.executable, str(SCRIPT), "--tp", "2",
                    "--vllm-src", str(Path(tmp) / "missing-src"),
                    "--work-dir", str(work),
                ],
                capture_output=True, text=True, timeout=120,
            )  # fmt: skip
            self.assertEqual(done.returncode, 2, done.stderr)
            receipt = json.loads((work / "result.json").read_text())
            self.assertFalse(receipt["identity"]["verified"])
            self.assertEqual(
                {v["reason"] for v in receipt["outcomes"].values()},
                {"identity_unverified"},
            )
            self.assertEqual(len(receipt["outcomes"]), 3)
            self.assertFalse((work / "server.log").exists())


if __name__ == "__main__":
    unittest.main()
