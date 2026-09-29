from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "pr52365-event-timeout"
    / "runner.py"
)
SPEC = importlib.util.spec_from_file_location("pr52365_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def wait(
    count=1, returned_max_s=2.0, timeout_count=0, valid=True, sources=None
) -> dict:
    return {
        "valid": valid,
        "count": count,
        "returned_max_s": returned_max_s,
        "timeout_count": timeout_count,
        "sources": ["v1"] if sources is None else sources,
    }


def arm(long_ok=True, long_s=80.0, short_s=20.0, timeout_logged=False) -> dict:
    return {
        "healthy": True,
        "short_ok": True,
        "short_s": short_s,
        "long_ok": long_ok,
        "long_s": long_s,
        "timeout_logged": timeout_logged,
        "fatal_logged": not long_ok,
        "short_wait": wait(),
        "long_wait": wait(returned_max_s=80.0),
    }


def supported_arms() -> dict:
    arms = {
        "base": arm(),
        "pr_default": arm(long_ok=False, long_s=61.0, timeout_logged=True),
        "pr_disabled": arm(),
    }
    arms["pr_default"]["long_wait"] = wait(returned_max_s=0.0, timeout_count=1)
    return arms


def outcome(result: dict) -> tuple[str, str]:
    return result["outcome"], result["reason"]


class SelectTests(unittest.TestCase):
    def test_picks_step_after_first_long_screen(self) -> None:
        sweep = [
            {"tokens": 32768, "ok": True, "seconds": 12.0},
            {"tokens": 65536, "ok": True, "seconds": 45.0},
            {"tokens": 98304, "ok": True, "seconds": 62.0},
            {"tokens": 131072, "ok": True, "seconds": 90.0},
            {"tokens": 163840, "ok": True, "seconds": 104.0},
        ]
        self.assertEqual(
            runner.select_lengths(sweep),
            {
                "screen": 131072,
                "long": 163840,
                "short": 65536,
                "max_seconds": 104.0,
                "outcome": "candidate_found",
            },
        )

    def test_failed_or_dead_zone_lengths_are_not_candidates(self) -> None:
        sweep = [
            {"tokens": 65536, "ok": True, "seconds": 55.0},
            {"tokens": 131072, "ok": False, "seconds": None},
        ]
        result = runner.select_lengths(sweep)
        self.assertEqual(result["outcome"], "no_candidate")
        self.assertIsNone(result["long"])

    def test_first_long_at_last_ladder_step_is_no_candidate(self) -> None:
        sweep = [
            {"tokens": 1024, "ok": True, "seconds": 20.0},
            {"tokens": 2048, "ok": True, "seconds": 75.0},
        ]
        result = runner.select_lengths(sweep)
        self.assertEqual(result["screen"], 2048)
        self.assertEqual(result["outcome"], "no_candidate")

    def test_failed_step_after_screen_is_no_candidate(self) -> None:
        sweep = [
            {"tokens": 1024, "ok": True, "seconds": 20.0},
            {"tokens": 2048, "ok": True, "seconds": 75.0},
            {"tokens": 4096, "ok": False, "seconds": None},
        ]
        self.assertEqual(runner.select_lengths(sweep)["outcome"], "no_candidate")


class ScoreTests(unittest.TestCase):
    def test_supported_needs_timeout_attribution(self) -> None:
        self.assertEqual(
            outcome(runner.score_ab(True, supported_arms())),
            ("supported", "default_bound_rejected_completing_request"),
        )

    def test_completing_default_arm_is_not_a_refutation(self) -> None:
        arms = supported_arms()
        arms["pr_default"] = arm()
        self.assertEqual(runner.score_ab(True, arms)["outcome"], "not_reproduced")

    def test_completing_default_without_event_witness_is_unscored(self) -> None:
        arms = supported_arms()
        arms["pr_default"] = arm()
        arms["pr_default"]["long_wait"] = wait(count=0)
        self.assertEqual(
            outcome(runner.score_ab(True, arms)),
            ("unscored", "default_long_wait_unobserved"),
        )

    def test_mutations_are_unscored(self) -> None:
        cases = {
            "identity_unverified": (False, lambda a: None),
            "server_not_healthy": (True, lambda a: a["base"].update(healthy=False)),
            "short_control_failed": (
                True,
                lambda a: a["pr_default"].update(short_ok=False),
            ),
            "short_control_not_short": (
                True,
                lambda a: a["base"].update(short_s=51.0),
            ),
            "base_long_step_not_established": (
                True,
                lambda a: a["base"].update(long_s=65.0),
            ),
            "pr_disabled_long_failed": (
                True,
                lambda a: a["pr_disabled"].update(long_ok=False),
            ),
            "event_trace_invalid": (
                True,
                lambda a: a["pr_disabled"]["long_wait"].update(valid=False),
            ),
            "unexpected_runner_path": (
                True,
                lambda a: a["pr_disabled"]["long_wait"].update(sources=["v2"]),
            ),
            "long_event_wait_not_established": (
                True,
                lambda a: a["pr_disabled"]["long_wait"].update(returned_max_s=10.0),
            ),
            "short_event_control_unverified": (
                True,
                lambda a: a["pr_default"]["short_wait"].update(count=0),
            ),
            "failure_not_attributed_to_event_timeout": (
                True,
                lambda a: a["pr_default"].update(timeout_logged=False),
            ),
        }
        for reason, (identity_ok, mutate) in cases.items():
            with self.subTest(reason=reason):
                arms = copy.deepcopy(supported_arms())
                mutate(arms)
                self.assertEqual(
                    outcome(runner.score_ab(identity_ok, arms)), ("unscored", reason)
                )

    def test_missing_arm_is_unscored(self) -> None:
        arms = supported_arms()
        del arms["pr_disabled"]
        self.assertEqual(
            outcome(runner.score_ab(True, arms)), ("unscored", "server_not_healthy")
        )


class IdentityTests(unittest.TestCase):
    def test_failed_git_status_is_not_clean(self) -> None:
        def fake_run(command, **kwargs):
            if "rev-parse" in command:
                return subprocess.CompletedProcess(command, 0, runner.PR_PIN, "")
            return subprocess.CompletedProcess(command, 128, "", "fatal")

        with mock.patch.object(runner.subprocess, "run", fake_run):
            identity = runner.check_identity("python", Path("unused"), runner.PR_PIN)
        self.assertTrue(identity["head_matches_pin"])
        self.assertFalse(identity["source_tracked_clean"])
        self.assertFalse(identity["verified"])

    def test_base_matches_when_new_file_is_absent_in_both(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            for rel in runner.PR_FILES[:1] + runner.PR_FILES[2:]:
                (src / rel).parent.mkdir(parents=True, exist_ok=True)
                (src / rel).write_text("same\n")

            def fake_run(command, **kwargs):
                if "rev-parse" in command:
                    out = runner.BASE_PIN
                elif "status" in command:
                    out = ""
                elif "sitecustomize" in " ".join(command):
                    out = ""
                else:
                    out = str(src / "vllm")
                return subprocess.CompletedProcess(command, 0, out, "")

            with mock.patch.object(runner.subprocess, "run", fake_run):
                identity = runner.check_identity("python", src, runner.BASE_PIN)
        self.assertEqual(identity["mismatched_files"], [])
        self.assertTrue(identity["verified"])

    def test_untracked_new_event_file_at_base_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            for rel in runner.PR_FILES:
                (src / rel).parent.mkdir(parents=True, exist_ok=True)
                (src / rel).write_text("same\n")

            def fake_run(command, **kwargs):
                if "rev-parse" in command:
                    out = runner.BASE_PIN
                elif "status" in command:
                    out = ""
                elif "sitecustomize" in " ".join(command):
                    out = ""
                else:
                    out = str(src / "vllm")
                return subprocess.CompletedProcess(command, 0, out, "")

            with mock.patch.object(runner.subprocess, "run", fake_run):
                identity = runner.check_identity("python", src, runner.BASE_PIN)
        self.assertFalse(identity["verified"])

    def test_existing_sitecustomize_blocks_identity_before_vllm_import(self) -> None:
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            if "rev-parse" in command:
                out = runner.PR_PIN
            elif "sitecustomize" in " ".join(command):
                out = "/opt/python/sitecustomize.py"
            else:
                out = ""
            return subprocess.CompletedProcess(command, 0, out, "")

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(runner.subprocess, "run", fake_run):
                identity = runner.check_identity("python", Path(tmp), runner.PR_PIN)
        self.assertTrue(identity["existing_sitecustomize"])
        self.assertFalse(identity["verified"])
        self.assertFalse(any("import os, vllm" in " ".join(c) for c in calls))


class FailClosedTests(unittest.TestCase):
    def run_cli(self, *extra: str, env: dict | None = None):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *extra],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )

    def test_unverified_identity_writes_receipt_without_launching(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "work"
            done = self.run_cli(
                "ab", "--model", "m", "--model-revision", "0" * 40,
                "--max-model-len", "1000",
                "--work-dir", str(work),
                "--base-python", sys.executable,
                "--base-src", str(Path(tmp) / "missing-base"),
                "--pr-python", sys.executable,
                "--pr-src", str(Path(tmp) / "missing-pr"),
                "--short", "10", "--long", "100",
            )  # fmt: skip
            self.assertEqual(done.returncode, 2, done.stderr)
            receipt = json.loads((work / "result.json").read_text())
            self.assertEqual(receipt["outcome"]["reason"], "identity_unverified")
            self.assertFalse(list(work.glob("*.log")))

    def test_reused_work_dir_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "work"
            work.mkdir()
            (work / "earlier.txt").write_text("keep\n")
            done = self.run_cli(
                "sweep", "--model", "m", "--model-revision", "0" * 40,
                "--max-model-len", "1000",
                "--work-dir", str(work), "--python", sys.executable,
                "--vllm-src", str(Path(tmp) / "missing"), "--lengths", "10", "20",
            )  # fmt: skip
            self.assertEqual(done.returncode, 2)
            self.assertIn("--work-dir must be new or empty", done.stderr)
            self.assertEqual((work / "earlier.txt").read_text(), "keep\n")


class WitnessTests(unittest.TestCase):
    def test_wait_summary_requires_valid_complete_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            path.write_text(
                '{"kind":"installed"}\n'
                '{"kind":"wait","seconds":70.5,"status":"returned","source":"v1"}\n'
                '{"kind":"wait","seconds":60.0,"status":"timeout","source":"v1"}\n'
            )
            summary, offset = runner.wait_summary(path, 1)
            self.assertEqual(offset, 3)
            self.assertEqual(summary["returned_max_s"], 70.5)
            self.assertEqual(summary["timeout_count"], 1)
            self.assertEqual(summary["sources"], ["v1"])
            path.write_text(path.read_text() + "{broken\n")
            self.assertFalse(runner.wait_summary(path, offset)[0]["valid"])

    def test_import_hook_records_real_function_calls_without_vllm_install(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = root / "vllm" / "v1" / "worker" / "gpu"
            package.mkdir(parents=True)
            for parent in (
                root / "vllm",
                root / "vllm" / "v1",
                package.parent,
                package,
            ):
                (parent / "__init__.py").write_text("")
            (package / "event_utils.py").write_text(
                "def wait_for_gpu_event(event, operation):\n"
                "    if operation == 'timeout':\n"
                "        raise TimeoutError('test')\n"
                "    return event\n"
            )
            trace = root / "trace.jsonl"
            hook_dir = SCRIPT.parent
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join((str(hook_dir), str(root)))
            env["LLR_WAIT_TRACE_PATH"] = str(trace)
            done = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from vllm.v1.worker.gpu.event_utils import wait_for_gpu_event; "
                    "assert wait_for_gpu_event(4, 'ok') == 4; "
                    "\ntry: wait_for_gpu_event(None, 'timeout')\n"
                    "except TimeoutError: pass\n",
                ],
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )
            self.assertEqual(done.returncode, 0, done.stderr)
            lines = [json.loads(line) for line in trace.read_text().splitlines()]
            self.assertEqual(
                [line["kind"] for line in lines], ["installed", "wait", "wait"]
            )
            self.assertEqual(
                [line["status"] for line in lines[1:]], ["returned", "timeout"]
            )


if __name__ == "__main__":
    unittest.main()
