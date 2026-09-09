from __future__ import annotations

import importlib.util
import json
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = ROOT / "experiments" / "overhead" / "paired_overhead.py"
SPEC = importlib.util.spec_from_file_location("paired_overhead", HARNESS_PATH)
assert SPEC is not None and SPEC.loader is not None
paired_overhead = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = paired_overhead
SPEC.loader.exec_module(paired_overhead)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class PairedOverheadPlanTest(unittest.TestCase):
    def test_reports_pairwise_relative_deltas(self) -> None:
        trials = [
            {
                "arm": arm,
                "valid": True,
                "workload": {
                    "workload_signature": "same",
                    "metrics": {"latency_ms": value},
                },
            }
            for arm, value in (
                ("disabled", 100),
                ("enabled", 102),
                ("disabled", 200),
                ("enabled", 198),
            )
        ]
        comparison = paired_overhead.compare_paired_metrics(trials)["latency_ms"]
        self.assertEqual(comparison["pair_count"], 2)
        self.assertEqual(comparison["relative_delta_pct_by_pair"], [2.0, -1.0])
        self.assertEqual(comparison["median_relative_delta_pct"], 0.5)

    def test_rejects_unpaired_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan.json"
            plan = json.loads(
                (ROOT / "experiments/overhead/config.cpu-example.json").read_text()
            )
            plan["sequence"] = ["enabled", "disabled", "enabled", "disabled"]
            path.write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaisesRegex(paired_overhead.PlanError, "alternate"):
                paired_overhead.load_plan(path)

    def test_detects_workload_signature_mismatch(self) -> None:
        trials = [
            {
                "arm": "disabled",
                "valid": True,
                "workload": {"workload_signature": "a", "metrics": {}},
            },
            {
                "arm": "enabled",
                "valid": True,
                "workload": {"workload_signature": "b", "metrics": {}},
            },
        ]
        self.assertIn(
            "workload_signatures_do_not_match",
            paired_overhead.validate_pairing(trials),
        )

    def test_cpu_fake_server_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            port = free_port()
            plan = json.loads(
                (ROOT / "experiments/overhead/config.cpu-example.json").read_text()
            )
            plan["variables"]["repo_root"] = str(ROOT)
            plan["variables"]["base_url"] = f"http://127.0.0.1:{port}"
            plan["variables"]["port"] = str(port)
            plan_path = tmp_path / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            output = tmp_path / "result"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(HARNESS_PATH),
                    "--plan",
                    str(plan_path),
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads((output / "paired-overhead.private.json").read_text())
            self.assertTrue(result["pairing_valid"])
            self.assertEqual(len(result["trials"]), 4)
            self.assertTrue(all(trial["valid"] for trial in result["trials"]))
            self.assertTrue(
                all(trial["server_running_before_stop"] for trial in result["trials"])
            )
            self.assertTrue(
                all(
                    trial.get("recorder_running_before_stop", True)
                    for trial in result["trials"]
                )
            )
            self.assertEqual(
                [trial["arm"] for trial in result["trials"]], plan["sequence"]
            )
            self.assertTrue((output / "REPORT.md").is_file())


if __name__ == "__main__":
    unittest.main()
