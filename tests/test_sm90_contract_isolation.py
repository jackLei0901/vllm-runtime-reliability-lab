import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PACKET = Path(__file__).resolve().parents[1] / (
    "experiments/kernel-operand-contracts/sm90-contract-repair"
)
spec = importlib.util.spec_from_file_location(
    "sm90_isolated", PACKET / "run_isolated.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class TestSM90Isolation(unittest.TestCase):
    def test_relocated_binary_uses_receipt_directory_and_checks_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ["test_repair_sm90.py", "run_isolated.py", "pytest.ini"]
            for name in names:
                (root / name).write_text("fixture")
            binary = root / "library.so"
            binary.write_bytes(b"compiled-fixture")
            receipt = {
                "status": "built_not_validated",
                "source": {
                    "arm": "base",
                    "head": "7b054aca96cea8be1369d651c3434ad140580b92",
                },
                "harness_sha256": {name: runner.sha256(root / name) for name in names},
                "binary": {
                    "binary": "/unavailable/build-host/library.so",
                    "binary_file": "library.so",
                    "binary_sha256": runner.sha256(binary),
                },
            }
            path = root / "build_receipt.json"
            path.write_text(json.dumps(receipt))
            with patch.object(runner, "PACKET", root):
                self.assertEqual(runner.verify_receipt(path, "base"), receipt)
                binary.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "binary changed"):
                    runner.verify_receipt(path, "base")

    def test_runner_pins_working_directory_and_root(self):
        command = runner.pytest_command()
        self.assertIn(f"--rootdir={PACKET.resolve()}", command)
        self.assertEqual(command[command.index("-c") + 1], str(PACKET / "pytest.ini"))
        process = MagicMock()
        process.wait.return_value = 0
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(runner.subprocess, "Popen", return_value=process) as popen,
        ):
            runner.run_once(command, {}, Path(tmp) / "log", 10)
            self.assertEqual(popen.call_args.kwargs["cwd"], PACKET)

    @unittest.skipUnless(importlib.util.find_spec("pytest"), "pytest not installed")
    def test_collected_id_resolves_from_unrelated_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packet = root / "packet"
            packet.mkdir()
            (packet / "pytest.ini").write_text("[pytest]\n")
            test = packet / "test_repair_sm90.py"
            test.write_text("def test_one():\n    pass\n")
            env = dict(
                os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTEST_ADDOPTS=""
            )
            env.pop("PYTEST_PLUGINS", None)
            command = runner.pytest_command(packet)
            log = root / "collection.log"
            self.assertEqual(
                runner.run_once(
                    command + ["--collect-only", str(test)], env, log, 30, packet
                ),
                0,
            )
            nodes = [
                line.strip()
                for line in log.read_text().splitlines()
                if "test_repair_sm90.py::" in line
            ]
            self.assertEqual(nodes, ["test_repair_sm90.py::test_one"])
            log2 = root / "one.log"
            self.assertEqual(
                runner.run_once(
                    command + ["--collect-only", nodes[0]], env, log2, 30, packet
                ),
                0,
            )
            self.assertIn("1 test collected", log2.read_text())

    def test_full_unique_collection(self):
        nodes = [f"test_repair_sm90.py::test_case[{i}]" for i in range(30)]
        self.assertEqual(runner.parse_collection("\n".join(nodes)), nodes)

    def test_duplicate_or_missing_case_fails(self):
        nodes = [f"test_repair_sm90.py::test_case[{i}]" for i in range(30)]
        for text in ("\n".join(nodes[:-1]), "\n".join(nodes[:-1] + [nodes[0]])):
            with self.assertRaises(ValueError):
                runner.parse_collection(text)

    def classify(self, body, code=0, node="test_repair_sm90.py::test_valid", arm="fix"):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "case.xml"
            path.write_text(f"<testsuite>{body}</testsuite>")
            return runner.classify_xml(path, code, node, arm)

    def test_pass_requires_context_witness(self):
        self.assertEqual(self.classify("<testcase/>"), "unscored")
        body = (
            '<testcase><properties><property name="cuda_context_usable_after" '
            'value="True"/></properties></testcase>'
        )
        self.assertEqual(self.classify(body), "prediction_matched")
        self.assertEqual(self.classify(body, code=1), "unscored")

    def test_failure_and_setup_error_are_distinct(self):
        props = (
            '<properties><property name="cuda_context_usable_after" '
            'value="True"/></properties>'
        )
        self.assertEqual(
            self.classify(f"<testcase>{props}<failure/></testcase>", 1),
            "prediction_missed",
        )
        self.assertEqual(self.classify("<testcase><error/></testcase>", 1), "unscored")

    def test_context_loss_is_not_scored_as_missed_prediction(self):
        body = (
            '<testcase><properties><property name="cuda_context_usable_after" '
            'value="False"/></properties><failure/></testcase>'
        )
        self.assertEqual(self.classify(body, 1), "unscored")

    def test_only_exact_base_capture_skip_is_planned(self):
        props = (
            '<properties><property name="cuda_context_usable_after" '
            'value="True"/></properties>'
        )
        node = "test_repair_sm90.py::test_fix_rejects_during_capture[dtype]"
        reason = "base unsafe inputs are measured eagerly, not captured"
        body = f'<testcase>{props}<skipped message="{reason}"/></testcase>'
        self.assertEqual(self.classify(body, node=node, arm="base"), "planned_skip")
        self.assertEqual(self.classify(body, node=node, arm="fix"), "unscored")
        other = f'<testcase>{props}<skipped message="context unavailable"/></testcase>'
        self.assertEqual(self.classify(other, node=node, arm="base"), "unscored")


if __name__ == "__main__":
    unittest.main()
