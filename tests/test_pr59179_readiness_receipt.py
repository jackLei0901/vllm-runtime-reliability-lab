"""CPU-only receipt tests for the PR #59179 exploratory method check."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROBE = Path(__file__).resolve().parents[1] / "experiments/pr59179-readiness/probe.py"
SPEC = importlib.util.spec_from_file_location("pr59179_probe", PROBE)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class ReceiptTest(unittest.TestCase):
    def test_writes_one_normalized_json_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "result.json"
            probe.write_receipt(receipt, {"result": "example", "count": 2})
            self.assertEqual(
                receipt.read_text(encoding="utf-8"),
                '{"count": 2, "result": "example"}\n',
            )
            self.assertEqual(
                json.loads(receipt.read_text()), {"count": 2, "result": "example"}
            )

    def test_refuses_to_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "result.json"
            probe.write_receipt(receipt, {"result": "first"})
            with self.assertRaises(FileExistsError):
                probe.write_receipt(receipt, {"result": "second"})
            self.assertEqual(json.loads(receipt.read_text())["result"], "first")

    def test_existing_receipt_stops_before_source_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "result.json"
            receipt.write_text("prior result\n", encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROBE),
                    "--vllm-src",
                    str(Path(directory) / "missing-source"),
                    "--receipt",
                    str(receipt),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(json.loads(completed.stdout)["reason"], "FileExistsError")
            self.assertEqual(receipt.read_text(encoding="utf-8"), "prior result\n")


if __name__ == "__main__":
    unittest.main()
