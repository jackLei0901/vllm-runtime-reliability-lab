from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from dfxlab.cli import main as cli_main
from dfxlab.replay import MANIFEST_NAME, ReplayError, main, replay

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "results" / "vllm-zmq-backpressure-stage1-r3-20260916"


class ReplayTests(unittest.TestCase):
    def test_published_case_replays(self) -> None:
        lines = replay(PUBLISHED)
        rendered = "\n".join(lines)
        self.assertIn("/health remained 2xx", rendered)
        self.assertIn("TRADE-OFF: 4 event batches dropped", rendered)
        self.assertIn("does not rerun the GPU experiment", rendered)

    def test_tampered_evidence_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "case"
            shutil.copytree(PUBLISHED, copied)
            path = copied / "cell-3-base-pause.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            record["health_during_stall"] = None
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ReplayError, "evidence hash mismatch"):
                replay(copied)

    def test_manifest_with_extra_field_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "case"
            shutil.copytree(PUBLISHED, copied)
            path = copied / MANIFEST_NAME
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["unexpected"] = True
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ReplayError, "manifest shape changed"):
                replay(copied)

    def test_cli_reports_failure_without_traceback(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            status = main([str(PUBLISHED / "missing")])
        self.assertEqual(status, 1)
        self.assertIn("FAIL:", stderr.getvalue())

    def test_installed_entry_point_exposes_replay(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            status = cli_main(["replay", str(PUBLISHED)])
        self.assertEqual(status, 0)
        self.assertIn("TRADE-OFF: 4 event batches dropped", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
