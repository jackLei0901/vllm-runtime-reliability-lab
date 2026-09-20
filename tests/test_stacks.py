from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from dfxlab.stacks import capture_stack, disabled_stack


class StackProducerTest(unittest.TestCase):
    def test_disabled_shape_is_explicit(self) -> None:
        self.assertEqual({"state": "disabled", "producer": None}, disabled_stack())

    @patch("dfxlab.stacks.shutil.which", return_value=None)
    def test_missing_sampler_is_visible(self, _which) -> None:
        result = capture_stack(123, Path("unused"))
        self.assertEqual("unavailable", result["state"])
        self.assertEqual("binary_missing", result["producer"]["error_kind"])
        self.assertFalse(result["producer"]["output_produced"])

    def test_success_records_identity_without_public_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "py-spy"
            binary.write_bytes(b"sampler")
            output = root / "private" / "stacks.txt"

            def run(command, **_kwargs):
                if command[1] == "--version":
                    return CompletedProcess(command, 0, "py-spy 0.4.1\n", "")
                output.write_text("private stack", encoding="utf-8")
                return CompletedProcess(command, 0, b"", b"")

            with (
                patch("dfxlab.stacks.shutil.which", return_value=str(binary)),
                patch("dfxlab.stacks.subprocess.run", side_effect=run),
            ):
                result = capture_stack(123, output)
            self.assertEqual("produced", result["state"])
            producer = result["producer"]
            self.assertEqual("0.4.1", producer["sampler_version"])
            self.assertEqual(
                hashlib.sha256(b"sampler").hexdigest(), producer["binary_sha256"]
            )
            self.assertEqual(
                hashlib.sha256(b"private stack").hexdigest(),
                producer["raw_output_sha256"],
            )
            self.assertNotIn(str(binary), str(result))
            self.assertNotIn(str(output), str(result))


if __name__ == "__main__":
    unittest.main()
