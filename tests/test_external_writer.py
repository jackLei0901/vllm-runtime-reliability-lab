import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_external_schema import artifact

from dfxlab.external_schema import WriterErrorKind
from dfxlab.external_writer import IncidentWriter


class ExternalWriterTest(unittest.TestCase):
    def test_initialization_removes_stale_temporary_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            stale = directory / ".incident-stale.tmp"
            stale.write_text("incomplete", encoding="utf-8")
            IncidentWriter(directory)
            self.assertFalse(stale.exists())

    def test_writes_and_rotates_to_four_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            writer = IncidentWriter(directory, max_artifacts=4)
            for index in range(5):
                item = artifact()
                object.__setattr__(item, "incident_id", f"hmac-sha256:{index:024x}")
                self.assertTrue(writer.try_write(item).ok)
            self.assertEqual(len(list(directory.glob("incident-*.json"))), 4)

    def test_size_limit_fails_open(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = IncidentWriter(Path(tmp), max_artifact_bytes=1)
            outcome = writer.try_write(artifact())
            self.assertFalse(outcome.ok)
            self.assertEqual(outcome.error_kind, WriterErrorKind.SIZE_LIMIT)

    def test_io_failure_is_reported_not_raised(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = IncidentWriter(Path(tmp))
            with patch("dfxlab.external_writer.os.replace", side_effect=OSError):
                outcome = writer.try_write(artifact())
            self.assertFalse(outcome.ok)
            self.assertEqual(outcome.error_kind, WriterErrorKind.IO)
            self.assertEqual(writer.health().artifacts_failed_total, 1)

    @unittest.skipIf(os.name == "nt", "POSIX mode is not meaningful on Windows")
    def test_posix_artifact_mode_is_0600(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outcome = IncidentWriter(Path(tmp)).try_write(artifact())
            self.assertTrue(outcome.ok)
            self.assertEqual(outcome.path.stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
