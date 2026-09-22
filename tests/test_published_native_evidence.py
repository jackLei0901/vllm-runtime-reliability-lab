from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from dfxlab.native_evidence import validate_capture

ROOT = Path(__file__).resolve().parents[1]
RECORD = (
    ROOT / "results" / "native-stack-pair-stage-a-cpu-20260922" / "capture-record.json"
)
RECORD_SHA256 = "89e0f9b4f1f19cf9fc0cf662ccf604b4f5485ce7f8e6b41ba2a0b7e1aaf4a0a1"


class PublishedNativeEvidenceTest(unittest.TestCase):
    def test_stage_a_cpu_record_is_closed_and_claim_free(self) -> None:
        raw = RECORD.read_bytes()
        self.assertEqual(RECORD_SHA256, hashlib.sha256(raw).hexdigest())
        record = json.loads(raw)
        self.assertEqual(
            {
                "captures",
                "claim",
                "pairing_status",
                "schema_version",
                "sequence",
                "target_runtime",
            },
            set(record),
        )
        self.assertEqual("native-producer-pair-capture-v0", record["schema_version"])
        self.assertEqual("normalization_pending", record["pairing_status"])
        self.assertIsNone(record["claim"])
        self.assertEqual(["py-spy-a", "pystack-b", "py-spy-a2"], record["sequence"])

        captures = {item["label"]: item["capture"] for item in record["captures"]}
        self.assertEqual(set(record["sequence"]), set(captures))
        for capture in captures.values():
            validate_capture(capture)
            self.assertEqual("execution", capture["outcome"]["attempt_stage"])
            self.assertEqual("produced", capture["outcome"]["outcome_code"])
            self.assertEqual(
                capture["subject"]["process_identity_value"],
                capture["subject"]["post_capture_identity_value"],
            )
        self.assertEqual(
            captures["py-spy-a"]["outcome"]["raw_output_sha256"],
            captures["py-spy-a2"]["outcome"]["raw_output_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
