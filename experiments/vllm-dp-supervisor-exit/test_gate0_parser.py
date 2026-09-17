#!/usr/bin/env python3
"""Focused tests for Gate 0's decorated subject record parser."""

from __future__ import annotations

import unittest

from gate0_dp_supervisor_exit import parse_subject


class ParseSubjectTest(unittest.TestCase):
    def test_accepts_vllm_process_prefix(self) -> None:
        output = (
            "(DPSupervisor pid=123) GATE0_SUBJECT_JSON="
            '{"case":"abnormal-child-exit","child_exitcode":17}'
        )
        self.assertEqual(
            parse_subject(output),
            {"case": "abnormal-child-exit", "child_exitcode": 17},
        )

    def test_rejects_multiple_records(self) -> None:
        output = "\n".join(
            [
                'GATE0_SUBJECT_JSON={"case":"a"}',
                'prefix GATE0_SUBJECT_JSON={"case":"b"}',
            ]
        )
        self.assertIsNone(parse_subject(output))


if __name__ == "__main__":
    unittest.main()
