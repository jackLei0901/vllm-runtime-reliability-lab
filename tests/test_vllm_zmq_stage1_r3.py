from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "vllm-zmq-event-backpressure"
RESULTS = ROOT / "results" / "vllm-zmq-backpressure-stage1-r3-20260916"


def load_verifier():
    name = "stage1_r3_verifier_test"
    path = EXPERIMENT / "verify_stage1_r3_results.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Stage1R3VerifierTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.verifier = load_verifier()

    def records(self):
        return [
            json.loads(
                (RESULTS / self.verifier.FILES[index]).read_text(encoding="utf-8")
            )
            for index in self.verifier.CELLS
        ]

    def test_published_four_cell_result_verifies(self) -> None:
        records = self.records()
        for index, record in zip(self.verifier.CELLS, records, strict=True):
            self.verifier.verify_cell(record, index)
        self.verifier.verify_cross_cell(records)

    def test_cross_cell_verifier_rejects_runtime_drift(self) -> None:
        records = self.records()
        records[3]["environment"]["torch"] = "unexpected"
        with self.assertRaisesRegex(AssertionError, "runtime environment"):
            self.verifier.verify_cross_cell(records)

    def test_verifier_rejects_unpinned_source_hash(self) -> None:
        record = deepcopy(self.records()[1])
        record["environment"]["kv_events_sha256"] = "f" * 64
        record["environment"]["tree_kv_events_sha256"] = "f" * 64
        record["engine_core_kv_events_sha256"] = "f" * 64
        with self.assertRaisesRegex(AssertionError, "source-specific kv_events"):
            self.verifier.verify_cell(record, 2)


if __name__ == "__main__":
    unittest.main()
