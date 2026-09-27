from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "vllm-tp-dfx"
    / "legacy_callback_aggregate.py"
)
spec = importlib.util.spec_from_file_location("llr_tp_legacy_aggregate_test", PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class LegacyCallbackAggregateTest(unittest.TestCase):
    def test_closed_aggregate_and_conflict_run_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for rank in (0, 1):
                lines = [f"PROFILER/Plugin: init nranks: 2 rank: {rank}\n"]
                lines.extend(
                    "LLR_TP_EVT coll_start comm=0123456789abcdef "
                    f"seq=7 channels={count}\n"
                    for count in (1, 1, 2, 2)
                )
                (Path(directory) / f"nccl.{rank}.log").write_text(
                    "".join(lines), encoding="utf-8"
                )
            result = module.summarize(str(Path(directory) / "nccl.*.log"))
            self.assertEqual(result[0]["conflict_sizes"], [4])
            self.assertEqual(result[0]["largest_conflict_run_lengths"], [2, 2])
            self.assertNotIn("0123456789abcdef", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
