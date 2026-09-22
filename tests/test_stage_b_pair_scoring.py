from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "experiments"
    / "native-evidence-capability"
    / "score_stage_b_pair.py"
)
SPEC = importlib.util.spec_from_file_location("stage_b_pair_scoring", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
scoring = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scoring)


class StageBPairScoringTest(unittest.TestCase):
    def test_normalizes_py_spy_leaf_first_stack(self) -> None:
        raw = """Thread 42 (idle): \"MainThread\"
    PyThread_acquire_lock_timed (python3.12)
    put (queue.py:140)
    publish (vllm/distributed/kv_events.py:391)
"""
        self.assertEqual(
            list(scoring.FRAME_CLASSES),
            scoring.normalize_stack(raw)["ordered_frame_classes"],
        )

    def test_normalizes_pystack_root_first_stack(self) -> None:
        raw = """Traceback for thread 42 (worker) [] (most recent call last):
    (Python) File "vllm/distributed/kv_events.py", line 391, in publish
    (Python) File "queue.py", line 140, in put
    (C) File "thread_pthread.h", line 490, in PyThread_acquire_lock_timed
"""
        self.assertEqual(
            list(scoring.FRAME_CLASSES),
            scoring.normalize_stack(raw)["ordered_frame_classes"],
        )

    def test_healthy_queue_get_does_not_match(self) -> None:
        raw = """Thread 42 (idle): \"MainThread\"
    PyThread_acquire_lock_timed (python3.12)
    get (queue.py:171)
    run_busy_loop (vllm/v1/engine/core.py:1407)
"""
        self.assertEqual([], scoring.normalize_stack(raw)["ordered_frame_classes"])

    def test_multiple_matching_threads_fail_closed(self) -> None:
        block = """Thread {tid} (idle):
    PyThread_acquire_lock_timed (python3.12)
    put (queue.py:140)
    publish (vllm/distributed/kv_events.py:391)
"""
        with self.assertRaisesRegex(ValueError, "multiple threads"):
            scoring.normalize_stack(block.format(tid=1) + block.format(tid=2))


if __name__ == "__main__":
    unittest.main()
