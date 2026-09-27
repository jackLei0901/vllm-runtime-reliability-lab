from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1] / "experiments" / "vllm-tp-dfx"
RUNNER = ROOT / "serving_v2_stall_gate.py"


def load_runner() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("serving_v2_stall_gate_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(ROOT))
    return module


class TPV2StallGateTest(unittest.TestCase):
    def test_environment_requires_exact_private_paths_and_v2_default(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "inspector").mkdir()
            (root / "witness").mkdir()
            library = root / "inspector.so"
            library.write_bytes(b"test library")
            environment = {
                "LLR_TP_ARM_FILE": str(root / "arm"),
                "LLR_TP_ENTER_FILE": str(root / "entered"),
                "LLR_TP_OBSERVE_FILE": str(root / "observe"),
                "LLR_TP_WITNESS_DIR": str(root / "witness"),
                "LLR_TP_HOLD_SECONDS": "3",
                "VLLM_PLUGINS": "llr_tp_v2_stall",
                "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
                "NCCL_DEBUG": "TRACE",
                "NCCL_DEBUG_SUBSYS": "INIT,PROFILE",
                "NCCL_DEBUG_FILE": str(root / "nccl.%p.log"),
                "NCCL_INSPECTOR_DUMP_DIR": str(root / "inspector"),
                "NCCL_INSPECTOR_ENABLE": "1",
                "NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS": "500",
                "NCCL_INSPECTOR_DUMP_VERBOSE": "1",
                "NCCL_PROFILER_PLUGIN": str(library),
            }
            with patch.object(runner, "_private_directory"), patch.dict(
                os.environ, environment, clear=True
            ):
                self.assertEqual(
                    runner._check_environment(root),
                    (root / "arm", root / "entered", root / "observe", root / "witness"),
                )
                for name, value in (
                    ("VLLM_PLUGINS", "wrong_plugin"),
                    ("VLLM_USE_V2_MODEL_RUNNER", "0"),
                    ("NCCL_DEBUG_SUBSYS", "INIT"),
                    ("NCCL_INSPECTOR_ENABLE", "0"),
                    ("LLR_TP_HOLD_SECONDS", "10"),
                ):
                    with self.subTest(name=name), patch.dict(os.environ, {name: value}):
                        with self.assertRaises(ValueError):
                            runner._check_environment(root)
                (root / "witness" / "stale.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "start empty"):
                    runner._check_environment(root)

    def test_public_witness_drops_identity_and_unlisted_fields(self) -> None:
        runner = load_runner()
        facts = {
            "manager_kind": "ModelCudaGraphManager",
            "process_origin": "inherited_after_fork",
            "runner_v2": True,
            "configured_graph_mode": "FULL",
            "breakable_enabled": False,
            "replay_calls": 2,
            "armed_replay_calls": 1,
            "full_cached_calls": 2,
            "observed_full_cached_calls": 1,
            "eligible_calls": 1,
            "hold_entered": True,
            "tp_rank": 1,
            "pid": 12345,
            "start_ticks": 999,
            "raw_path": "/private/secret",
        }
        result = runner._public_witness({0: facts, 1: facts})
        for rank in (0, 1):
            self.assertEqual(result[rank]["manager_kind"], "ModelCudaGraphManager")
            self.assertNotIn("tp_rank", result[rank])
            self.assertNotIn("pid", result[rank])
            self.assertNotIn("start_ticks", result[rank])
            self.assertNotIn("raw_path", result[rank])

    def test_rank_pid_binding_is_used_for_each_witness(self) -> None:
        runner = load_runner()
        with patch.object(runner, "read_witness", side_effect=lambda _d, pid, rank: (pid, rank)) as read:
            pair = runner._witness_pair(Path("private"), {0: (101, 7), 1: (202, 8)})
        self.assertEqual(pair, {0: (101, 0), 1: (202, 1)})
        self.assertEqual(read.call_count, 2)

    def test_warmup_replays_cannot_pass_the_control(self) -> None:
        runner = load_runner()
        facts = {
            "replay_calls": 2,
            "full_cached_calls": 2,
            "observed_full_cached_calls": 1,
            "eligible_calls": 0,
            "hold_entered": False,
        }
        pair = {0: dict(facts), 1: dict(facts)}
        self.assertTrue(
            runner._control_passed(pair, tokens=16, identity_stable=True, entered=False)
        )
        pair[1]["observed_full_cached_calls"] = 0
        self.assertFalse(
            runner._control_passed(pair, tokens=16, identity_stable=True, entered=False)
        )


if __name__ == "__main__":
    unittest.main()
