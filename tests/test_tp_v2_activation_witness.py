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
    / "v2_activation_witness.py"
)
spec = importlib.util.spec_from_file_location("llr_v2_witness_test", PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def valid_witness() -> dict:
    return {
        "schema": "tp-v2-activation-v1",
        "install_seen": True,
        "process_origin": "inherited_after_fork",
        "manager_instances": 1,
        "manager_kind": "ModelCudaGraphManager",
        "runner_v2": True,
        "tp_rank": 1,
        "tp_world_size": 2,
        "configured_graph_mode": "FULL_AND_PIECEWISE",
        "breakable_enabled": False,
        "replay_calls": 2,
        "armed_replay_calls": 1,
        "full_cached_calls": 2,
        "observed_full_cached_calls": 1,
        "eligible_calls": 1,
        "hold_entered": True,
    }


class TPV2ActivationWitnessTest(unittest.TestCase):
    def _write(self, directory: Path, data: str) -> None:
        path = directory / "witness.42.json"
        path.write_text(data, encoding="ascii")
        path.chmod(0o600)

    def test_closed_valid_witness(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "witness"
            directory.mkdir(mode=0o700)
            expected = valid_witness()
            self._write(directory, json.dumps(expected))
            self.assertEqual(module.read_witness(directory, 42, 1), expected)

    def test_every_identity_and_precondition_is_checked(self) -> None:
        mutations = {
            "install_seen": False,
            "process_origin": "unknown",
            "manager_instances": 2,
            "manager_kind": "CUDAGraphWrapper",
            "runner_v2": False,
            "tp_rank": 0,
            "tp_world_size": 1,
            "configured_graph_mode": "PIECEWISE",
            "breakable_enabled": True,
            "replay_calls": True,
            "armed_replay_calls": 3,
            "full_cached_calls": -1,
            "observed_full_cached_calls": 3,
            "eligible_calls": "1",
        }
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "witness"
            directory.mkdir(mode=0o700)
            for field, value in mutations.items():
                with self.subTest(field=field):
                    self._write(directory, json.dumps(valid_witness() | {field: value}))
                    with self.assertRaises(ValueError):
                        module.read_witness(directory, 42, 1)

    def test_missing_extra_duplicate_and_inconsistent_facts_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "witness"
            directory.mkdir(mode=0o700)
            with self.assertRaises(FileNotFoundError):
                module.read_witness(directory, 42, 1)
            for data in (
                valid_witness() | {"extra": 1},
                {key: value for key, value in valid_witness().items() if key != "schema"},
                valid_witness() | {"eligible_calls": 0},
                valid_witness() | {"armed_replay_calls": 0},
                valid_witness() | {"full_cached_calls": 0},
            ):
                self._write(directory, json.dumps(data))
                with self.assertRaises(ValueError):
                    module.read_witness(directory, 42, 1)
            self._write(directory, json.dumps(valid_witness())[:-1] + ',"tp_rank":1}')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                module.read_witness(directory, 42, 1)

    def test_decode_only_full_graph_mode_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "witness"
            directory.mkdir(mode=0o700)
            expected = valid_witness() | {
                "configured_graph_mode": "FULL_DECODE_ONLY",
                "process_origin": "direct_install",
            }
            self._write(directory, json.dumps(expected))
            self.assertEqual(module.read_witness(directory, 42, 1), expected)


if __name__ == "__main__":
    unittest.main()
