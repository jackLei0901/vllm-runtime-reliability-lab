from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from unittest.mock import patch

PLUGIN = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "vllm-tp-dfx"
    / "serving-v2-stall-plugin"
    / "llr_tp_v2_stall.py"
)


def load_plugin() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("llr_tp_v2_stall_test", PLUGIN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GraphMode(Enum):
    FULL = 1
    PIECEWISE = 2


@dataclass(frozen=True)
class Descriptor:
    cg_mode: GraphMode
    key: int = 0


class TPV2StallPluginTest(unittest.TestCase):
    def test_entry_point_is_discoverable(self) -> None:
        entries = [
            entry
            for distribution in importlib.metadata.distributions(
                path=[str(PLUGIN.parent)]
            )
            for entry in distribution.entry_points
            if entry.group == "vllm.general_plugins"
        ]
        self.assertEqual(
            [(entry.name, entry.value) for entry in entries],
            [("llr_tp_v2_stall", "llr_tp_v2_stall:install")],
        )

    def test_every_eligibility_fact_is_required(self) -> None:
        module = load_plugin()
        facts = {
            "armed": True,
            "held": False,
            "manager_unique": True,
            "runner_v2": True,
            "tp_rank": 1,
            "tp_world_size": 2,
            "mode": "FULL",
            "graph_cached": True,
            "breakable_enabled": False,
        }
        self.assertTrue(module.eligible_v2(**facts))
        mutations = {
            "armed": False,
            "held": True,
            "manager_unique": False,
            "runner_v2": False,
            "tp_rank": 0,
            "tp_world_size": 1,
            "mode": "PIECEWISE",
            "graph_cached": False,
            "breakable_enabled": True,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                self.assertFalse(module.eligible_v2(**(facts | {field: value})))

    def _fixture(self, directory: str, *, rank: int = 1, v2: bool = True,
                 breakable: bool = False) -> tuple[dict, type, Path, Path]:
        root = Path(directory)
        witness_dir = root / "witness"
        witness_dir.mkdir(mode=0o700)
        arm, entered = root / "arm", root / "entered"

        class ModelCudaGraphManager:
            def __init__(self) -> None:
                self.vllm_config = types.SimpleNamespace(use_v2_model_runner=v2)
                self.cudagraph_mode = types.SimpleNamespace(name="FULL_AND_PIECEWISE")
                self.use_breakable_cg = breakable
                self.graphs: dict[Descriptor, object] = {}

            def run_fullgraph(self, desc: Descriptor) -> str:
                return "replayed"

        fake_modules = {
            name: types.ModuleType(name)
            for name in (
                "vllm",
                "vllm.distributed",
                "vllm.v1",
                "vllm.v1.worker",
                "vllm.v1.worker.gpu",
                "vllm.v1.worker.gpu.cudagraph_utils",
            )
        }
        distributed = fake_modules["vllm.distributed"]
        distributed.get_tensor_model_parallel_rank = lambda: rank
        distributed.get_tensor_model_parallel_world_size = lambda: 2
        fake_modules["vllm.v1.worker.gpu.cudagraph_utils"].ModelCudaGraphManager = (
            ModelCudaGraphManager
        )
        environment = {
            "LLR_TP_ARM_FILE": str(arm),
            "LLR_TP_ENTER_FILE": str(entered),
            "LLR_TP_OBSERVE_FILE": str(root / "observe"),
            "LLR_TP_WITNESS_DIR": str(witness_dir),
            "LLR_TP_HOLD_SECONDS": "3",
            "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
        }
        return (
            {"modules": fake_modules, "environment": environment},
            ModelCudaGraphManager, arm, entered,
        )

    def _witness(self, directory: str) -> dict:
        path = Path(directory) / "witness" / f"witness.{os.getpid()}.json"
        return json.loads(path.read_text(encoding="ascii"))

    def test_v2_manager_witness_and_one_shot_cached_full_hold(self) -> None:
        module = load_plugin()
        with tempfile.TemporaryDirectory() as directory:
            fixture, Manager, arm, entered = self._fixture(directory)
            with (
                patch.dict(sys.modules, fixture["modules"]),
                patch.dict(os.environ, fixture["environment"]),
                patch.object(module.time, "sleep") as sleep,
                patch.object(os, "register_at_fork", create=True),
            ):
                module.install()
                module.install()
                self.assertEqual(self._witness(directory)["manager_instances"], 0)
                manager = Manager()
                full = Descriptor(GraphMode.FULL)
                uncached = Descriptor(GraphMode.FULL, 1)
                piecewise = Descriptor(GraphMode.PIECEWISE, 2)
                manager.graphs[full] = object()
                self.assertEqual(manager.run_fullgraph(full), "replayed")
                self.assertFalse(entered.exists())
                self.assertEqual(
                    self._witness(directory)["observed_full_cached_calls"], 0
                )
                (Path(directory) / "observe").touch()
                arm.touch()
                self.assertEqual(manager.run_fullgraph(uncached), "replayed")
                self.assertFalse(entered.exists())
                manager.graphs[piecewise] = object()
                self.assertEqual(manager.run_fullgraph(piecewise), "replayed")
                self.assertFalse(entered.exists())
                self.assertEqual(manager.run_fullgraph(full), "replayed")
                self.assertTrue(entered.is_file())
                sleep.assert_called_once_with(3.0)
                self.assertEqual(manager.run_fullgraph(full), "replayed")
                sleep.assert_called_once()
                witness = self._witness(directory)
                self.assertEqual(witness["manager_kind"], "ModelCudaGraphManager")
                self.assertIs(witness["runner_v2"], True)
                self.assertEqual(witness["tp_rank"], 1)
                self.assertIs(witness["breakable_enabled"], False)
                self.assertEqual(witness["replay_calls"], 2)  # Saturated.
                self.assertEqual(witness["observed_full_cached_calls"], 2)
                self.assertEqual(witness["armed_replay_calls"], 2)
                self.assertEqual(witness["eligible_calls"], 1)
                self.assertIs(witness["hold_entered"], True)

    def test_ineligible_managers_do_not_hold(self) -> None:
        for case in ("wrong_rank", "non_v2", "breakable", "duplicate_manager"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                module = load_plugin()
                fixture, Manager, arm, entered = self._fixture(
                    directory,
                    rank=0 if case == "wrong_rank" else 1,
                    v2=case != "non_v2",
                    breakable=case == "breakable",
                )
                with (
                    patch.dict(sys.modules, fixture["modules"]),
                    patch.dict(os.environ, fixture["environment"]),
                    patch.object(module.time, "sleep") as sleep,
                    patch.object(os, "register_at_fork", create=True),
                ):
                    module.install()
                    manager = Manager()
                    if case == "duplicate_manager":
                        manager = Manager()
                    desc = Descriptor(GraphMode.FULL)
                    manager.graphs[desc] = object()
                    arm.touch()
                    self.assertEqual(manager.run_fullgraph(desc), "replayed")
                    self.assertFalse(entered.exists())
                    sleep.assert_not_called()

    def test_witness_writes_saturate_and_missing_path_fails_before_patch(self) -> None:
        module = load_plugin()
        with tempfile.TemporaryDirectory() as directory:
            fixture, Manager, arm, _ = self._fixture(directory)
            with patch.dict(sys.modules, fixture["modules"]), patch.object(
                os, "register_at_fork", create=True
            ):
                with patch.dict(os.environ, fixture["environment"] | {
                    "LLR_TP_WITNESS_DIR": "relative"
                }):
                    with self.assertRaisesRegex(RuntimeError, "absolute"):
                        module.install()
                    self.assertIsNot(Manager.run_fullgraph.__module__, module.__name__)
                with patch.dict(os.environ, fixture["environment"]):
                    module.install()
                    manager = Manager()
                    desc = Descriptor(GraphMode.FULL)
                    manager.graphs[desc] = object()
                    with patch.object(
                        module, "_write_witness", wraps=module._write_witness
                    ) as write:
                        for _ in range(10):
                            manager.run_fullgraph(desc)
                    self.assertEqual(write.call_count, 2)
                    self.assertFalse(arm.exists())

    @unittest.skipUnless(hasattr(os, "fork"), "requires POSIX fork")
    def test_two_forked_workers_have_distinct_witness_files(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "from tests.test_tp_v2_stall_plugin import _fork_probe; _fork_probe()",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "FORK_WITNESSES_OK")


def _fork_probe() -> None:
    module = load_plugin()
    with tempfile.TemporaryDirectory() as directory:
        fixture, Manager, _arm, _entered = TPV2StallPluginTest()._fixture(directory)
        with patch.dict(sys.modules, fixture["modules"]), patch.dict(
            os.environ, fixture["environment"]
        ):
            module.install()
            parent_pid = os.getpid()
            read_fd, write_fd = os.pipe()
            children = []
            for _ in range(2):
                child_pid = os.fork()
                if child_pid == 0:
                    try:
                        os.close(write_fd)
                        os.read(read_fd, 1)
                        manager = Manager()
                        full = Descriptor(GraphMode.FULL)
                        manager.graphs[full] = object()
                        manager.run_fullgraph(full)
                        os._exit(0)
                    except BaseException:
                        os._exit(1)
                children.append(child_pid)
            os.close(read_fd)
            os.write(write_fd, b"xx")
            os.close(write_fd)
            for child_pid in children:
                _pid, status = os.waitpid(child_pid, 0)
                assert os.waitstatus_to_exitcode(status) == 0
            witness_dir = Path(directory) / "witness"
            assert sorted(path.name for path in witness_dir.iterdir()) == sorted(
                f"witness.{pid}.json" for pid in (parent_pid, *children)
            )
            for child_pid in children:
                witness = json.loads(
                    (witness_dir / f"witness.{child_pid}.json").read_text()
                )
                assert witness["process_origin"] == "inherited_after_fork"
                assert witness["manager_instances"] == 1
                assert witness["tp_rank"] == 1
    print("FORK_WITNESSES_OK")


if __name__ == "__main__":
    unittest.main()
