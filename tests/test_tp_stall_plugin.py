from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

PLUGIN = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "vllm-tp-dfx"
    / "serving-stall-plugin"
    / "llr_tp_stall.py"
)
spec = importlib.util.spec_from_file_location("llr_tp_stall_test", PLUGIN)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TPStallPreconditionTest(unittest.TestCase):
    def test_general_plugin_entry_point_is_discoverable(self) -> None:
        distributions = importlib.metadata.distributions(path=[str(PLUGIN.parent)])
        entries = [
            entry
            for distribution in distributions
            for entry in distribution.entry_points
            if entry.group == "vllm.general_plugins"
        ]
        self.assertEqual(
            [(entry.name, entry.value) for entry in entries],
            [("llr_tp_stall", "llr_tp_stall:install")],
        )

    def test_all_facts_required(self) -> None:
        facts = {
            "armed": True,
            "held": False,
            "tp_rank": 1,
            "tp_world_size": 2,
            "wrapper_mode": "FULL",
            "runtime_mode": "FULL",
            "graph_cached": True,
        }
        self.assertTrue(module.eligible(**facts))
        changed = {
            "armed": False,
            "held": True,
            "tp_rank": 0,
            "tp_world_size": 1,
            "wrapper_mode": "PIECEWISE",
            "runtime_mode": "PIECEWISE",
            "graph_cached": False,
        }
        for field, value in changed.items():
            with self.subTest(field=field):
                self.assertFalse(module.eligible(**(facts | {field: value})))

    def test_entry_point_holds_once_only_after_cached_full_replay(self) -> None:
        class Wrapper:
            runtime_mode = types.SimpleNamespace(name="FULL")

            def __init__(self) -> None:
                self.concrete_cudagraph_entries = {
                    "batch": types.SimpleNamespace(cudagraph=object())
                }

            def __call__(self) -> str:
                return "returned"

        state = {"rank": 1, "runtime": "FULL"}
        context = types.SimpleNamespace(
            batch_descriptor="batch",
            cudagraph_runtime_mode=types.SimpleNamespace(name="FULL"),
        )
        fake_modules = {
            name: types.ModuleType(name)
            for name in (
                "vllm",
                "vllm.compilation",
                "vllm.compilation.cuda_graph",
                "vllm.distributed",
                "vllm.forward_context",
            )
        }
        fake_modules["vllm.compilation.cuda_graph"].CUDAGraphWrapper = Wrapper
        distributed = fake_modules["vllm.distributed"]
        distributed.get_tensor_model_parallel_rank = lambda: state["rank"]
        distributed.get_tensor_model_parallel_world_size = lambda: 2
        forward = fake_modules["vllm.forward_context"]
        forward.get_forward_context = lambda: context
        forward.is_forward_context_available = lambda: True

        with tempfile.TemporaryDirectory() as directory:
            arm = Path(directory) / "arm"
            entered = Path(directory) / "entered"
            environment = {
                "LLR_TP_ARM_FILE": str(arm),
                "LLR_TP_ENTER_FILE": str(entered),
                "LLR_TP_HOLD_SECONDS": "3",
            }
            module._installed = False
            module._held = False
            with (
                patch.dict(sys.modules, fake_modules),
                patch.dict(os.environ, environment),
                patch.object(module.time, "sleep") as sleep,
            ):
                module.install()
                wrapper = Wrapper()
                self.assertEqual(wrapper(), "returned")  # unarmed
                arm.touch()
                state["rank"] = 0
                self.assertEqual(wrapper(), "returned")  # wrong rank
                state["rank"] = 1
                context.cudagraph_runtime_mode.name = "PIECEWISE"
                self.assertEqual(wrapper(), "returned")  # wrong mode
                context.cudagraph_runtime_mode.name = "FULL"
                wrapper.concrete_cudagraph_entries["batch"].cudagraph = None
                self.assertEqual(wrapper(), "returned")  # uncached
                wrapper.concrete_cudagraph_entries["batch"].cudagraph = object()
                self.assertEqual(wrapper(), "returned")
                self.assertTrue(entered.is_file())
                sleep.assert_called_once_with(3.0)
                self.assertEqual(wrapper(), "returned")
                sleep.assert_called_once()


if __name__ == "__main__":
    unittest.main()
