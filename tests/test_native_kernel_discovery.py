import importlib.util
import unittest
from pathlib import Path

PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments/native-kernel-discovery/collect.py"
)
SPEC = importlib.util.spec_from_file_location("discovery", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestDiscovery(unittest.TestCase):
    def test_default_command(self):
        args = MODULE.command("python", Path("model"), Path("out"))
        self.assertIn("--cuda-graph-trace=node", args)
        self.assertNotIn("--enforce-eager", args)
        self.assertNotIn("--distributed-executor-backend", args)
        self.assertEqual(args[args.index("--max-num-seqs") + 1], "32")

    def test_hidden_overrides_refused(self):
        for key in (
            "VLLM_DISABLED_KERNELS",
            "VLLM_USE_DEEP_GEMM",
            "VLLM_USE_V2_MODEL_RUNNER",
            "VLLM_BATCH_INVARIANT",
            "VLLM_ATTENTION_BACKEND",
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                MODULE.launch_env({key: "1"})

    def test_environment_preserved(self):
        original = {"PATH": "unchanged"}
        self.assertEqual(MODULE.launch_env(original)["PATH"], "unchanged")
        self.assertNotIn("VLLM_SERVER_DEV_MODE", original)

    def test_workload(self):
        body = MODULE.payload(31)
        self.assertEqual(len(body["prompt"]), 128)
        self.assertEqual(body["max_tokens"], 64)
        self.assertTrue(body["ignore_eos"])
        self.assertEqual(body["temperature"], 0)

    def test_usage(self):
        valid = {"prompt_tokens": 128, "completion_tokens": 64}
        MODULE.validate_usage([valid] * 32, 32)
        for values in ([valid] * 31, [{**valid, "completion_tokens": 63}] * 32):
            with self.assertRaises(ValueError):
                MODULE.validate_usage(values, 32)


if __name__ == "__main__":
    unittest.main()
