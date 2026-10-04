import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments/native-kernel-discovery/collect.py"
)
SPEC = importlib.util.spec_from_file_location("discovery", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(PATH.parent))
try:
    SPEC.loader.exec_module(MODULE)
    ADMIT_SPEC = importlib.util.spec_from_file_location(
        "admission", PATH.with_name("admit.py")
    )
    ADMIT = importlib.util.module_from_spec(ADMIT_SPEC)
    ADMIT_SPEC.loader.exec_module(ADMIT)
finally:
    sys.path.pop(0)
PREFLIGHT = sys.modules["preflight"]


class TestDiscovery(unittest.TestCase):
    def test_idle_inventory(self):
        gpu = "NVIDIA H800 PCIe, GPU-private, 595.71.05, 81559 MiB, 0 MiB"
        ADMIT.validate_idle(gpu, "")
        for record, pids in (
            (gpu, "123"),
            (gpu, "[N/A]"),
            (gpu.replace("0 MiB", "129 MiB"), ""),
            (gpu.replace("H800", "4090"), ""),
        ):
            with self.subTest(record=record, pids=pids), self.assertRaises(ValueError):
                ADMIT.validate_idle(record, pids)

    def test_no_gpu_admission_dry_run_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root, receipt, commit, fetch = self.public_fixture(directory)
            for name in ("admit.py", "collect.py", "preflight.py"):
                (root / "experiments/native-kernel-discovery" / name).write_bytes(
                    PATH.with_name(name).read_bytes()
                )
            PREFLIGHT.prepare_public(root, commit, receipt, fetch)
            model = Path(directory) / "model"
            model.mkdir()
            (model / "config.json").write_bytes(b"{}")
            (model / "weights").write_bytes(b"abc")
            prep = {
                "pin": MODULE.PIN,
                "revision": MODULE.REVISION,
                "installed_verified": True,
                "source_verified": True,
                "weight_hashes_verified": True,
                "model_directory": str(model),
                "config_sha256": PREFLIGHT.sha(model / "config.json"),
                "weight_file_sizes": {"weights": 3},
                "isolated_environment_created": sys.prefix,
                "installed_versions": {
                    "vllm": "0.30.1rc1.dev618+gb0e21b308",
                    "torch": "2.13.0+cu130",
                    "flashinfer-python": "0.7.0.post1",
                    "flashinfer-cubin": "0.7.0.post1",
                },
            }
            preparation = Path(directory) / "prep.json"
            preparation.write_text(json.dumps(prep))
            identity = Path(directory) / "identity.json"
            identity.write_text(
                json.dumps(
                    {
                        "wheel_members_checked": 5305,
                        "mismatch_count": 0,
                        "mismatches": [],
                    }
                )
            )
            work = Path(directory) / "dry-run"
            argv = [
                "admit.py",
                "--packet-root",
                str(root),
                "--freeze",
                commit,
                "--public-receipt",
                str(receipt),
                "--preparation",
                str(preparation),
                "--identity",
                str(identity),
                "--work",
                str(work),
                "--dry-run",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    ADMIT.importlib.metadata,
                    "version",
                    side_effect=prep["installed_versions"].__getitem__,
                ),
                patch.object(
                    ADMIT.subprocess,
                    "check_output",
                    side_effect=AssertionError("GPU command"),
                ),
                patch.object(
                    ADMIT.subprocess, "run", side_effect=AssertionError("model launch")
                ),
                patch("urllib.request.urlopen", side_effect=AssertionError("network")),
            ):
                self.assertEqual(ADMIT.main(), 0)
            self.assertEqual(
                json.loads((work / "offline-checks.json").read_text())["status"],
                "offline_checks_passed",
            )
            self.assertFalse((work / "admission.json").exists())
            self.assertFalse((work / "collection").exists())

    def test_timeout_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "collection").mkdir()
            (work / "collection/collection.json").write_text(
                json.dumps({"private_process_group": 12345})
            )
            with (
                patch.object(
                    ADMIT.os,
                    "killpg",
                    side_effect=[None, ProcessLookupError()],
                    create=True,
                ) as kill,
                patch.object(ADMIT.time, "sleep"),
            ):
                ADMIT.cleanup_server(work)
            self.assertEqual(kill.call_args_list[0].args[0], 12345)

    def test_identity_real_receipt_schema(self):
        PREFLIGHT.validate_identity(
            {"wheel_members_checked": 5305, "mismatch_count": 0, "mismatches": []}, 5305
        )

    def test_identity_fail_closed(self):
        valid = {"wheel_members_checked": 5305, "mismatch_count": 0, "mismatches": []}
        mutations = [
            {k: v for k, v in valid.items() if k != missing} for missing in valid
        ]
        mutations += [{**valid, "mismatch_count": value} for value in (1, False, "0")]
        mutations += [{**valid, "mismatches": value} for value in (None, 0, ["bad"])]
        mutations += [
            {**valid, "wheel_members_checked": value}
            for value in (0, 5304, "5305", True)
        ]
        for receipt in mutations:
            with self.subTest(receipt=receipt), self.assertRaises(ValueError):
                PREFLIGHT.validate_identity(receipt, 5305)

    def public_fixture(self, directory):
        root = Path(directory) / "packet"
        for relative in PREFLIGHT.FILES:
            p = root / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(relative.encode())
        receipt = Path(directory) / "private" / "public.json"
        commit = "a" * 40

        def fetch(url):
            return (root / url.split(commit + "/", 1)[1]).read_bytes()

        return root, receipt, commit, fetch

    def test_verified_receipt_works_with_network_down(self):
        with tempfile.TemporaryDirectory() as directory:
            root, receipt, commit, fetch = self.public_fixture(directory)
            PREFLIGHT.prepare_public(root, commit, receipt, fetch)
            original = receipt.read_bytes()
            with patch("urllib.request.urlopen", side_effect=OSError("offline")):
                PREFLIGHT.check_public(root, commit, receipt)
                PREFLIGHT.prepare_public(root, commit, receipt)
            self.assertEqual(receipt.read_bytes(), original)

    def test_partial_success_is_not_refetched(self):
        with tempfile.TemporaryDirectory() as directory:
            root, receipt, commit, fetch = self.public_fixture(directory)
            calls = []

            def fail_second(url):
                calls.append(url)
                if len(calls) == 2:
                    raise OSError("503")
                return fetch(url)

            with self.assertRaises(OSError):
                PREFLIGHT.prepare_public(root, commit, receipt, fail_second)
            self.assertFalse(receipt.exists())
            failures = list(receipt.parent.glob("*.failure-*.json"))
            self.assertEqual(len(failures), 1)
            original_failure = failures[0].read_bytes()
            resumed = []

            def resume(url):
                resumed.append(url)
                return fetch(url)

            PREFLIGHT.prepare_public(root, commit, receipt, resume)
            self.assertNotIn(calls[0], resumed)
            self.assertEqual(len(resumed), len(PREFLIGHT.FILES) - 1)
            self.assertEqual(failures[0].read_bytes(), original_failure)

    def test_public_mismatch_missing_and_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root, receipt, commit, fetch = self.public_fixture(directory)
            with self.assertRaises(FileNotFoundError):
                PREFLIGHT.check_public(root, commit, receipt)
            with self.assertRaises(ValueError):
                PREFLIGHT.prepare_public(root, commit, receipt, lambda _: b"wrong")
            self.assertFalse(receipt.exists())
            PREFLIGHT.prepare_public(root, commit, receipt, fetch)
            with self.assertRaises(ValueError):
                PREFLIGHT.check_public(root, "b" * 40, receipt)
            data = json.loads(receipt.read_text())
            data["files"].pop(PREFLIGHT.FILES[0])
            receipt.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                PREFLIGHT.check_public(root, commit, receipt)

    def test_local_edit_rejected_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root, receipt, commit, fetch = self.public_fixture(directory)
            PREFLIGHT.prepare_public(root, commit, receipt, fetch)
            (root / PREFLIGHT.FILES[2]).write_bytes(b"edited")
            with patch("urllib.request.urlopen", side_effect=AssertionError("network")):
                with self.assertRaises(ValueError):
                    PREFLIGHT.check_public(root, commit, receipt)

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
