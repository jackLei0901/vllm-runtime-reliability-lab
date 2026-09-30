"""Post-run publication checks; these do not rescore the GPU experiment."""

import hashlib
import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path

PACKET = Path(__file__).resolve().parents[1] / (
    "experiments/kernel-operand-contracts/sm90-contract-repair"
)
spec = importlib.util.spec_from_file_location("sm90_r2", PACKET / "build_minimal_r2.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)
RECORDED_BUILDER_SHA = (
    "bb624fef3235d207f8c115fdffa313c84d086203d60dc9b208757e777da7d3b9"
)


def layout_accepted(name, shape, strides):
    terms = re.findall(
        rf"{name}_scales.size\((\d)\) <= 1 \|\| {name}_scales.stride\((\d)\) == "
        rf"(1|{name}_scales.size\(\d\))",
        build.GUARDS,
    )
    if len(terms) != 2:
        raise AssertionError("expected two actual patch guard terms")
    return all(
        shape[int(dim)] <= 1
        or strides[int(stride_dim)] == (1 if rhs == "1" else shape[int(rhs[-2])])
        for dim, stride_dim, rhs in terms
    )


class TestSM90R2Publication(unittest.TestCase):
    def test_successor_differs_only_in_self_receipt_name(self):
        # Reverse the sole publication change to recover exact executed bytes.
        text = (PACKET / "build_minimal_r2.py").read_text(encoding="utf-8")
        self.assertEqual(text.count("Path(__file__).name,"), 1)
        recorded = text.replace("Path(__file__).name,", '"build_minimal.py",')
        self.assertEqual(
            hashlib.sha256(recorded.encode()).hexdigest(), RECORDED_BUILDER_SHA
        )

    def test_frozen_experiment_files_still_match_manifest(self):
        manifest = json.loads((PACKET / "FREEZE_MANIFEST.json").read_text())
        for name in (
            "build_minimal.py",
            "candidate.patch",
            "test_repair_sm90.py",
            "run_isolated.py",
        ):
            path = PACKET / name
            key = path.relative_to(PACKET.parents[2]).as_posix()
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), manifest["sha256"][key]
            )

    def test_patch_guards_and_source_set_unchanged(self):
        spec = importlib.util.spec_from_file_location(
            "sm90_v1", PACKET / "build_minimal.py"
        )
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        self.assertEqual(build.GUARDS, original.GUARDS)
        self.assertEqual(build.SOURCES, original.SOURCES)
        self.assertEqual(build.PIN, original.PIN)

    def test_caller_stride_arithmetic(self):
        for m, kt, nt in ((258, 8, 4), (1, 8, 4), (256, 1, 4), (256, 8, 1)):
            self.assertTrue(layout_accepted("a", (m, kt), (1, m)))
            self.assertTrue(layout_accepted("b", (kt, nt), (1, kt)))
        self.assertFalse(layout_accepted("a", (258, 8), (8, 1)))
        self.assertFalse(layout_accepted("b", (8, 4), (4, 1)))
        self.assertFalse(layout_accepted("a", (258, 8), (1, 260)))

    def test_reference_requires_real_command_and_patched_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory)
            row = {
                "file": build.TARGET,
                "command": " ".join(
                    (
                        "nvcc -std=c++20",
                        "-DTORCH_TARGET_VERSION=0x020B000000000000ULL",
                        "-DUSE_CUDA",
                        "-DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1",
                        "-DENABLE_SCALED_MM_SM90=1",
                        "arch=compute_90a,code=sm_90a",
                    )
                ),
            }
            (reference / "compile_commands.json").write_text(json.dumps([row]))
            with self.assertRaisesRegex(ValueError, "patched torch header"):
                build.validate_reference(reference)
            headers = reference / "torch_patched_headers"
            headers.mkdir()
            (headers / "test.h").write_text("test fixture, not a real build header")
            result = build.validate_reference(reference)
            self.assertIn("test.h", result["patched_headers_sha256"])
            row["command"] = row["command"].replace("-std=c++20", "-std=c++17")
            (reference / "compile_commands.json").write_text(json.dumps([row]))
            with self.assertRaisesRegex(ValueError, "missing reference flag"):
                build.validate_reference(reference)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "CPU torch not installed")
    def test_real_caller_scale_allocations(self):
        import torch

        # Exact 2-D allocation expressions from the pinned callers. No GPU
        # quantization, vLLM import, or serving execution is represented here.
        for m, k, n in (
            (258, 1024, 512),
            (1, 1024, 512),
            (256, 128, 512),
            (256, 1024, 128),
        ):
            x = torch.empty(m, k)
            shape = x.shape[:-2] + (x.shape[-1] // 128, x.shape[-2])
            activation = torch.empty(shape, dtype=torch.float32).permute(-1, -2)
            bs = torch.empty(n // 128, k // 128, dtype=torch.float32)
            weight = bs.T
            self.assertTrue(layout_accepted("a", activation.shape, activation.stride()))
            self.assertTrue(layout_accepted("b", weight.shape, weight.stride()))
        self.assertFalse(layout_accepted("a", (258, 8), (1, 260)))


if __name__ == "__main__":
    unittest.main()
