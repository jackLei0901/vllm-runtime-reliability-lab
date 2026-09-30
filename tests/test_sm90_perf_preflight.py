"""CPU checks for the performance definitions; no performance claim is tested."""

import hashlib
import importlib.util
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "experiments/kernel-operand-contracts/sm90-block-fp8-perf"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, PACKET / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = load("sm90_perf_builder", "build_perf.py")
protocol = load("sm90_perf_protocol", "protocol.py")


class TestSM90PerfPreflight(unittest.TestCase):
    def test_expanded_capture_points_and_derived_work_counts(self):
        self.assertEqual(len(protocol.M_VALUES), 20)
        self.assertEqual(len(protocol.CHANGED_M), 9)
        self.assertTrue({24, 40, 56} <= set(protocol.CHANGED_M))
        shapes = len(protocol.SHAPES)
        rotating = shapes * len(protocol.M_VALUES)
        changed = shapes * len(protocol.CHANGED_M)
        self.assertEqual((changed, rotating - changed), (36, 44))
        per_cell = protocol.BLOCKS * 4 * (protocol.WARMUP + protocol.REPLAYS)
        hot = shapes * 3
        self.assertEqual((rotating + hot + shapes + hot) * per_cell, 42336)
        self.assertEqual(2 * rotating * per_cell, 62720)
        self.assertEqual(len(protocol.CORRECTNESS_M), 16)
        per_arm = (shapes + 1) * len(protocol.CORRECTNESS_M) + shapes * len(
            protocol.GRAPH_M
        )
        self.assertEqual(per_arm, 88)
        self.assertEqual(2 * per_arm, 176)
        self.assertEqual(protocol.FP16_SHAPE, (4096, 4096))

    def test_historical_r2_is_unchanged(self):
        path = PACKET.parent / "sm90-contract-repair/build_minimal_r2.py"
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "a16daa2bafc476a25aea35e51de439168a205516578ac9b1e24f0e2007ed2168",
        )

    def test_namespace_rejects_injection_and_reserved_scope(self):
        for name in (
            "_C",
            "lab_sm90_contract",
            "lab_sm90_perf_a;bad",
            "lab_sm90_perf_1",
            "lab_sm90_perf_A",
            "lab_sm90_perf_é",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                build.validate_namespace(name)

    def test_namespaces_are_distinct_in_both_registration_sites(self):
        template = (PACKET / "binding.cpp").read_text()
        left = build.render_binding(template, "lab_sm90_perf_base")
        right = build.render_binding(template, "lab_sm90_perf_variant")
        self.assertEqual(left.count("lab_sm90_perf_base"), 2)
        self.assertEqual(right.count("lab_sm90_perf_variant"), 2)
        self.assertEqual(
            left.replace("lab_sm90_perf_base", "NAMESPACE"),
            right.replace("lab_sm90_perf_variant", "NAMESPACE"),
        )
        self.assertIn("Tensor! out", left)

    def test_bad_binding_anchor_fails_closed(self):
        with self.assertRaises(ValueError):
            build.render_binding("not a binding", "lab_sm90_perf_base")

    def test_variant_changes_only_declared_dispatch(self):
        base = "before\n" + build.DISPATCH_ANCHOR + "\nafter\n"
        result = build.proposed_variant(base)
        self.assertEqual(
            result.replace(build.DISPATCH_VARIANT, build.DISPATCH_ANCHOR), base
        )
        self.assertNotIn("STD_TORCH_CHECK", result)
        for invalid in ("", base + base):
            with self.assertRaises(ValueError):
                build.proposed_variant(invalid)

    def test_variant_only_changes_divisible_by_four_small_m(self):
        changed = tuple(
            m for m in protocol.M_VALUES if (m % 4 != 0) != (m <= 64 or m % 4 != 0)
        )
        self.assertEqual(changed, protocol.CHANGED_M)
        for m in (65, 68, 128, 258):
            self.assertEqual(m % 4 != 0, m <= 64 or m % 4 != 0)

    def test_rotation_is_above_twice_l2_and_whole_cycles(self):
        for set_bytes in (24 << 20, 16 << 20, 96 << 20, 48 << 20):
            count, calls = protocol.rotation_plan(50 << 20, set_bytes)
            self.assertGreater(count * set_bytes, 2 * (50 << 20))
            self.assertEqual(calls % count, 0)
            self.assertEqual(calls // count, 8)

    def test_invalid_rotation_sizes_fail(self):
        for sizes in ((0, 1), (1, 0), (-1, 1)):
            with self.assertRaises(ValueError):
                protocol.rotation_plan(*sizes)

    def test_session_a_pairs_and_session_b_pairs_use_same_order(self):
        self.assertEqual(
            protocol.pair_order(0, "m64", "m63"), ("m64", "m63", "m63", "m64")
        )
        self.assertEqual(
            protocol.pair_order(1, "base", "variant"),
            ("variant", "base", "base", "variant"),
        )
        with self.assertRaises(ValueError):
            protocol.pair_order(0, "same", "same")

    def test_pair_has_fixed_sample_count(self):
        self.assertEqual(protocol.paired_difference([12.0] * 18, [10.0] * 18), 2)
        with self.assertRaises(ValueError):
            protocol.paired_difference([12.0] * 17, [10.0] * 18)

    def test_zero_spread_cannot_remove_effect_floor(self):
        self.assertEqual(protocol.decision_bound(10, [0.0] * 7), 0.5)
        self.assertEqual(protocol.decision_bound(100, [0.0] * 7), 3)

    def test_aa_bias_and_insufficient_calibration_fail(self):
        for differences in ([1.0] * 7, [0.0] * 5):
            with self.assertRaises(ValueError):
                protocol.decision_bound(10, differences)

    def test_rejected_blocks_are_not_replaced(self):
        self.assertTrue(protocol.exceeds_bound([1.0] * 6 + [None], 0.5))
        self.assertFalse(protocol.exceeds_bound([1.0] * 5 + [None, None], 0.5))
        self.assertFalse(protocol.exceeds_bound([0.5] * 7, 0.5))
        with self.assertRaises(ValueError):
            protocol.exceeds_bound([1.0] * 8, 0.5)

    def test_nonfinite_measurements_fail_closed(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                protocol.exceeds_bound([value] * 7, 0.5)
        with self.assertRaises(ValueError):
            protocol.paired_difference([0.0] * 18, [10.0] * 18)

    def test_reviewed_model_shapes_and_quantization(self):
        config = dict(
            hidden_size=4096,
            intermediate_size=12288,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            num_hidden_layers=36,
            quantization_config=dict(
                weight_block_size=[128, 128],
                quant_method="fp8",
                fmt="e4m3",
                activation_scheme="dynamic",
            ),
        )
        self.assertEqual(protocol.model_shapes(config), protocol.SHAPES)
        config["num_hidden_layers"] = 32
        with self.assertRaises(ValueError):
            protocol.model_shapes(config)

    def test_compile_uses_namespace_without_changing_kernel_flags(self):
        # This tests arguments to load(), not compilation, linkage or CUDA.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bin").mkdir()
            (root / "bin/nvcc").touch()
            out = root / "out"
            out.mkdir()
            (out / "fake.so").write_bytes(b"fake, never loaded")
            (out / "build.ninja").write_text("fake")
            torch = types.ModuleType("torch")
            torch.__version__, torch.version = (
                "2.13.0+cu130",
                types.SimpleNamespace(cuda="13.0"),
            )
            torch.utils = types.ModuleType("torch.utils")
            cpp = types.ModuleType("torch.utils.cpp_extension")
            cpp.CUDA_HOME, cpp.COMMON_NVCC_FLAGS = str(root), []
            calls = []

            def fake_load(**kwargs):
                calls.append(kwargs)
                return str(out / "fake.so")

            cpp.load, torch.utils.cpp_extension = fake_load, cpp
            modules = {
                "torch": torch,
                "torch.utils": torch.utils,
                "torch.utils.cpp_extension": cpp,
            }
            with (
                patch.dict(sys.modules, modules),
                patch.object(sys, "platform", "linux"),
            ):
                for namespace in ("lab_sm90_perf_base", "lab_sm90_perf_variant"):
                    result = build.compile_extension(root, root, out, root, namespace)
                    self.assertEqual(result["namespace"], namespace)
                    self.assertIn(
                        namespace, (out / "binding_generated.cpp").read_text()
                    )
            self.assertEqual(
                calls[0]["extra_cuda_cflags"], calls[1]["extra_cuda_cflags"]
            )
            self.assertEqual(calls[0]["extra_cflags"], calls[1]["extra_cflags"])
            self.assertEqual(calls[0]["extra_ldflags"], calls[1]["extra_ldflags"])
            self.assertIn("-Wl,-Bsymbolic", calls[0]["extra_ldflags"])
            self.assertNotEqual(calls[0]["name"], calls[1]["name"])


if __name__ == "__main__":
    unittest.main()
