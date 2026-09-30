import importlib.util
import re
import unittest
from pathlib import Path
from unittest.mock import patch

PACKET = Path(__file__).resolve().parents[1] / (
    "experiments/kernel-operand-contracts/sm90-contract-repair"
)
spec = importlib.util.spec_from_file_location("sm90_build", PACKET / "build_minimal.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


def b_layout_accepted(shape, strides):
    """Evaluate only the declared B guard, independent of GPU fixtures."""
    terms = re.findall(
        r"b_scales.size\((\d)\) <= 1 \|\| b_scales.stride\((\d)\) == "
        r"(1|b_scales.size\(\d\))",
        build.GUARDS,
    )
    if len(terms) != 2:
        raise AssertionError("expected two B layout terms")
    for dim, stride_dim, rhs in terms:
        expected = 1 if rhs == "1" else shape[int(rhs[-2])]
        if shape[int(dim)] > 1 and strides[int(stride_dim)] != expected:
            return False
    return True


class TestSM90BuildPlan(unittest.TestCase):
    def test_b1_weight_scale_stride_arithmetic(self):
        self.assertTrue(b_layout_accepted((8, 4), (1, 8)))
        self.assertFalse(b_layout_accepted((8, 4), (4, 1)))

    def test_weight_scale_singleton_equivalence(self):
        for shape in ((8, 1), (1, 4)):
            original = (1, shape[0])
            alternative = (shape[1], 1)
            self.assertNotEqual(original, alternative)
            self.assertFalse(alternative == (1, shape[0]))
            self.assertTrue(b_layout_accepted(shape, alternative))
            for strides in ((1, shape[0]), (shape[1], 1)):
                self.assertTrue(b_layout_accepted(shape, strides))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "CPU torch not installed")
    def test_real_cpu_bs_transpose_passes_and_b1c_fails(self):
        import torch

        bs = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        valid = bs.T
        wrong = valid.contiguous()
        self.assertEqual(valid.stride(), (1, 8))
        self.assertEqual(wrong.stride(), (4, 1))
        self.assertTrue(b_layout_accepted(valid.shape, valid.stride()))
        self.assertFalse(b_layout_accepted(wrong.shape, wrong.stride()))
        self.assertTrue(torch.equal(valid, wrong))

    def test_unique_guard_insertion(self):
        base = "prefix\n" + build.ANCHOR + "suffix\n"
        fix = build.proposed_fix(base)
        self.assertEqual(fix, "prefix\n" + build.ANCHOR + build.GUARDS + "suffix\n")

    def test_missing_or_duplicate_anchor_rejected(self):
        for text in ("absent", build.ANCHOR * 2):
            with self.assertRaises(ValueError):
                build.proposed_fix(text)

    def test_real_entry_and_kernel_in_source_set(self):
        self.assertIn(build.ROOT + "scaled_mm_entry.cu", build.SOURCES)
        self.assertIn(build.TARGET, build.SOURCES)
        self.assertEqual(len(build.SOURCES), len(set(build.SOURCES)))

    def test_wrong_head_fails_before_reading_source(self):
        with patch.object(build, "git", return_value="0" * 40):
            with self.assertRaisesRegex(ValueError, "wrong source HEAD"):
                build.validate_source(Path("unused"), "base")

    def test_dirty_base_fails_before_reading_source(self):
        with patch.object(build, "git", side_effect=[build.PIN, "?? unexpected"]):
            with self.assertRaisesRegex(ValueError, "clean base"):
                build.validate_source(Path("unused"), "base")

    def test_fix_cannot_include_unrelated_changes(self):
        with patch.object(build, "git", side_effect=[build.PIN, "M unrelated"]):
            with self.assertRaisesRegex(ValueError, "exactly the unstaged fix"):
                build.validate_source(Path("unused"), "fix")

    def test_cutlass_wrong_revision(self):
        with patch.object(build, "git", side_effect=["a" * 40, "b" * 40]):
            with self.assertRaisesRegex(ValueError, "v4.7.1"):
                build.validate_cutlass(Path("unused"))

    def test_binding_has_private_namespace(self):
        text = (PACKET / "binding.cpp").read_text()
        self.assertIn("STABLE_TORCH_LIBRARY(lab_sm90_contract, ops)", text)
        self.assertNotIn("STABLE_TORCH_LIBRARY(_C", text)

    def test_candidate_contains_exact_declared_guards(self):
        text = (PACKET / "candidate.patch").read_text()
        additions = (
            "\n".join(
                line[1:]
                for line in text.splitlines()
                if line.startswith("+") and not line.startswith("+++")
            )
            + "\n"
        )
        self.assertEqual(additions, build.GUARDS)


if __name__ == "__main__":
    unittest.main()
