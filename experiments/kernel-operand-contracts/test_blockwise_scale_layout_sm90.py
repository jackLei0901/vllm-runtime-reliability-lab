"""B1: does SM90 blockwise ``cutlass_scaled_mm`` honor the scale tensors' layout?

Inventory: docs/kernel/QUANT_GEMM_OPERAND_CONTRACT_INVENTORY_2026-09-29.md (B1).
At vLLM ``91d7324c`` the SM90 blockwise dispatch derives both scale layouts from
shape alone (``tile_atom_to_shape_SFA/SFB``) and requires activation scales
M-major (column-major ``[M, K/128]``) and weight scales K-major (``[K/128,
N/128]``); the helper checks only their dims and shape. Predictions, frozen
before running:

* B1-a  row-major activation scales, M > 1 and K/128 > 1  -> silent wrong
* B1-b  same, but M == 1 or K == 128 (layouts coincide)    -> correct
* B1-c  weight scales N-major instead of the ``Bs.T`` view  -> silent wrong
* controls with the required layouts                       -> correct

A "raised" outcome refutes a silent-wrong prediction (the op is loud). The
dispatch swaps A and B when ``M % 4 != 0`` (dispatch L210), so both an M that
is a multiple of 4 and one that is not are run. Scales are drawn log-uniform so
that a transposed read changes the result by far more than FP8 rounding.

Run on an SM90 GPU inside the vLLM environment under test:
    pytest -q -rA test_blockwise_scale_layout_sm90.py --junitxml=b1.xml
The per-case ``rel_diff`` and outcome are recorded as junit properties.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("vllm")
from vllm import _custom_ops as ops  # noqa: E402

BLOCK = 128
CORRECT_MAX = 5e-3  # bf16 output rounding alone can approach 2**-8 relative
WRONG_MIN = 5e-2  # below this a "wrong" prediction is not considered shown

needs_sm90 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="B1 targets the SM90 blockwise CUTLASS path",
)


def make_case(m: int, n: int, k: int, seed: int = 0):
    """FP8 operands plus log-uniform block scales, and a float reference."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    fp8 = torch.float8_e4m3fn
    a = (torch.rand(m, k, device="cuda", generator=gen) * 2 - 1).to(fp8)
    b = (torch.rand(n, k, device="cuda", generator=gen) * 2 - 1).to(fp8)
    kt, nt = -(-k // BLOCK), -(-n // BLOCK)
    a_scales = 10 ** (torch.rand(m, kt, device="cuda", generator=gen) * 2 - 2)
    b_scales = 10 ** (torch.rand(nt, kt, device="cuda", generator=gen) * 2 - 2)
    a_deq = a.float() * a_scales.repeat_interleave(BLOCK, dim=1)[:, :k]
    b_full = b_scales.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
    b_deq = b.float() * b_full[:n, :k]
    return a, b, a_scales, b_scales, a_deq @ b_deq.t()


def run(a, b, scale_a, scale_b, ref) -> tuple[str, float | None]:
    """Return (outcome, rel_diff) with outcome in correct/wrong/ambiguous/raised."""
    try:
        out = ops.cutlass_scaled_mm(a, b.t(), scale_a, scale_b, torch.bfloat16)
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001 - any error means the op was loud
        return "raised", None
    rel = (out.float() - ref).abs().mean() / ref.abs().mean()
    rel = float(rel)
    if rel < CORRECT_MAX:
        return "correct", rel
    if rel > WRONG_MIN:
        return "wrong", rel
    return "ambiguous", rel


def check(record_property, expected: str, outcome: str, rel: float | None) -> None:
    record_property("outcome", outcome)
    record_property("rel_diff", rel)
    record_property("expected", expected)
    assert outcome == expected, f"predicted {expected}, observed {outcome} ({rel})"


def column_major(t: torch.Tensor) -> torch.Tensor:
    return t.t().contiguous().t()


@pytest.fixture(autouse=True)
def _require_blockwise():
    if not ops.cutlass_scaled_mm_supports_block_fp8(90):
        pytest.skip("this build has no SM90 blockwise CUTLASS FP8 kernel")


# M = 256 takes the non-swapped kernel; M = 258 (M % 4 != 0) the swapped one.
SHAPES = [(256, 512, 1024), (258, 512, 1024)]


@needs_sm90
@pytest.mark.parametrize("m,n,k", SHAPES)
def test_control_required_layouts(record_property, m, n, k):
    a, b, a_s, b_s, ref = make_case(m, n, k)
    check(record_property, "correct", *run(a, b, column_major(a_s), b_s.t(), ref))


@needs_sm90
@pytest.mark.parametrize("m,n,k", SHAPES)
def test_b1a_row_major_activation_scales(record_property, m, n, k):
    a, b, a_s, b_s, ref = make_case(m, n, k)
    check(record_property, "wrong", *run(a, b, a_s.contiguous(), b_s.t(), ref))


@needs_sm90
@pytest.mark.parametrize("m,n,k", [(1, 512, 1024), (256, 512, 128)])
def test_b1b_layouts_coincide(record_property, m, n, k):
    a, b, a_s, b_s, ref = make_case(m, n, k)
    check(record_property, "correct", *run(a, b, a_s.contiguous(), b_s.t(), ref))


@needs_sm90
@pytest.mark.parametrize("m,n,k", SHAPES)
def test_b1c_n_major_weight_scales(record_property, m, n, k):
    a, b, a_s, b_s, ref = make_case(m, n, k)
    wrong_b_scales = b_s.t().contiguous()
    check(
        record_property,
        "wrong",
        *run(a, b, column_major(a_s), wrong_b_scales, ref),
    )
