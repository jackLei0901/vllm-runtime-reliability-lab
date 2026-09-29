"""B2 and B0: operand dtype and operand-stride checks on SM90 blockwise scaled_mm.

Inventory: docs/kernel/QUANT_GEMM_OPERAND_CONTRACT_INVENTORY_2026-09-29.md (B0,
B2). Scoring, unscored rules, reruns and the session limit are in
B2_B0_H800_PROTOCOL_2026-09-29.md next to this file. Source anchors are at the
inventory pin ``91d7324c`` and are checked by ``check_anchors.py``:

* B2 (new question): ``dispatch_scaled_mm`` checks operand dtype only in its
  standard branch (helper L26-29); the blockwise branch (helper L39-56) checks
  scale dims, shapes and bias; the SM90 blockwise file checks only the output
  dtype. The SM90 standard path checks A and B separately (S7). Whether other
  dtypes should be rejected is an API-contract question this test does not
  settle.
* B0 (known #55534 class, Q6 regression vector): the SM90 blockwise caller
  rebuilds A, B and C strides as packed from the shape (dispatcher L155-162);
  the entry checks inner strides and 16-element alignment but not A's leading
  stride. #56248 would honor the strides, #55537 would reject them.

Predictions, frozen before running (controls use packed FP8 e4m3 operands and
the required scale layouts confirmed in B1):

* control              packed e4m3, required layouts             -> correct
* B2-a                 e5m2 activations                          -> wrong
* B2-b                 e5m2 weights                              -> wrong
* B2-c                 int8 activations and weights              -> wrong
* B2-std               e5m2 activations, per-tensor scales
                       (standard branch of the same op)          -> raised
* B0-a                 A as a padded view, stride(0) = K + 128   -> wrong
* B0-b                 B as a padded view, column stride K + 128 -> wrong
* B0-c                 output as a padded view, stride(0) = N+128 -> wrong

Eight case types at M = 256 (non-swapped kernel) and M = 258 (M % 4 != 0,
swapped kernel; dispatcher L210): 16 collected cases. "wrong" includes a
non-finite output. A caught exception is "raised". After a raised case the test
probes the CUDA context; if the probe fails, every later case in this process is
skipped as unusable (unscored). Receipts go to the JSONL file named by
``KOC_RECEIPTS``, flushed and fsynced per record: ``start`` before any tensor or
reference setup, ``result`` after the op, ``end`` at teardown. So ``start``
without ``end`` marks a hard abort at any stage, and ``start`` plus ``end``
without ``result`` marks an in-process failure before the op returned. The raw
op is called with an explicit output tensor, so the Python wrapper's Triton
fallback is never taken.

Run on an SM90 GPU inside the vLLM environment under test (see the protocol):
    KOC_RECEIPTS=b2_b0_receipts.jsonl pytest -p no:randomly -q -rA \
        -o junit_family=xunit1 test_blockwise_dtype_and_stride_sm90.py \
        --junitxml=b2_b0.xml
"""

from __future__ import annotations

import json
import os

import pytest
import torch

pytest.importorskip("vllm")
from vllm import _custom_ops as ops  # noqa: E402

BLOCK = 128
PAD = 128  # keeps every padded stride a multiple of 16 elements
CORRECT_MAX = 5e-3  # bf16 output rounding alone can approach 2**-8 relative
WRONG_MIN = 5e-2  # below this a "wrong" prediction is not considered shown
E4M3 = torch.float8_e4m3fn
E5M2 = torch.float8_e5m2
RECEIPTS = os.environ.get("KOC_RECEIPTS", "b2_b0_receipts.jsonl")
_UNUSABLE_AFTER: list[str] = []  # node id of the case that broke the context

needs_sm90 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="targets the SM90 blockwise CUTLASS path",
)

# M = 256 takes the non-swapped kernel; M = 258 (M % 4 != 0) the swapped one.
SHAPES = [(256, 512, 1024), (258, 512, 1024)]


def emit(record: dict) -> None:
    """Append one receipt and force it to disk before continuing."""
    with open(RECEIPTS, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def cuda_usable() -> bool:
    try:
        probe = torch.ones(1, device="cuda")
        probe.add_(1)
        torch.cuda.synchronize()
        return bool(probe.item() == 2.0)
    except Exception:  # noqa: BLE001 - any failure means the context is unusable
        return False


def operand(rows: int, cols: int, dtype: torch.dtype, gen) -> torch.Tensor:
    if dtype is torch.int8:
        return torch.randint(
            -100, 101, (rows, cols), device="cuda", generator=gen, dtype=torch.int8
        )
    return (torch.rand(rows, cols, device="cuda", generator=gen) * 2 - 1).to(dtype)


def make_case(m, n, k, a_dtype=E4M3, b_dtype=E4M3, seed=0):
    """Operands of the given dtypes, log-uniform block scales, float reference.

    The reference uses each operand's true values in its own dtype, so a kernel
    that reinterprets the bytes as e4m3 disagrees with it.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    a = operand(m, k, a_dtype, gen)
    b = operand(n, k, b_dtype, gen)
    kt, nt = -(-k // BLOCK), -(-n // BLOCK)
    a_scales = 10 ** (torch.rand(m, kt, device="cuda", generator=gen) * 2 - 2)
    b_scales = 10 ** (torch.rand(nt, kt, device="cuda", generator=gen) * 2 - 2)
    a_deq = a.float() * a_scales.repeat_interleave(BLOCK, dim=1)[:, :k]
    b_full = b_scales.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
    b_deq = b.float() * b_full[:n, :k]
    return a, b, a_scales, b_scales, a_deq @ b_deq.t(), gen


def column_major(t: torch.Tensor) -> torch.Tensor:
    return t.t().contiguous().t()


def padded_view(t: torch.Tensor, gen) -> torch.Tensor:
    """Same logical values as ``t`` inside a wider buffer with random padding."""
    rows, cols = t.shape
    full = operand(rows, cols + PAD, t.dtype, gen)
    full[:, :cols].copy_(t)
    return full[:, :cols]


def run(a, b_nk, scale_a, scale_b, ref, out=None):
    """Call the raw op; return (outcome, rel_diff, error)."""
    m, n = a.shape[0], b_nk.shape[0]
    if out is None:
        out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    try:
        torch.ops._C.cutlass_scaled_mm(out, a, b_nk.t(), scale_a, scale_b, None)
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - any error means the op was loud
        first_line = str(exc).splitlines()[0] if str(exc) else ""
        return "raised", None, f"{type(exc).__name__}: {first_line[:200]}"
    result = out.float()
    if not torch.isfinite(result).all():
        return "wrong", float("inf"), None
    rel = float((result - ref).abs().mean() / ref.abs().mean())
    if rel < CORRECT_MAX:
        return "correct", rel, None
    if rel > WRONG_MIN:
        return "wrong", rel, None
    return "ambiguous", rel, None


@pytest.fixture(autouse=True)
def _guard(request):
    """Write ``start`` before any setup and ``end`` at teardown.

    Skips unsupported builds (no receipt) and, with a receipt, every case after
    the CUDA context broke. ``start`` precedes tensor and reference setup, so a
    hard abort at any stage leaves ``start`` without ``end``; an in-process
    failure before the op returns leaves ``start`` and ``end`` without
    ``result``.
    """
    if not ops.cutlass_scaled_mm_supports_block_fp8(90):
        pytest.skip("this build has no SM90 blockwise CUTLASS FP8 kernel")
    case = request.node.nodeid
    if _UNUSABLE_AFTER:
        emit({"case": case, "event": "skipped_context_unusable"})
        pytest.skip(f"CUDA context unusable after {_UNUSABLE_AFTER[0]}; unscored")
    emit({"case": case, "event": "start"})
    yield
    emit({"case": case, "event": "end"})


def score(request, record_property, expected, a, b_nk, scale_a, scale_b, ref, out=None):
    """Run the op, write the durable ``result``, then compare with the prediction."""
    case = request.node.nodeid
    outcome, rel, error = run(a, b_nk, scale_a, scale_b, ref, out)
    usable = True
    if outcome == "raised":
        usable = cuda_usable()
        if not usable:
            _UNUSABLE_AFTER.append(case)
    emit(
        {
            "case": case,
            "event": "result",
            "expected": expected,
            "outcome": outcome,
            "rel_diff": rel,
            "error": error,
            "cuda_usable_after": usable,
        }
    )
    record_property("outcome", outcome)
    record_property("rel_diff", rel)
    record_property("expected", expected)
    record_property("error", error)
    assert outcome == expected, f"predicted {expected}, observed {outcome} ({rel})"


@needs_sm90
@pytest.mark.parametrize("m,n,k", SHAPES)
def test_control_packed_e4m3(request, record_property, m, n, k):
    a, b, a_s, b_s, ref, _ = make_case(m, n, k)
    score(request, record_property, "correct", a, b, column_major(a_s), b_s.t(), ref)


@needs_sm90
@pytest.mark.parametrize("m,n,k", SHAPES)
@pytest.mark.parametrize(
    "case,a_dtype,b_dtype",
    [("B2-a", E5M2, E4M3), ("B2-b", E4M3, E5M2), ("B2-c", torch.int8, torch.int8)],
)
def test_b2_blockwise_operand_dtype(
    request, record_property, case, a_dtype, b_dtype, m, n, k
):
    a, b, a_s, b_s, ref, _ = make_case(m, n, k, a_dtype, b_dtype)
    score(request, record_property, "wrong", a, b, column_major(a_s), b_s.t(), ref)


@needs_sm90
@pytest.mark.parametrize("m,n,k", SHAPES)
def test_b2_std_standard_branch_rejects_e5m2(request, record_property, m, n, k):
    a, b, _, _, _, _ = make_case(m, n, k, E5M2, E4M3)
    one = torch.ones(1, 1, device="cuda")
    ref = a.float() @ b.float().t()
    score(request, record_property, "raised", a, b, one, one, ref)


@needs_sm90
@pytest.mark.parametrize("m,n,k", SHAPES)
def test_b0a_padded_activation_view(request, record_property, m, n, k):
    a, b, a_s, b_s, ref, gen = make_case(m, n, k)
    a_view = padded_view(a, gen)
    assert a_view.stride(0) == k + PAD
    score(request, record_property, "wrong", a_view, b, column_major(a_s), b_s.t(), ref)


@needs_sm90
@pytest.mark.parametrize("m,n,k", SHAPES)
def test_b0b_padded_weight_view(request, record_property, m, n, k):
    a, b, a_s, b_s, ref, gen = make_case(m, n, k)
    b_view = padded_view(b, gen)
    assert b_view.t().stride(1) == k + PAD
    score(request, record_property, "wrong", a, b_view, column_major(a_s), b_s.t(), ref)


@needs_sm90
@pytest.mark.parametrize("m,n,k", SHAPES)
def test_b0c_padded_output_view(request, record_property, m, n, k):
    a, b, a_s, b_s, ref, _ = make_case(m, n, k)
    full = torch.zeros(m, n + PAD, device="cuda", dtype=torch.bfloat16)
    out_view = full[:, :n]
    assert out_view.stride(0) == n + PAD
    score(
        request,
        record_property,
        "wrong",
        a,
        b,
        column_major(a_s),
        b_s.t(),
        ref,
        out_view,
    )
