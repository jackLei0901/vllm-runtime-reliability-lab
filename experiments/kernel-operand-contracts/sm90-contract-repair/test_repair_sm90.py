"""Draft base/fix expectations. Run each arm in a fresh process, without vLLM."""

import hashlib
import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ARM = os.environ.get("KOC_ARM")
RECEIPT = os.environ.get("KOC_BUILD_RECEIPT")
SHAPES = [(256, 512, 1024), (258, 512, 1024)]
CORRECT_MAX = 5e-3
WRONG_MIN = 5e-2
_CONTEXT_FAILED = False


@pytest.fixture(scope="module", autouse=True)
def load_verified_binary():
    assert ARM in ("base", "fix"), "KOC_ARM must be declared"
    assert RECEIPT, "KOC_BUILD_RECEIPT required"
    receipt = json.loads(Path(RECEIPT).read_text())
    assert receipt["status"] == "built_not_validated"
    assert receipt["source"]["head"] == "7b054aca96cea8be1369d651c3434ad140580b92"
    assert receipt["source"]["arm"] == ARM
    assert (
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        == receipt["harness_sha256"]["test_repair_sm90.py"]
    )
    assert torch.__version__ == receipt["binary"]["torch"]
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
    assert "vllm" not in __import__("sys").modules
    name = receipt["binary"]["binary_file"]
    assert name and Path(name).name == name and name not in (".", "..")
    binary = Path(RECEIPT).resolve().parent / name
    assert (
        hashlib.sha256(binary.read_bytes()).hexdigest()
        == receipt["binary"]["binary_sha256"]
    )
    torch.ops.load_library(str(binary))


@pytest.fixture(autouse=True)
def context_gate(record_property):
    global _CONTEXT_FAILED
    if _CONTEXT_FAILED:
        pytest.skip("unscored: CUDA context failed in an earlier case; no rerun")
    try:
        assert torch.ones(1, device="cuda").item() == 1
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001 - do not launch into an unusable context
        _CONTEXT_FAILED = True
        pytest.skip("unscored: CUDA context unavailable before case")
    yield
    try:
        assert torch.ones(1, device="cuda").item() == 1
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001 - later cases remain collected, not rerun
        _CONTEXT_FAILED = True
    record_property("cuda_context_usable_after", not _CONTEXT_FAILED)


def case(shape, out_dtype):
    m, n, k = shape
    gen = torch.Generator(device="cuda").manual_seed(730)
    a = (torch.rand(m, k, device="cuda", generator=gen) * 2 - 1).to(torch.float8_e4m3fn)
    b = (
        (torch.rand(n, k, device="cuda", generator=gen) * 2 - 1)
        .to(torch.float8_e4m3fn)
        .t()
    )
    sa = (10 ** (torch.rand(k // 128, m, device="cuda", generator=gen) * 2 - 2)).t()
    sb = (
        10 ** (torch.rand(n // 128, k // 128, device="cuda", generator=gen) * 2 - 2)
    ).t()
    out = torch.empty(m, n, device="cuda", dtype=out_dtype)
    return out, a, b, sa, sb


def invoke(tensors):
    torch.ops.lab_sm90_contract.scaled_mm(*tensors, None)


def reference(tensors):
    _, a, b, sa, sb = tensors
    aq = a.float() * sa.repeat_interleave(128, dim=1)
    bq = b.float() * sb.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    return aq @ bq


def error(out, ref):
    torch.cuda.synchronize()
    if not torch.isfinite(out).all():
        return float("inf")
    return ((out.float() - ref).norm() / ref.norm()).item()


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
def test_valid(shape, out_dtype):
    tensors = case(shape, out_dtype)
    ref = reference(tensors)
    invoke(tensors)
    assert error(tensors[0], ref) < CORRECT_MAX


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("operand", ["A", "B"])
@pytest.mark.parametrize("dtype", [torch.float8_e5m2, torch.int8])
def test_wrong_dtype(shape, operand, dtype):
    tensors = list(case(shape, torch.bfloat16))
    index = 1 if operand == "A" else 2
    source = tensors[index].float()
    if dtype == torch.int8:
        source = source * 50
    tensors[index] = source.to(dtype)
    if ARM == "fix":
        with pytest.raises(RuntimeError, match=f"requires {operand} dtype"):
            invoke(tensors)
    else:
        ref = reference(tensors)
        invoke(tensors)
        assert error(tensors[0], ref) > WRONG_MIN


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("scale", ["A", "B"])
def test_wrong_scale_layout(shape, scale):
    tensors = list(case(shape, torch.bfloat16))
    index = 3 if scale == "A" else 4
    tensors[index] = tensors[index].contiguous()
    if ARM == "fix":
        layout = "column-major" if scale == "A" else "K-major"
        with pytest.raises(RuntimeError, match=f"packed {layout} {scale} scales"):
            invoke(tensors)
    else:
        ref = reference(tensors)
        invoke(tensors)
        assert error(tensors[0], ref) > WRONG_MIN


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("index,name", [(1, "A"), (2, "B"), (0, "output")])
def test_inherited_padded_operand(shape, index, name):
    tensors = list(case(shape, torch.bfloat16))
    target = tensors[index]
    if index == 2:
        padded = torch.empty(
            target.shape[1], target.shape[0] + 128, device="cuda", dtype=target.dtype
        ).t()[: target.shape[0]]
    else:
        padded = torch.empty(
            target.shape[0], target.shape[1] + 128, device="cuda", dtype=target.dtype
        )[:, : target.shape[1]]
    padded.copy_(target)
    tensors[index] = padded
    with pytest.raises(RuntimeError, match=f"{name} is not packed"):
        invoke(tensors)


@pytest.mark.parametrize("shape", [(1, 512, 1024), (256, 128, 1024), (256, 512, 128)])
def test_singleton_scale_equivalence(shape):
    tensors = list(case(shape, torch.bfloat16))
    assert not (
        tensors[3].numel() in (1, shape[0]) and tensors[4].numel() in (1, shape[1])
    ), "this control must reach the blockwise branch"
    changed = 0
    for index in (3, 4):
        target = tensors[index]
        if 1 not in target.shape:
            continue
        # Allocate at the logical shape, not .contiguous() of a view which
        # PyTorch already considers contiguous due to its singleton dimension.
        alternative = torch.empty(
            tuple(target.shape), device=target.device, dtype=target.dtype
        )
        alternative.copy_(target)
        assert alternative.stride() != target.stride()
        assert not (
            alternative.stride(0) == 1 and alternative.stride(1) == alternative.size(0)
        ), "control must require the singleton exemption"
        tensors[index] = alternative
        changed += 1
    assert changed > 0
    ref = reference(tensors)
    invoke(tensors)
    assert error(tensors[0], ref) < CORRECT_MAX


@pytest.mark.parametrize("invalid", ["dtype", "A_layout", "B_layout"])
def test_fix_rejects_during_capture(invalid):
    if ARM != "fix":
        pytest.skip("base unsafe inputs are measured eagerly, not captured")
    warmup_control = case(SHAPES[0], torch.bfloat16)
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            invoke(warmup_control)
    torch.cuda.current_stream().wait_stream(warmup)
    tensors = list(case(SHAPES[0], torch.bfloat16))
    if invalid == "dtype":
        tensors[1] = tensors[1].to(torch.float8_e5m2)
        match = "requires A dtype"
    elif invalid == "A_layout":
        tensors[3] = tensors[3].contiguous()
        match = "packed column-major A scales"
    else:
        tensors[4] = tensors[4].contiguous()
        match = "packed K-major B scales"
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match=match):
        with torch.cuda.graph(graph):
            invoke(tensors)
    torch.cuda.synchronize()
    control = case(SHAPES[0], torch.bfloat16)
    ref = reference(control)
    invoke(control)
    assert error(control[0], ref) < CORRECT_MAX


@pytest.mark.parametrize("shape", SHAPES)
def test_valid_graph_capture_and_replay(shape):
    tensors = case(shape, torch.bfloat16)
    ref = reference(tensors)
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            invoke(tensors)
    torch.cuda.current_stream().wait_stream(warmup)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke(tensors)
    graph.replay()
    assert error(tensors[0], ref) < CORRECT_MAX
