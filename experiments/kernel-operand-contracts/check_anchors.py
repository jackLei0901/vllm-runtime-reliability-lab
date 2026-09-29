"""Check every source line anchor cited by the operand-contract records.

Default mode fetches each cited file at the pinned vLLM commit, verifies its git
blob id, and asserts that each cited line contains the expected text. Uses
``gh api`` when available, otherwise raw.githubusercontent.com. Exit status is
non-zero if any blob or anchor fails.

``--local PATH`` instead hashes the six source files the B2/B0 protocol requires
in a local vLLM checkout and compares them with the pinned blobs.
"""

# ruff: noqa: E501 - the pinned path/blob table is clearer unwrapped

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

PIN = "91d7324cb19d301c72d849e457221ee8dd645024"
C = "csrc/libtorch_stable/quantization/w8a8/cutlass/"
LIN = "vllm/model_executor/kernels/linear/scaled_mm/"
TQ = "tests/kernels/quantization/"

BLOBS = {
    C + "scaled_mm_entry.cu": "51f84d2ffd95d1a2dd5d1d4273c52de6fca93ec1",
    C + "c3x/scaled_mm_helper.hpp": "913436186c3a4abcb9f68c050553e278a1ae6370",
    C + "c3x/scaled_mm_blockwise_sm90_fp8_dispatch.cuh": (
        "529b28ceece5d4a50a30970e03689a6cb244fc1a"
    ),
    C
    + "c3x/scaled_mm_blockwise_sm90_fp8.cu": "a97909d37ae8f699075ee6c31497681e8bea3a83",
    C
    + "c3x/scaled_mm_sm90_fp8_dispatch.cuh": "2fae3016c3092a7e8ed8e43519ee32183ad57490",
    C
    + "c3x/scaled_mm_sm90_int8_dispatch.cuh": "a2ec816c9c937016fccbfbf729ed91b203261a52",
    C + "c3x/scaled_mm_sm90_fp8.cu": "e86c9bd48d3fc40512c080477907902bb27ac3ac",
    C + "moe/grouped_mm_c3x_sm90.cu": "d494bfb9d08cb5708b0d1ee877acf1175fb73ef2",
    C + "moe/grouped_mm_c3x.cuh": "b4cd520e9679792f26179730499cea7c46dd41e8",
    C + "moe/get_group_starts.cuh": "6942c4b48ccbca248d6fa734f96f36632ad08b8d",
    "csrc/libtorch_stable/torch_bindings.cpp": "a76852df89e5329c839b8fc734be86d55d942723",
    "vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py": (
        "cd7da3e1e3a0c702d7ece0b6eb52683d92efcdf4"
    ),
    LIN + "cutlass.py": "da8c69eed0a2ce640a358ac89497bb1e3631384d",
    LIN + "flashinfer.py": "1407530e9c7864f3b8ce82a2c5b4995b1aaa93ae",
    LIN + "deep_gemm.py": "19f9347e311e36327d93d6ea4bce090a53b50814",
    LIN + "pytorch.py": "1f2756cc267e6fbbab92494730ef5c400fe3af27",
    "vllm/model_executor/layers/quantization/utils/fp8_utils.py": (
        "79859a69d44055c1bfc512ee67b33ebde7b2f738"
    ),
    TQ + "test_cutlass_scaled_mm.py": "9e1d6d2cc76cb0ca2740a29b6c9a9b0e69b7feb4",
    TQ + "test_block_fp8.py": "038b94bd2d741f995a3857c2608f1bf36f2d0ace",
}

# The six files whose blobs gate a B2/B0 run (see the protocol).
RUN_GATE = [
    C + "scaled_mm_entry.cu",
    C + "c3x/scaled_mm_helper.hpp",
    C + "c3x/scaled_mm_blockwise_sm90_fp8_dispatch.cuh",
    C + "c3x/scaled_mm_blockwise_sm90_fp8.cu",
    C + "c3x/scaled_mm_sm90_fp8_dispatch.cuh",
    C + "c3x/scaled_mm_sm90_fp8.cu",
]

ENTRY = C + "scaled_mm_entry.cu"
HELPER = C + "c3x/scaled_mm_helper.hpp"
BW90 = C + "c3x/scaled_mm_blockwise_sm90_fp8_dispatch.cuh"
BW90CU = C + "c3x/scaled_mm_blockwise_sm90_fp8.cu"
FP8D = C + "c3x/scaled_mm_sm90_fp8_dispatch.cuh"
INT8D = C + "c3x/scaled_mm_sm90_int8_dispatch.cuh"
FP8CU = C + "c3x/scaled_mm_sm90_fp8.cu"
MOE90 = C + "moe/grouped_mm_c3x_sm90.cu"
GRP = C + "moe/grouped_mm_c3x.cuh"
STARTS = C + "moe/get_group_starts.cuh"
BIND = "csrc/libtorch_stable/torch_bindings.cpp"
CMOE = "vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py"
FP8U = "vllm/model_executor/layers/quantization/utils/fp8_utils.py"

# (path, line, text the line must contain)
ANCHORS = [
    (ENTRY, 183, "cuda_device_capability >= 100 && cuda_device_capability < 120"),
    (ENTRY, 197, "void cutlass_scaled_mm(torch::stable::Tensor& c"),
    (ENTRY, 203, "a.dim() == 2 && b.dim() == 2 && c.dim() == 2"),
    (ENTRY, 208, "a.stride(1) == 1 && c.stride(1) == 1"),
    (ENTRY, 209, "b.stride(0) == 1"),
    (ENTRY, 210, "c.stride(0) % 16 == 0"),
    (ENTRY, 214, "bias->numel() == b.size(1) && bias->is_contiguous()"),
    (ENTRY, 272, "void cutlass_moe_mm("),
    (ENTRY, 285, "version_num >= 100 && version_num < 110"),
    (ENTRY, 302, "No compiled cutlass_scaled_mm for CUDA device capability"),
    (ENTRY, 417, "bias->scalar_type() == c.scalar_type()"),
    (HELPER, 15, "a_scales.scalar_type() =="),
    (HELPER, 17, "b_scales.scalar_type() =="),
    (HELPER, 22, "a_scales.numel() == 1 || a_scales.numel() == a.size(0)"),
    (HELPER, 25, "a_scales.is_contiguous() && b_scales.is_contiguous()"),
    (HELPER, 26, "a.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn"),
    (HELPER, 29, "a.scalar_type() == torch::headeronly::ScalarType::Char"),
    (HELPER, 39, "} else {"),
    (HELPER, 40, "a scale must be 2d tensor."),
    (HELPER, 41, "b scale must be 2d tensor."),
    (HELPER, 43, "version_num >= 90"),
    (HELPER, 47, "a_scale_group_shape must be [1, 128]."),
    (HELPER, 51, "b_scale_group_shape must be [128, 128]."),
    (HELPER, 54, "Bias not yet supported blockwise scaled_mm"),
    (HELPER, 55, "blockwise_func(c, a, b, a_scales, b_scales);"),
    (BW90, 58, "using ScaleConfig = conditional_t<swap_ab,"),
    (BW90, 64, "cute::GMMA::Major::MN, cute::GMMA::Major::K>>;"),
    (BW90, 155, "a_stride ="),
    (BW90, 156, "make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1))"),
    (BW90, 158, "make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1))"),
    (BW90, 161, "swap_ab ? cute::make_shape(n, m, 1)"),
    (BW90, 162, ": cute::make_shape(m, n, 1));"),
    (BW90, 164, "LayoutSFA layout_SFA = swap_ab"),
    (BW90, 166, "tile_atom_to_shape_SFA(make_shape(m, n, k, 1))"),
    (BW90, 167, "LayoutSFB layout_SFB = swap_ab"),
    (BW90, 169, "tile_atom_to_shape_SFB(make_shape(m, n, k, 1))"),
    (BW90, 171, "auto a_ptr = static_cast<ElementAB const*>(a.data_ptr());"),
    (BW90, 172, "auto b_ptr = static_cast<ElementAB const*>(b.data_ptr());"),
    (BW90, 210, "bool swap_ab = (a.size(0) % 4) != 0;"),
    (BW90CU, 12, "out.scalar_type() == torch::headeronly::ScalarType::BFloat16"),
    (BW90CU, 17, "out.scalar_type() == torch::headeronly::ScalarType::Half"),
    (FP8D, 291, "a.scalar_type() =="),
    (FP8D, 292, "Float8_e4m3fn"),
    (FP8D, 293, "b.scalar_type() =="),
    (FP8D, 294, "Float8_e4m3fn"),
    (FP8D, 388, "a.scalar_type() =="),
    (FP8D, 389, "Float8_e4m3fn"),
    (FP8D, 390, "b.scalar_type() =="),
    (FP8D, 391, "Float8_e4m3fn"),
    (FP8D, 393, "out.scalar_type() == torch::headeronly::ScalarType::BFloat16"),
    (FP8D, 399, "out.scalar_type() == torch::headeronly::ScalarType::Half"),
    (INT8D, 97, "a.scalar_type() == torch::headeronly::ScalarType::Char"),
    (INT8D, 98, "b.scalar_type() == torch::headeronly::ScalarType::Char"),
    (INT8D, 151, "a.scalar_type() == torch::headeronly::ScalarType::Char"),
    (INT8D, 152, "b.scalar_type() == torch::headeronly::ScalarType::Char"),
    (FP8CU, 12, "a_scales.is_contiguous() && b_scales.is_contiguous()"),
    (FP8CU, 14, "bias->scalar_type() == out.scalar_type()"),
    (MOE90, 118, "No input A tensors provided."),
    (MOE90, 123, "a_tensors.scalar_type() == torch::headeronly::ScalarType::Float8"),
    (MOE90, 126, "b_tensors.scalar_type() == torch::headeronly::ScalarType::Float8"),
    (GRP, 129, "using StrideA = Stride<int64_t, Int<1>, Int<0>>;"),
    (GRP, 131, "using StrideC = typename GemmKernel::InternalStrideC;"),
    (GRP, 133, "problem_sizes_as_shapes ="),
    (GRP, 136, "ProblemShape prob_shape{num_experts, problem_sizes_as_shapes"),
    (GRP, 142, "static_cast<StrideB*>(b_strides.data_ptr())"),
    (GRP, 150, "static_cast<StrideB*>(b_strides.data_ptr())"),
    (GRP, 165, "swap_ab ? per_out_ch : per_act_token"),
    (GRP, 166, "swap_ab ? per_act_token : per_out_ch"),
    (GRP, 167, "static_cast<StrideC*>(c_strides.data_ptr())"),
    (GRP, 169, "static_cast<StrideC*>(c_strides.data_ptr())"),
    (GRP, 182, "CUTLASS_CHECK(gemm_op.can_implement(args));"),
    (STARTS, 24, "a_offsets[expert_id] = a_base_as_int + expert_offset * k;"),
    (STARTS, 25, "b_offsets[expert_id] = b_base_as_int + expert_id * k * n;"),
    (STARTS, 26, "out_offsets[expert_id] = out_base_as_int + expert_offset * n;"),
    (STARTS, 36, "<<<1, num_experts, 0, stream>>>"),
    (STARTS, 61, "a_tensors.scalar_type() =="),
    (STARTS, 70, "expert_offsets.scalar_type() =="),
    (STARTS, 74, "bool per_act_token = a_scales.numel() != 1;"),
    (STARTS, 75, "bool per_out_ch = b_scales.numel() != num_experts;"),
    (STARTS, 83, "__CALL_GET_STARTS_KERNEL(torch::headeronly::ScalarType::BFloat16"),
    (STARTS, 87, "Invalid output type (must be float16 or bfloat16)"),
    (BIND, 114, "cutlass_scaled_mm(Tensor! out, Tensor a,"),
    (BIND, 136, "cutlass_moe_mm(Tensor! out_tensors"),
    (CMOE, 223, "ops.cutlass_moe_mm("),
    (CMOE, 249, "ops.cutlass_moe_mm("),
    (CMOE, 298, "ab_strides1_c_strides2 = torch.full((e,), k"),
    (CMOE, 306, "self.c_strides2 = ab_strides1_c_strides2"),
    (CMOE, 1283, "a_strides1_c_strides2 = torch.full((e,), k"),
    (CMOE, 1287, "self.c_strides2 = a_strides1_c_strides2"),
    (LIN + "cutlass.py", 285, "column_major_scales=True,"),
    (LIN + "cutlass.py", 321, "return ops.cutlass_scaled_mm("),
    (LIN + "cutlass.py", 326, "scale_b=Bs.T,"),
    (LIN + "flashinfer.py", 282, "column_major_scales=True,"),
    (LIN + "deep_gemm.py", 43, "column_major_scales=True,"),
    (LIN + "pytorch.py", 288, "column_major_scales=current_platform.is_cuda_alike()"),
    (FP8U, 556, "column_major_scales: bool = False,"),
    (TQ + "test_cutlass_scaled_mm.py", 103, "make scales M-major for blockwise quant"),
    (TQ + "test_cutlass_scaled_mm.py", 105, "make scales K-major for blockwise quant"),
    (TQ + "test_cutlass_scaled_mm.py", 115, "opcheck(torch.ops._C.cutlass_scaled_mm,"),
    (TQ + "test_cutlass_scaled_mm.py", 146, "opcheck(torch.ops._C.cutlass_scaled_mm,"),
    (TQ + "test_cutlass_scaled_mm.py", 179, "def test_cutlass_fp8_gemm_padded("),
    (TQ + "test_cutlass_scaled_mm.py", 576, "def test_cutlass_subset():"),
    (TQ + "test_cutlass_scaled_mm.py", 582, "a = whole_a[0:m, 0:k]"),
    (TQ + "test_cutlass_scaled_mm.py", 655, "def test_cutlass_fp8_group_gemm("),
    (TQ + "test_block_fp8.py", 200, "CUTLASS uses column-major format for scales"),
    (TQ + "test_block_fp8.py", 202, "column_major_scales=True"),
]


def git_blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def fetch(path: str) -> bytes:
    if shutil.which("gh"):
        done = subprocess.run(
            [
                "gh",
                "api",
                f"repos/vllm-project/vllm/contents/{path}?ref={PIN}",
                "-H",
                "Accept: application/vnd.github.raw",
            ],
            capture_output=True,
        )
        if done.returncode == 0:
            return done.stdout
    url = f"https://raw.githubusercontent.com/vllm-project/vllm/{PIN}/{path}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def check_pinned() -> int:
    failures = 0
    files: dict[str, list[str]] = {}
    for path, expected_blob in BLOBS.items():
        data = fetch(path)
        blob = git_blob_id(data)
        status = "OK  " if blob == expected_blob else "FAIL"
        failures += blob != expected_blob
        print(f"{status} blob {blob[:12]} {path}")
        files[path] = data.decode("utf-8").split("\n")
    for path, line, text in ANCHORS:
        lines = files[path]
        actual = lines[line - 1] if 0 < line <= len(lines) else "<no such line>"
        ok = text in actual
        failures += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {path.split('/')[-1]}:L{line}  {text}")
        if not ok:
            print(f"      actual: {actual.strip()}")
    print(f"{len(BLOBS)} blobs, {len(ANCHORS)} anchors, {failures} failures at {PIN}")
    return 1 if failures else 0


def check_local(root: Path) -> int:
    failures = 0
    for path in RUN_GATE:
        target = root / path
        blob = git_blob_id(target.read_bytes()) if target.is_file() else "<missing>"
        ok = blob == BLOBS[path]
        failures += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {blob[:12]} {path}")
    print(f"{len(RUN_GATE)} gated files, {failures} mismatches in {root}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--local", type=Path, help="local vLLM checkout to gate")
    args = parser.parse_args()
    return check_local(args.local) if args.local else check_pinned()


if __name__ == "__main__":
    sys.exit(main())
