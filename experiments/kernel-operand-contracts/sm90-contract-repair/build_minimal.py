"""Build the pinned real SM90 path, not a copied validation implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
from pathlib import Path

PIN = "7b054aca96cea8be1369d651c3434ad140580b92"
CUTLASS_TAG = "v4.7.1"
ROOT = "csrc/libtorch_stable/quantization/w8a8/cutlass/"
TARGET = ROOT + "c3x/scaled_mm_blockwise_sm90_fp8.cu"
SOURCES = [
    ROOT + "scaled_mm_entry.cu",
    ROOT + "scaled_mm_c3x_sm90.cu",
    ROOT + "c3x/scaled_mm_sm90_fp8.cu",
    ROOT + "c3x/scaled_mm_sm90_int8.cu",
    ROOT + "c3x/scaled_mm_azp_sm90_int8.cu",
    TARGET,
    "csrc/libtorch_stable/cutlass_extensions/common.cpp",
]
ANCHOR = "    torch::stable::Tensor const& b_scales) {\n"
GUARDS = """  STD_TORCH_CHECK(
      a.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn,
                  "SM90 blockwise scaled_mm requires A dtype float8_e4m3fn");
  STD_TORCH_CHECK(
      b.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn,
                  "SM90 blockwise scaled_mm requires B dtype float8_e4m3fn");
  STD_TORCH_CHECK(
      (a_scales.size(0) <= 1 || a_scales.stride(0) == 1) &&
          (a_scales.size(1) <= 1 || a_scales.stride(1) == a_scales.size(0)),
      "SM90 blockwise scaled_mm requires packed column-major A scales");
  STD_TORCH_CHECK(
      (b_scales.size(0) <= 1 || b_scales.stride(0) == 1) &&
          (b_scales.size(1) <= 1 || b_scales.stride(1) == b_scales.size(0)),
      "SM90 blockwise scaled_mm requires packed K-major B scales");
"""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(src: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(src), *args], encoding="utf-8"
    ).strip()


def proposed_fix(base: str) -> str:
    if base.count(ANCHOR) != 1:
        raise ValueError("guard insertion anchor is not unique")
    return base.replace(ANCHOR, ANCHOR + GUARDS, 1)


def validate_source(src: Path, arm: str) -> dict:
    if git(src, "rev-parse", "HEAD") != PIN:
        raise ValueError("wrong source HEAD; do not use an old worktree")
    dirty = git(src, "status", "--porcelain", "--untracked-files=all")
    expected_status = "" if arm == "base" else "M " + TARGET
    if dirty != expected_status:
        raise ValueError("source must be clean base or exactly the unstaged fix")
    base = subprocess.check_output(["git", "-C", str(src), "show", f"HEAD:{TARGET}"])
    expected = base.decode() if arm == "base" else proposed_fix(base.decode())
    if (src / TARGET).read_text(encoding="utf-8") != expected:
        raise ValueError("target source does not match the declared arm")
    files = git(src, "ls-files", "csrc").splitlines()
    hashes = {name: digest((src / name).read_bytes()) for name in files}
    return {"head": PIN, "arm": arm, "csrc_sha256": hashes}


def validate_cutlass(src: Path) -> dict:
    if git(src, "rev-parse", "HEAD") != git(
        src, "rev-parse", f"{CUTLASS_TAG}^{{commit}}"
    ):
        raise ValueError("CUTLASS HEAD must match v4.7.1")
    if git(src, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("CUTLASS must be clean")
    files = git(src, "ls-files", "include", "tools/util/include").splitlines()
    return {
        "head": git(src, "rev-parse", "HEAD"),
        "tag": CUTLASS_TAG,
        "headers_sha256": {name: digest((src / name).read_bytes()) for name in files},
    }


def compile_extension(src: Path, cutlass: Path, out: Path) -> dict:
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, load

    if sys.platform != "linux":
        raise ValueError("Linux build required")
    if torch.__version__.split("+")[0] != "2.13.0":
        raise ValueError("torch 2.13.0 required by pinned build configuration")
    if CUDA_HOME is None or not (Path(CUDA_HOME) / "bin/nvcc").is_file():
        raise ValueError("CUDA_HOME must contain nvcc")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    os.environ["MAX_JOBS"] = "2"
    flags = [
        "-DENABLE_SCALED_MM_SM90=1",
        "-DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1",
        "-std=c++17",
        "-O3",
    ]
    library = load(
        name="lab_sm90_contract",
        sources=[str(src / name) for name in SOURCES]
        + [str(Path(__file__).with_name("binding.cpp"))],
        extra_include_paths=[
            str(src / "csrc"),
            str(src / "csrc/libtorch_stable/quantization/w8a8/cutlass"),
            str(cutlass / "include"),
            str(cutlass / "tools/util/include"),
        ],
        extra_cflags=flags,
        extra_cuda_cflags=flags + ["--expt-relaxed-constexpr"],
        extra_ldflags=[f"-L{CUDA_HOME}/lib64/stubs", "-lcuda"],
        build_directory=str(out),
        is_python_module=False,
        verbose=True,
    )
    binary = Path(library)
    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "binary": str(binary.resolve()),
        "binary_file": binary.name,
        "binary_sha256": digest(binary.read_bytes()),
        "build_ninja_sha256": digest((out / "build.ninja").read_bytes()),
        "scope": "real SM90 entry/dispatch/kernel; not full serving",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-src", type=Path, required=True)
    parser.add_argument("--cutlass-src", type=Path)
    parser.add_argument("--arm", choices=["base", "fix"], required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=2700)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.timeout_seconds <= 2700:
        parser.error("build timeout must be 1..2700 seconds")
    src = args.vllm_src.resolve()
    source = validate_source(src, args.arm)
    if args.plan:
        print(json.dumps({"arm": args.arm, "head": PIN, "sources": SOURCES}))
        return
    if args.out is None or args.cutlass_src is None:
        parser.error("build requires --out and --cutlass-src")
    if args.worker:
        result = compile_extension(src, args.cutlass_src.resolve(), args.out.resolve())
        (args.out / "binary.json").write_text(json.dumps(result, indent=2) + "\n")
        return
    cutlass = validate_cutlass(args.cutlass_src.resolve())
    args.out.mkdir(parents=True, exist_ok=False)
    receipt = {
        "status": "unscored",
        "source": source,
        "cutlass": cutlass,
        "build_host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "libc": platform.libc_ver(),
            "python": sys.version,
            "os_release": (
                Path("/etc/os-release").read_text()
                if Path("/etc/os-release").is_file()
                else None
            ),
        },
        "harness_sha256": {
            name: digest(Path(__file__).with_name(name).read_bytes())
            for name in (
                "build_minimal.py",
                "binding.cpp",
                "candidate.patch",
                "test_repair_sm90.py",
                "run_isolated.py",
                "pytest.ini",
            )
        },
        "build_timeout_seconds": args.timeout_seconds,
    }
    try:
        with (args.out / "build.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    *sys.argv[1:],
                    "--worker",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=args.timeout_seconds)
            except subprocess.TimeoutExpired:
                if sys.platform == "linux":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
                raise
            if returncode:
                raise subprocess.CalledProcessError(returncode, process.args)
        if validate_source(src, args.arm) != source:
            raise ValueError("source changed during build")
        if validate_cutlass(args.cutlass_src.resolve()) != cutlass:
            raise ValueError("CUTLASS changed during build")
        receipt["status"] = "built_not_validated"
        receipt["binary"] = json.loads((args.out / "binary.json").read_text())
    except (subprocess.SubprocessError, ValueError) as exc:
        receipt["failure"] = type(exc).__name__
    finally:
        (args.out / "build_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps({"arm": args.arm, "status": receipt["status"]}))
    if receipt["status"] != "built_not_validated":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
