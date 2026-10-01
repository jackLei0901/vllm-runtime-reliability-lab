"""Performance successor of R2; historical contract builders remain unchanged."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PIN = "7b054aca96cea8be1369d651c3434ad140580b92"


def remaining_timeout(cap, deadline_utc, now=None):
    if deadline_utc is None:
        return cap
    end = datetime.fromisoformat(deadline_utc.replace("Z", "+00:00"))
    if end.tzinfo is None:
        raise ValueError("shared deadline requires timezone")
    remaining = math.floor((end - (now or datetime.now(timezone.utc))).total_seconds())
    if remaining < 1:
        raise TimeoutError("shared build window exhausted")
    return min(cap, remaining)


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
DISPATCH = ROOT + "c3x/scaled_mm_blockwise_sm90_fp8_dispatch.cuh"
CUTLASS_PIN = "cb4247394dd82148787aed73e5dc7cef33cbf862"
DISPATCH_ANCHOR = "bool swap_ab = (a.size(0) % 4) != 0;"
DISPATCH_VARIANT = "bool swap_ab = a.size(0) <= 64 || (a.size(0) % 4) != 0;"


def validate_namespace(namespace: str) -> str:
    if not re.fullmatch(r"lab_sm90_perf_[a-z][a-z0-9_]{0,31}", namespace):
        raise ValueError("namespace must be a safe lab_sm90_perf_ identifier")
    return namespace


def render_binding(template: str, namespace: str) -> str:
    validate_namespace(namespace)
    if template.count("lab_sm90_perf_template") != 2:
        raise ValueError("binding needs exactly two registration namespace anchors")
    return template.replace("lab_sm90_perf_template", namespace)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(src: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(src), *args], encoding="utf-8"
    ).strip()


def proposed_variant(base: str) -> str:
    if base.count(DISPATCH_ANCHOR) != 1:
        raise ValueError("dispatch anchor is not unique")
    return base.replace(DISPATCH_ANCHOR, DISPATCH_VARIANT, 1)


def validate_source(src: Path, arm: str) -> dict:
    if git(src, "rev-parse", "HEAD") != PIN:
        raise ValueError("wrong source HEAD; do not use an old worktree")
    dirty = git(src, "status", "--porcelain", "--untracked-files=all")
    expected_status = "" if arm == "base" else "M " + DISPATCH
    if dirty != expected_status:
        raise ValueError("source must be clean base or exactly the unstaged fix")
    base = subprocess.check_output(["git", "-C", str(src), "show", f"HEAD:{DISPATCH}"])
    expected = base.decode() if arm == "base" else proposed_variant(base.decode())
    if (src / DISPATCH).read_text(encoding="utf-8") != expected:
        raise ValueError("target source does not match the declared arm")
    files = git(src, "ls-files", "csrc").splitlines()
    hashes = {name: digest((src / name).read_bytes()) for name in files}
    return {"head": PIN, "arm": arm, "csrc_sha256": hashes}


def validate_cutlass(src: Path) -> dict:
    if git(src, "rev-parse", "HEAD") != git(
        src, "rev-parse", f"{CUTLASS_TAG}^{{commit}}"
    ):
        raise ValueError("CUTLASS HEAD must match v4.7.1")
    if git(src, "rev-parse", "HEAD") != CUTLASS_PIN:
        raise ValueError("CUTLASS tag must also match the recorded commit")
    if git(src, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("CUTLASS must be clean")
    files = git(src, "ls-files", "include", "tools/util/include").splitlines()
    return {
        "head": git(src, "rev-parse", "HEAD"),
        "tag": CUTLASS_TAG,
        "headers_sha256": {name: digest((src / name).read_bytes()) for name in files},
    }


def validate_reference(reference: Path) -> dict:
    commands = reference / "compile_commands.json"
    rows = [
        row for row in json.loads(commands.read_text()) if row["file"].endswith(TARGET)
    ]
    if len(rows) != 1:
        raise ValueError("expected one real blockwise CMake reference")
    for flag in (
        "-std=c++20",
        "-DTORCH_TARGET_VERSION=0x020B000000000000ULL",
        "-DUSE_CUDA",
        "-DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1",
        "-DENABLE_SCALED_MM_SM90=1",
        "arch=compute_90a,code=sm_90a",
    ):
        if flag not in rows[0]["command"]:
            raise ValueError(f"missing reference flag: {flag}")
    patched = reference / "torch_patched_headers"
    headers = sorted(patched.rglob("*.h"))
    if not headers:
        raise ValueError("missing same-pin patched torch header")
    return {
        "command": rows[0],
        "compile_commands_sha256": digest(commands.read_bytes()),
        "patched_headers_sha256": {
            str(path.relative_to(patched)): digest(path.read_bytes())
            for path in headers
        },
    }


def compile_extension(
    src: Path, cutlass: Path, out: Path, reference: Path, namespace: str
) -> dict:
    validate_namespace(namespace)
    import torch
    import torch.utils.cpp_extension as cpp_extension
    from torch.utils.cpp_extension import CUDA_HOME, load

    if sys.platform != "linux":
        raise ValueError("Linux build required")
    if torch.__version__.split("+")[0] != "2.13.0":
        raise ValueError("torch 2.13.0 required by pinned build configuration")
    if CUDA_HOME is None or not (Path(CUDA_HOME) / "bin/nvcc").is_file():
        raise ValueError("CUDA_HOME must contain nvcc")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    os.environ["MAX_JOBS"] = "2"
    binding = out / "binding_generated.cpp"
    binding.write_text(
        render_binding(Path(__file__).with_name("binding.cpp").read_text(), namespace),
        encoding="utf-8",
    )
    # Match pinned cmake/utils.cmake for CUDA >= 12.0, rather than retaining
    # cpp_extension's half-conversion disabling definitions.
    excluded = {
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    }
    cpp_extension.COMMON_NVCC_FLAGS = [
        flag for flag in cpp_extension.COMMON_NVCC_FLAGS if flag not in excluded
    ]
    flags = [
        "-DENABLE_SCALED_MM_SM90=1",
        "-DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1",
        "-DTORCH_TARGET_VERSION=0x020B000000000000ULL",
        "-DUSE_CUDA",
        "-DPy_LIMITED_API=3",
        "-DNDEBUG",
        "-std=c++20",
        "-O3",
    ]
    library = load(
        name=namespace,
        sources=[str(src / name) for name in SOURCES] + [str(binding)],
        extra_include_paths=[
            str(reference / "torch_patched_headers"),
            str(src / "csrc"),
            str(src / "csrc/libtorch_stable/quantization/w8a8/cutlass"),
            str(cutlass / "include"),
            str(cutlass / "tools/util/include"),
        ],
        extra_cflags=flags,
        extra_cuda_cflags=flags
        + [
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-DENABLE_FP8",
            "--compress-mode=size",
        ],
        extra_ldflags=[f"-L{CUDA_HOME}/lib64/stubs", "-lcuda", "-Wl,-Bsymbolic"],
        build_directory=str(out),
        is_python_module=False,
        verbose=True,
    )
    binary = Path(library)
    return {
        "namespace": namespace,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "binary": str(binary.resolve()),
        "binary_file": binary.name,
        "binary_sha256": digest(binary.read_bytes()),
        "binding_generated_sha256": digest(binding.read_bytes()),
        "build_ninja_sha256": digest((out / "build.ninja").read_bytes()),
        "scope": "real SM90 entry/dispatch/kernel; not full serving",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-src", type=Path, required=True)
    parser.add_argument("--cutlass-src", type=Path)
    parser.add_argument("--cmake-reference", type=Path)
    parser.add_argument("--arm", choices=["base", "variant"], required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--freeze-commit")
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--deadline-utc")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.timeout_seconds <= 1200:
        parser.error("build timeout must be 1..1200 seconds")
    try:
        validate_namespace(args.namespace)
    except ValueError as exc:
        parser.error(str(exc))
    src = args.vllm_src.resolve()
    source = validate_source(src, args.arm)
    if args.plan:
        print(
            json.dumps(
                {
                    "arm": args.arm,
                    "head": PIN,
                    "namespace": args.namespace,
                    "sources": SOURCES,
                }
            )
        )
        return
    if args.out is None or args.cutlass_src is None or args.cmake_reference is None:
        parser.error("build requires --out, --cutlass-src and --cmake-reference")
    reference = validate_reference(args.cmake_reference.resolve())
    if args.worker:
        result = compile_extension(
            src,
            args.cutlass_src.resolve(),
            args.out.resolve(),
            args.cmake_reference.resolve(),
            args.namespace,
        )
        (args.out / "binary.json").write_text(json.dumps(result, indent=2) + "\n")
        return
    cutlass = validate_cutlass(args.cutlass_src.resolve())
    from runtime import public_freeze

    freeze = public_freeze(args.freeze_commit)
    args.out.mkdir(parents=True, exist_ok=False)
    receipt = {
        "status": "unscored",
        "public_freeze": freeze,
        "source": source,
        "cutlass": cutlass,
        "cmake_reference": reference,
        "apparatus_revision": "perf-R2 successor; no GPU validation yet",
        "namespace": args.namespace,
        "dispatch_definition_sha256": digest(DISPATCH_VARIANT.encode()),
        "link_isolation": "-Wl,-Bsymbolic in both arms; load check pending",
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
                Path(__file__).name,
                "binding.cpp",
                "protocol.py",
                "runtime.py",
                "correctness.py",
                "timing.py",
                "identity_probe.py",
                "trace_tools.py",
                "model_preflight.py",
                "serve_trace.py",
                "witness_review.py",
                "freeze_packet.py",
            )
        },
        "build_timeout_seconds": args.timeout_seconds,
        "shared_deadline_utc": args.deadline_utc,
    }
    try:
        timeout = remaining_timeout(args.timeout_seconds, args.deadline_utc)
        receipt["effective_timeout_seconds"] = timeout
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
                returncode = process.wait(timeout=timeout)
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
        if validate_reference(args.cmake_reference.resolve()) != reference:
            raise ValueError("CMake reference changed during build")
        remaining_timeout(args.timeout_seconds, args.deadline_utc)
        receipt["status"] = "built_not_validated"
        receipt["binary"] = json.loads((args.out / "binary.json").read_text())
    except (subprocess.SubprocessError, ValueError, TimeoutError) as exc:
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
