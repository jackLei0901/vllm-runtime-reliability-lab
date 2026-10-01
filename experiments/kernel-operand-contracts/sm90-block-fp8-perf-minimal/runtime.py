"""GPU helpers; imports of Torch/NVML are deferred until an explicit GPU run."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import build_perf
import protocol

MANIFEST = Path(__file__).with_name("FREEZE_MANIFEST.json")
FREEZE_RECEIPT_ENV = "SM90_PUBLIC_FREEZE_RECEIPT"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fresh(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def packet_hashes():
    root = Path(__file__).parent
    return {p.name: sha(p) for p in sorted(root.glob("*")) if p.is_file()}


def local_freeze(commit):
    import freeze_packet

    if not commit or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("full public freeze commit required")
    local = MANIFEST.read_bytes()
    data = json.loads(local)
    if data["files_sha256"] != freeze_packet.entries():
        raise ValueError("local packet differs from freeze manifest")
    return local, {
        "commit": commit,
        "manifest_sha256": hashlib.sha256(local).hexdigest(),
    }


def freeze_url(commit):
    return (
        "https://raw.githubusercontent.com/jackLei0901/"
        "vllm-runtime-reliability-lab/"
        + commit
        + "/experiments/kernel-operand-contracts/sm90-block-fp8-perf-minimal/"
        "FREEZE_MANIFEST.json"
    )


def verify_public_freeze(commit):
    """The sole online operation; run before expensive work."""
    local, identity = local_freeze(commit)
    url = freeze_url(commit)
    with urllib.request.urlopen(url, timeout=15) as response:
        public = response.read()
    if public != local:
        raise ValueError("public/local freeze bytes differ")
    if local_freeze(commit)[0] != local:
        raise ValueError("packet changed during public verification")
    return {
        "status": "verified_public_freeze",
        "schema_version": 1,
        **identity,
        "url": url,
        "verified_at_unix_ns": time.time_ns(),
    }


def public_freeze(commit, receipt_path=None):
    """Offline only; never silently falls back to a network fetch."""
    _, identity = local_freeze(commit)
    receipt_path = receipt_path or os.environ.get(FREEZE_RECEIPT_ENV)
    if not receipt_path:
        raise ValueError("verified public-freeze receipt required; no network fallback")
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "verified_public_freeze"
        or receipt.get("schema_version") != 1
        or receipt.get("commit") != identity["commit"]
        or receipt.get("manifest_sha256") != identity["manifest_sha256"]
        or receipt.get("url") != freeze_url(commit)
        or type(receipt.get("verified_at_unix_ns")) is not int
        or receipt["verified_at_unix_ns"] <= 0
    ):
        raise ValueError("public-freeze receipt differs from local packet/commit")
    return identity


def freeze_preflight():
    parser = argparse.ArgumentParser(
        description="Verify GitHub once and save a private receipt"
    )
    parser.add_argument("--freeze-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("freeze receipt already exists; preserve first attempt")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = {"status": "unscored", "commit": args.freeze_commit}
    try:
        result = verify_public_freeze(args.freeze_commit)
    except Exception as exc:  # noqa: BLE001 - retain first network failure
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    with args.out.open("xb") as file:
        file.write(
            (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
        )
    print(json.dumps({"status": result["status"], "commit": args.freeze_commit}))
    if result["status"] != "verified_public_freeze":
        raise SystemExit(1)


def verify_build(path, arm):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if data.get("status") != "built_not_validated":
        raise ValueError("successful build receipt required")
    if data["source"]["head"] != build_perf.PIN or data["source"]["arm"] != arm:
        raise ValueError("source arm/pin mismatch")
    if data["cutlass"]["head"] != build_perf.CUTLASS_PIN:
        raise ValueError("CUTLASS pin mismatch")
    namespace = build_perf.validate_namespace(data["namespace"])
    if namespace != data["binary"]["namespace"]:
        raise ValueError("namespace mismatch")
    for name, expected in data["harness_sha256"].items():
        if Path(name).name != name or sha(Path(__file__).with_name(name)) != expected:
            raise ValueError("builder input identity mismatch")
    binary_name = data["binary"]["binary_file"]
    if Path(binary_name).name != binary_name:
        raise ValueError("binary must be relative to receipt")
    binary = path.parent / binary_name
    if sha(binary) != data["binary"]["binary_sha256"]:
        raise ValueError("binary identity mismatch")
    return data, binary


def verify_pair(base, variant):
    if base["namespace"] == variant["namespace"]:
        raise ValueError("distinct namespaces required")
    for key in ("cutlass", "cmake_reference", "harness_sha256", "public_freeze"):
        if base[key] != variant[key]:
            raise ValueError(f"build pair differs in {key}")
    left, right = base["source"]["csrc_sha256"], variant["source"]["csrc_sha256"]
    if set(left) != set(right):
        raise ValueError("source sets differ")
    if {k for k in left if left[k] != right[k]} != {build_perf.DISPATCH}:
        raise ValueError("pair must differ only in the declared dispatch header")
    if base["binary"]["torch"] != variant["binary"]["torch"]:
        raise ValueError("Torch builds differ")


def gpu():
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("exactly one visible GPU required")
    if torch.cuda.get_device_capability() != (9, 0):
        raise ValueError("SM90 required")
    torch.backends.cuda.matmul.allow_tf32 = False
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    driver.cuDeviceGetAttribute.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
    ]
    device, cache = ctypes.c_int(), ctypes.c_int()
    if driver.cuInit(0) or driver.cuDeviceGet(ctypes.byref(device), 0):
        raise ValueError("CUDA driver query failed")
    # CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38, queried on the actual device.
    if driver.cuDeviceGetAttribute(ctypes.byref(cache), 38, device.value):
        raise ValueError("L2 query failed")
    if cache.value <= 0:
        raise ValueError("missing positive L2 size")
    props = torch.cuda.get_device_properties(0)
    driver_version = ctypes.c_int()
    if driver.cuDriverGetVersion(ctypes.byref(driver_version)):
        raise ValueError("CUDA driver version query failed")
    toolkit = subprocess.check_output(["nvcc", "--version"], encoding="utf-8").strip()
    nvidia_driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        encoding="utf-8",
    ).strip()
    return torch, {
        "name": props.name,
        "sm_count": props.multi_processor_count,
        "l2_bytes": cache.value,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "uuid": str(props.uuid),
        "cuda_driver_version": driver_version.value,
        "nvcc_version": toolkit,
        "nvidia_driver": nvidia_driver,
    }


def compatible_gpu(left, right):
    """Cross-session only: same hardware/toolchain, physical UUID may differ."""
    keys = (
        "name",
        "sm_count",
        "l2_bytes",
        "torch",
        "torch_cuda",
        "cuda_driver_version",
        "nvcc_version",
        "nvidia_driver",
    )
    if any(k not in left or k not in right or left[k] != right[k] for k in keys):
        raise ValueError("cross-session GPU model/driver/toolchain differs")
    return left.get("uuid") != right.get("uuid")


def load(torch, receipt, binary):
    if torch.__version__ != receipt["binary"]["torch"]:
        raise ValueError("Torch runtime differs from build")
    torch.ops.load_library(str(binary))
    return getattr(torch.ops, receipt["namespace"]).scaled_mm


def tensors(torch, m, n, k, dtype, seed=730, device="cuda"):
    generator = torch.Generator(device=device).manual_seed(seed)
    weights = torch.Generator(device=device).manual_seed(seed + 10000)
    a = (torch.rand(m, k, device=device, generator=generator) * 2 - 1).to(
        torch.float8_e4m3fn
    )
    b = (
        (torch.rand(n, k, device=device, generator=weights) * 2 - 1)
        .to(torch.float8_e4m3fn)
        .t()
    )
    sa = (
        10 ** (torch.rand(k // 128, m, device=device, generator=generator) * 2 - 2)
    ).t()
    sb = (
        10 ** (torch.rand(n // 128, k // 128, device=device, generator=weights) * 2 - 2)
    ).t()
    out = torch.empty(m, n, device=device, dtype=dtype)
    return out, a, b, sa, sb


def reference(torch, values):
    _, a, b, sa, sb = values
    # Independently dequantize before FP32 matmul, not the candidate kernel.
    return (a.float() * sa.repeat_interleave(128, 1)) @ (
        b.float() * sb.repeat_interleave(128, 0).repeat_interleave(128, 1)
    )


def relative_error(torch, out, expected):
    if not torch.isfinite(out).all() or not torch.isfinite(expected).all():
        raise ValueError("nonfinite numerical output")
    denominator = expected.norm().item()
    if denominator <= 0:
        raise ValueError("zero reference norm")
    return ((out.float() - expected).norm().item()) / denominator


class Monitor:
    """NVML sampling only; no clock or power setting is changed."""

    def __init__(self, torch):
        import pynvml

        self.nv = pynvml
        self.nv.nvmlInit()
        uuid = str(torch.cuda.get_device_properties(0).uuid)
        if not uuid.startswith("GPU-"):
            uuid = "GPU-" + uuid
        self.handle = self.nv.nvmlDeviceGetHandleByUUID(uuid)
        self.uuid = self.nv.nvmlDeviceGetUUID(self.handle)
        if isinstance(self.uuid, bytes):
            self.uuid = self.uuid.decode("ascii")
        self.samples = []
        self.failure = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()

    def sample(self):
        nv, h = self.nv, self.handle
        return {
            "time": time.monotonic(),
            "sm_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
            "memory_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM),
            "temperature": nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU),
            "power_mw": nv.nvmlDeviceGetPowerUsage(h),
            "limit_mw": nv.nvmlDeviceGetPowerManagementLimit(h),
            "reasons": nv.nvmlDeviceGetCurrentClocksThrottleReasons(h),
        }

    def _poll(self):
        try:
            while not self.stop.is_set():
                self.samples.append(self.sample())
                self.stop.wait(0.005)
        except Exception as exc:  # noqa: BLE001 - apparatus failure is unscored
            self.failure = type(exc).__name__

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        self.nv.nvmlShutdown()


def clock_valid(samples, reference_clock, fixed_limit):
    if not samples or reference_clock <= 0:
        return False
    # NVML idle=1 and steady software power cap=4 are permitted; all other
    # reasons fail closed. Raw flags and endpoints are retained for review.
    return all(
        s["sm_mhz"] > 0
        and not (s["reasons"] & ~5)
        and s["limit_mw"] == fixed_limit
        and abs(s["sm_mhz"] / reference_clock - 1) <= protocol.CLOCK_DEVIATION
        for s in samples
    )


if __name__ == "__main__":
    freeze_preflight()
