"""Offline admission and launch. Use --dry-run before paying for a GPU."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from collect import PIN, REVISION, launch_env
from preflight import check_public, sha, validate_identity, write_new


def validate_preparation(prep):
    if prep.get("pin") != PIN or prep.get("revision") != REVISION:
        raise ValueError("preparation pin/model revision differs")
    for key in ("installed_verified", "source_verified", "weight_hashes_verified"):
        if prep.get(key) is not True:
            raise ValueError(f"incomplete preparation: {key}")
    expected = {
        "vllm": "0.30.1rc1.dev618+gb0e21b308",
        "torch": "2.13.0+cu130",
        "flashinfer-python": "0.7.0.post1",
        "flashinfer-cubin": "0.7.0.post1",
    }
    if prep.get("installed_versions") != expected:
        raise ValueError("prepared package versions differ")
    for package, version in expected.items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"installed version differs: {package}")
    model = Path(prep["model_directory"])
    if sha(model / "config.json") != prep["config_sha256"]:
        raise ValueError("model config differs")
    for name, size in prep["weight_file_sizes"].items():
        if (model / name).stat().st_size != size:
            raise ValueError(f"weight size differs: {name}")
    return model


def validate_idle(gpu, processes):
    fields = [field.strip() for field in gpu.split(",")]
    if len(fields) != 5 or "H800" not in fields[0] or processes.strip():
        raise ValueError("GPU is not a single idle H800")
    if int(fields[-1].split()[0]) > 128:
        raise ValueError("GPU memory is not idle")


def output(*args):
    return subprocess.check_output(args, text=True, timeout=30).strip()


def cleanup_server(work):
    """Collector timeout must not leave its separate server group alive."""
    receipt = Path(work) / "collection" / "collection.json"
    if not receipt.exists():
        return
    group = json.loads(receipt.read_text()).get("private_process_group")
    if type(group) is not int or group <= 1:
        return
    try:
        os.killpg(group, signal.SIGTERM)
        until = time.monotonic() + 10
        while time.monotonic() < until:
            time.sleep(0.2)
            os.killpg(group, 0)
        os.killpg(group, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--public-receipt", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--session-start-monotonic", type=float)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    try:
        public = check_public(args.packet_root, args.freeze, args.public_receipt)
        for name in ("admit.py", "preflight.py", "collect.py"):
            if (
                sha(Path(__file__).with_name(name))
                != public["files"]["experiments/native-kernel-discovery/" + name]
            ):
                raise ValueError("executed tool differs from public freeze")
        prep = json.loads(args.preparation.read_text())
        validate_identity(json.loads(args.identity.read_text()), 5305)
        model = validate_preparation(prep)
        env = launch_env(os.environ)
        if (
            Path(prep["isolated_environment_created"]).resolve()
            != Path(sys.prefix).resolve()
        ):
            raise ValueError("interpreter differs from prepared environment")
        # No torch import or GPU inventory in this branch.
        write_new(
            args.work / "offline-checks.json",
            {
                "freeze": args.freeze,
                "public_receipt_sha256": sha(args.public_receipt),
                "preparation_sha256": sha(args.preparation),
                "identity_sha256": sha(args.identity),
                "status": "offline_checks_passed",
                "gpu_runtime_tested": False,
            },
        )
        if args.dry_run:
            print("OFFLINE_CHECKS_PASS; GPU admission not performed")
            return 0
        start = args.session_start_monotonic
        if start is None or not 0 <= time.monotonic() - start <= 300:
            raise ValueError("fresh session start required; do not reset the clock")
        gpu = output(
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,memory.used",
            "--format=csv,noheader",
        )
        processes = output(
            "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"
        )
        validate_idle(gpu, processes)
        # Probe capability in a short-lived child; its context exits before collection.
        capability = output(
            sys.executable,
            "-c",
            "import torch; print(torch.cuda.get_device_capability(0))",
        )
        if capability != "(9, 0)":
            raise ValueError("SM90 capability not confirmed")
        import shutil

        free = {path: shutil.disk_usage(path).free for path in ("/", str(args.work))}
        memory = int(Path("/sys/fs/cgroup/memory.max").read_text())
        if memory < 32 * 1024**3 or min(free.values()) < 2 * 1024**3:
            raise ValueError("insufficient memory/disk headroom")
        # Recheck after the capability subprocess has released its CUDA context.
        validate_idle(
            output(
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total,memory.used",
                "--format=csv,noheader",
            ),
            output("nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"),
        )
        if time.monotonic() - start > 300:
            raise ValueError("five-minute admission window exhausted")
        extensions = {
            str(path): sha(path)
            for path in (Path(sys.prefix) / "lib/python3.12/site-packages/vllm").glob(
                "_C*.so"
            )
        }
        if not extensions:
            raise ValueError("native extension files missing")
        admission = {
            "pin": PIN,
            "revision": REVISION,
            "freeze": args.freeze,
            "public_receipt_sha256": sha(args.public_receipt),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "session_started_monotonic": start,
            "python": sys.executable,
            "model_dir": str(model),
            "gpu": gpu,
            "capability": capability,
            "free_bytes": free,
            "memory_limit": memory,
            "native_extension_files_sha256": extensions,
            "runtime_load_verified": False,
            "wheel_verified": True,
            "source_verified": True,
            "installed_verified": True,
            "model_verified": True,
            "resources_verified": True,
            "gpu_idle_verified": True,
        }
        write_new(args.work / "admission.json", admission)
        env.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "VLLM_CACHE_ROOT": str(args.work / "vllm-cache"),
                "TORCHINDUCTOR_CACHE_DIR": str(args.work / "inductor-cache"),
            }
        )
        command = [
            sys.executable,
            str(Path(__file__).with_name("collect.py")),
            "--python",
            sys.executable,
            "--model-dir",
            str(model),
            "--preflight",
            str(args.work / "admission.json"),
            "--out",
            str(args.work / "collection"),
            "--freeze",
            args.freeze,
            "--public-receipt",
            str(args.public_receipt),
            "--packet-root",
            str(args.packet_root),
        ]
        return subprocess.run(
            command, env=env, timeout=max(1, start + 58 * 60 - time.monotonic())
        ).returncode
    except Exception as exc:
        if isinstance(exc, subprocess.TimeoutExpired):
            cleanup_server(args.work)
        write_new(
            args.work / "operational-failure.json",
            {"type": type(exc).__name__, "message": str(exc), "freeze": args.freeze},
        )
        print(f"ADMISSION_FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
