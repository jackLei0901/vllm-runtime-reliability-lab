"""Bind an existing serving install; never install, serve or initialize CUDA."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import build_perf
import runtime
import serve_trace
import wheel_preflight


def check_dependencies():
    import torch

    if torch.__version__ != "2.13.0+cu130" or torch.version.cuda != "13.0":
        raise ValueError("exact Torch 2.13.0+cu130 / CUDA 13.0 required")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode:
        raise ValueError("pip check failed: " + result.stdout + result.stderr)
    packages = sorted(
        (d.metadata["Name"], d.version) for d in importlib.metadata.distributions()
    )
    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "packages": packages,
        "pip_check": result.stdout.strip(),
        "cuda_initialized": "not requested; no device operations in this tool",
    }


def prepare_provenance(src, index, wheel, log, command_file, installed):
    info = wheel_preflight.inspect_archive(index, wheel)
    command = command_file.read_text().strip()
    if not command or not log.is_file() or not log.stat().st_size:
        raise ValueError("nonempty retained install command and log required")
    provenance = {
        "kind": "precompiled_parent",
        "source_pin": build_perf.PIN,
        "wheel_commit": serve_trace.PARENT,
        "wheel_url": info["wheel_url"],
        "wheel_path": str(wheel.resolve()),
        "wheel_sha256": info["wheel_sha256"],
        "wheel_index_url": serve_trace.WHEEL_INDEX,
        "wheel_index_path": str(index.resolve()),
        "wheel_index_sha256": runtime.sha(index),
        "extensions_sha256": installed["extensions_sha256"],
        "build_command": command,
        "build_log": str(log.resolve()),
        "build_log_sha256": runtime.sha(log),
    }
    serve_trace.verify_provenance(provenance, installed, src)
    return provenance


def main():
    parser = argparse.ArgumentParser()
    for name in (
        "vllm-src",
        "index",
        "wheel",
        "install-log",
        "install-command-file",
        "out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--freeze-commit", required=True)
    args = parser.parse_args()
    out = runtime.fresh(args.out)
    result = {"status": "unscored", "runtime_loading": "not tested"}
    try:
        result["public_freeze"] = runtime.public_freeze(args.freeze_commit)
        result["environment"] = check_dependencies()
        result["installed"] = serve_trace.installed_identity(args.vllm_src.resolve())
        provenance = prepare_provenance(
            args.vllm_src.resolve(),
            args.index,
            args.wheel,
            args.install_log,
            args.install_command_file,
            result["installed"],
        )
        runtime.write(out / "serving_build.json", provenance)
        result["status"] = "installation_identity_verified_runtime_unverified"
    except Exception as exc:  # noqa: BLE001 - retain failed apparatus check
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        runtime.write(out / "installation.json", result)
    print(json.dumps({"status": result["status"]}))
    if result["status"] == "unscored":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
