#!/usr/bin/env python3
"""Record a closed build identity for one vLLM #53859 Stage 1 arm."""

from __future__ import annotations

import argparse
import email
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

EXTENSIONS = (
    "vllm._C_stable_libtorch",
    "vllm._moe_C_stable_libtorch",
    "vllm.vllm_flash_attn._vllm_fa2_C",
    "vllm.vllm_flash_attn._vllm_fa3_C",
)
BASE_COMMIT_PREFIX = "22258a26b"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        args,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def wheel_version(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.endswith(".dist-info/METADATA")
        ]
        if len(names) != 1:
            raise RuntimeError("wheel must contain exactly one METADATA file")
        message = email.message_from_bytes(archive.read(names[0]))
    version = message.get("Version")
    if version is None or f"g{BASE_COMMIT_PREFIX}" not in version:
        raise RuntimeError("wheel version does not identify the baseline commit")
    return version


def extension_identity(worktree: Path, wheel: Path) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        for module in EXTENSIONS:
            spec = importlib.util.find_spec(module)
            if spec is None or spec.origin is None:
                raise RuntimeError(f"extension unavailable: {module}")
            path = Path(spec.origin).resolve()
            if not path.is_relative_to(worktree):
                raise RuntimeError(f"extension outside source tree: {module}")
            relative = path.relative_to(worktree).as_posix()
            if relative not in names:
                raise RuntimeError(f"extension absent from wheel: {relative}")
            loaded_hash = sha256(path)
            wheel_hash = hashlib.sha256(archive.read(relative)).hexdigest()
            if loaded_hash != wheel_hash:
                raise RuntimeError(f"loaded extension differs from wheel: {module}")
            records[module] = {"relative_file": relative, "sha256": loaded_hash}
    return records


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    worktree = args.worktree.resolve()
    wheel = args.wheel.resolve()
    tree = command("git", "rev-parse", "HEAD^{tree}", cwd=worktree)
    if command("git", "status", "--porcelain", cwd=worktree):
        raise RuntimeError("source worktree is not clean")

    import torch
    import vllm

    vllm_file = Path(vllm.__file__).resolve()
    if not vllm_file.is_relative_to(worktree):
        raise RuntimeError("vLLM Python import is outside source tree")
    distributions = list(metadata.distributions(name="vllm"))
    if len(distributions) != 1:
        raise RuntimeError("environment must expose exactly one vLLM distribution")
    entry_points = [
        (entry.name, entry.value)
        for entry in metadata.entry_points(group="vllm.general_plugins")
        if entry.name == "dfx_stage1_backpressure"
    ]
    expected_entry_points = [
        ("dfx_stage1_backpressure", "dfx_stage1_backpressure:register")
    ]
    if entry_points != expected_entry_points:
        raise RuntimeError("Stage 1 plugin entry point is missing or duplicated")
    if torch.__version__ != "2.13.0+cu130" or torch.version.cuda != "13.0":
        raise RuntimeError("runtime does not match the pinned torch/CUDA pair")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA GPU must be visible")

    driver, gpu_name, capability = command(
        "nvidia-smi",
        "--query-gpu=driver_version,name,compute_cap",
        "--format=csv,noheader",
    ).split(", ")
    record: dict[str, object] = {
        "cuda_variant": "cu130",
        "driver_version": driver,
        "extensions": extension_identity(worktree, wheel),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "gpu_capability": capability,
        "gpu_name": gpu_name,
        "python": ".".join(map(str, __import__("sys").version_info[:3])),
        "schema_version": 1,
        "source_tree": tree,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "vllm_distribution_version": distributions[0].version,
        "vllm_relative_file": vllm_file.relative_to(worktree).as_posix(),
        "wheel_filename": wheel.name,
        "wheel_sha256": sha256(wheel),
        "wheel_version": wheel_version(wheel),
    }
    atomic_json(args.output, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
