#!/usr/bin/env python3
"""Record a closed build and dependency identity for one Stage 1 arm."""

from __future__ import annotations

import argparse
import base64
import csv
import email
import hashlib
import importlib.metadata as metadata
import json
import os
import re
import subprocess
import sys
import sysconfig
import tempfile
import zipfile
from pathlib import Path
from typing import Any

BASE_COMMIT_PREFIX = "22258a26b"
POOL_PTH_NAME = "stage1-dependency-pool.pth"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def command(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        args, cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def wheel_version(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(names) != 1:
            raise RuntimeError("wheel must contain exactly one METADATA file")
        message = email.message_from_bytes(archive.read(names[0]))
    version = message.get("Version")
    if version is None or f"g{BASE_COMMIT_PREFIX}" not in version:
        raise RuntimeError("wheel version does not identify the baseline commit")
    return version


def wheel_binary_identity(worktree: Path, wheel: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            mode = (member.external_attr >> 16) & 0o777
            is_binary = ".so" in Path(member.filename).name or bool(mode & 0o111)
            if member.is_dir() or not is_binary:
                continue
            installed = worktree / member.filename
            if not installed.is_file():
                raise RuntimeError(f"wheel binary not installed: {member.filename}")
            installed_hash = sha256(installed)
            wheel_hash = sha256_bytes(archive.read(member.filename))
            if installed_hash != wheel_hash:
                raise RuntimeError(f"installed binary differs: {member.filename}")
            records[member.filename] = {
                "executable": bool(mode & 0o111),
                "sha256": installed_hash,
            }
    if not records:
        raise RuntimeError("wheel has no binary members")
    return dict(sorted(records.items()))


def dependency_pool_attachment(pool: Path) -> dict[str, str]:
    site_packages = Path(sysconfig.get_paths()["purelib"]).resolve()
    pth = site_packages / POOL_PTH_NAME
    if not pth.is_file():
        raise RuntimeError(f"missing dependency-pool attachment: {pth}")
    lines = [line for line in pth.read_text(encoding="utf-8").splitlines() if line]
    if lines != [str(pool)]:
        raise RuntimeError("dependency-pool .pth must contain exactly its path")
    return {
        "mechanism": "site-packages-pth",
        "pth_relative_file": pth.relative_to(Path(sys.prefix)).as_posix(),
        "pth_sha256": sha256(pth),
        "value_sha256": sha256_bytes(str(pool).encode()),
    }


def verify_record_files(metadata_path: Path) -> tuple[int, set[str]]:
    record_file = metadata_path / "RECORD"
    if not record_file.is_file():
        raise RuntimeError(f"distribution has no RECORD: {metadata_path.name}")
    installed_root = metadata_path.resolve().parent
    verified = 0
    owned_top_level: set[str] = set()
    with record_file.open(encoding="utf-8", newline="") as stream:
        for relative, encoded_hash, _size in csv.reader(stream):
            relative_path = Path(relative)
            if relative_path.parts and relative_path.parts[0] != "..":
                owned_top_level.add(relative_path.parts[0])
            if not encoded_hash:
                continue
            algorithm, separator, expected = encoded_hash.partition("=")
            if separator != "=" or algorithm != "sha256" or not expected:
                raise RuntimeError(f"unsupported RECORD hash: {relative}")
            installed = installed_root / relative_path
            if not installed.is_file():
                raise RuntimeError(f"RECORD file is missing: {relative}")
            actual = base64.urlsafe_b64encode(
                hashlib.sha256(installed.read_bytes()).digest()
            ).rstrip(b"=")
            if actual.decode("ascii") != expected:
                raise RuntimeError(f"RECORD file hash mismatch: {relative}")
            verified += 1
    if verified == 0:
        raise RuntimeError(
            f"distribution has no hashed RECORD entries: {metadata_path.name}"
        )
    return verified, owned_top_level


def distribution_identity(pool: Path) -> dict[str, dict[str, Any]]:
    site_packages = Path(sysconfig.get_paths()["purelib"]).resolve()
    result: dict[str, dict[str, Any]] = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            raise RuntimeError("visible distribution has no Name")
        key = canonical_name(name)
        raw_path = Path(distribution._path).absolute()  # type: ignore[attr-defined]
        if raw_path.is_relative_to(site_packages):
            source = "arm"
        elif raw_path.is_relative_to(pool):
            source = "pool"
        else:
            raise RuntimeError(f"distribution outside arm and pool: {name}")
        record_file = raw_path / "RECORD"
        verified_count, _owned = verify_record_files(raw_path)
        if key in result:
            raise RuntimeError(f"duplicate visible distribution: {key}")
        result[key] = {
            "name": name,
            "record_sha256": sha256(record_file),
            "source": source,
            "version": distribution.version,
            "verified_file_count": verified_count,
        }
    return dict(sorted(result.items()))


def verify_pool_ownership(pool: Path) -> dict[str, int]:
    owned: set[str] = set()
    distribution_count = 0
    for distribution in metadata.distributions(path=[str(pool)]):
        distribution_count += 1
        raw_path = Path(distribution._path).absolute()  # type: ignore[attr-defined]
        _verified_count, distribution_owned = verify_record_files(raw_path)
        owned.update(distribution_owned)
    ignored = {"__pycache__"}
    entries = {
        path.name
        for path in pool.iterdir()
        if path.name not in ignored and not path.name.startswith(".")
    }
    unowned = sorted(entries - owned)
    if unowned:
        raise RuntimeError(f"dependency pool has unowned top-level entries: {unowned}")
    return {
        "distribution_count": distribution_count,
        "owned_top_level_entry_count": len(entries),
    }


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
    parser.add_argument("--dependency-pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    worktree = args.worktree.resolve()
    wheel = args.wheel.resolve()
    pool = args.dependency_pool.resolve()
    if not pool.is_dir():
        raise RuntimeError("dependency pool is missing")
    tree = command("git", "rev-parse", "HEAD^{tree}", cwd=worktree)
    head = command("git", "rev-parse", "HEAD", cwd=worktree)
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
    if entry_points != [
        ("dfx_stage1_backpressure", "dfx_stage1_backpressure:register")
    ]:
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
        "dependency_pool_attachment": dependency_pool_attachment(pool),
        "distributions": distribution_identity(pool),
        "driver_version": driver,
        "generator_sha256": sha256(Path(__file__).resolve()),
        "gpu_capability": capability,
        "gpu_name": gpu_name,
        "installed_wheel_binaries": wheel_binary_identity(worktree, wheel),
        "local_head_commit": head,
        "python": ".".join(map(str, sys.version_info[:3])),
        "pool_ownership": verify_pool_ownership(pool),
        "schema_version": 2,
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
