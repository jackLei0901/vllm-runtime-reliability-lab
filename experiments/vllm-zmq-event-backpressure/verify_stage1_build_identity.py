#!/usr/bin/env python3
"""Verify paired Stage 1 build and dependency identities."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
TREES = {
    "base": "b7061e73a6ed4773e16bd2ae3acf47aebfd1342d",
    "fix": "46bc6e191b14ce12a04827454b4588ea5d3a435f",
}
WHEEL_FILENAME = "vllm-0.1.1.dev19+g22258a26b-cp38-abi3-manylinux_2_28_x86_64.whl"
WHEEL_SHA256 = "1d7b69a0bc85dacd9722e2af896b6a1b602e6b4712e1ea9e81b4b4ea9fcab978"
WHEEL_VERSION = "0.1.1.dev19+g22258a26b"
EXCLUDED_CROSS_ARM_DISTRIBUTIONS = {
    "vllm",
    "vllm-dfx-stage1-backpressure-hook",
}
EXPECTED_KEYS = {
    "cuda_variant",
    "dependency_pool_attachment",
    "distributions",
    "driver_version",
    "generator_sha256",
    "gpu_capability",
    "gpu_name",
    "installed_wheel_binaries",
    "local_head_commit",
    "python",
    "pool_ownership",
    "schema_version",
    "source_tree",
    "torch",
    "torch_cuda",
    "vllm_distribution_version",
    "vllm_relative_file",
    "wheel_filename",
    "wheel_sha256",
    "wheel_version",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def is_git_hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def verify(record: dict[str, Any], arm: str) -> None:
    require(set(record) == EXPECTED_KEYS, f"{arm}: unexpected identity fields")
    require(record["schema_version"] == 2, f"{arm}: schema mismatch")
    require(record["source_tree"] == TREES[arm], f"{arm}: source tree mismatch")
    require(is_git_hash(record["local_head_commit"]), f"{arm}: local HEAD invalid")
    require(record["torch"] == "2.13.0+cu130", f"{arm}: torch mismatch")
    require(record["torch_cuda"] == "13.0", f"{arm}: CUDA mismatch")
    require(record["cuda_variant"] == "cu130", f"{arm}: wheel variant mismatch")
    require(record["vllm_relative_file"] == "vllm/__init__.py", f"{arm}: import")
    require(record["wheel_filename"] == WHEEL_FILENAME, f"{arm}: wheel filename")
    require(record["wheel_sha256"] == WHEEL_SHA256, f"{arm}: wheel hash")
    require(record["wheel_version"] == WHEEL_VERSION, f"{arm}: wheel version")
    require(record["driver_version"] == "580.105.08", f"{arm}: driver mismatch")
    require(record["gpu_name"] == "NVIDIA GeForce RTX 4090", f"{arm}: GPU")
    require(record["gpu_capability"] == "8.9", f"{arm}: capability")
    require(
        record["generator_sha256"] == sha256(HERE / "stage1_build_identity.py"),
        f"{arm}: generator hash mismatch",
    )
    attachment = record["dependency_pool_attachment"]
    require(
        isinstance(attachment, dict)
        and set(attachment)
        == {"mechanism", "pth_relative_file", "pth_sha256", "value_sha256"},
        f"{arm}: dependency-pool attachment shape",
    )
    require(attachment["mechanism"] == "site-packages-pth", f"{arm}: pool")
    require(is_hash(attachment["pth_sha256"]), f"{arm}: .pth hash")
    require(is_hash(attachment["value_sha256"]), f"{arm}: pool value hash")

    distributions = record["distributions"]
    require(isinstance(distributions, dict) and distributions, f"{arm}: manifest")
    require("vllm" in distributions, f"{arm}: vLLM absent from manifest")
    for name, value in distributions.items():
        require(
            isinstance(value, dict)
            and set(value)
            == {
                "name",
                "record_sha256",
                "source",
                "verified_file_count",
                "version",
            },
            f"{arm}: invalid distribution {name}",
        )
        require(value["source"] in {"arm", "pool"}, f"{arm}: source {name}")
        require(is_hash(value["record_sha256"]), f"{arm}: RECORD {name}")
        require(
            isinstance(value["verified_file_count"], int)
            and not isinstance(value["verified_file_count"], bool)
            and value["verified_file_count"] > 0,
            f"{arm}: no installed files verified for {name}",
        )

    ownership = record["pool_ownership"]
    require(
        isinstance(ownership, dict)
        and set(ownership) == {"distribution_count", "owned_top_level_entry_count"}
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in ownership.values()
        ),
        f"{arm}: pool ownership evidence invalid",
    )

    binaries = record["installed_wheel_binaries"]
    require(isinstance(binaries, dict) and binaries, f"{arm}: wheel binaries")
    for relative, value in binaries.items():
        require(relative.startswith("vllm/"), f"{arm}: binary path")
        require(
            isinstance(value, dict) and set(value) == {"executable", "sha256"},
            f"{arm}: binary identity",
        )
        require(isinstance(value["executable"], bool), f"{arm}: executable flag")
        require(is_hash(value["sha256"]), f"{arm}: binary hash")


def comparable_manifest(record: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value
        for name, value in record["distributions"].items()
        if name not in EXCLUDED_CROSS_ARM_DISTRIBUTIONS
    }


def verify_pair(records: dict[str, dict[str, Any]]) -> None:
    common = {
        "cuda_variant",
        "dependency_pool_attachment",
        "driver_version",
        "gpu_capability",
        "gpu_name",
        "installed_wheel_binaries",
        "python",
        "pool_ownership",
        "torch",
        "torch_cuda",
        "wheel_filename",
        "wheel_sha256",
        "wheel_version",
    }
    require(
        all(records["base"][key] == records["fix"][key] for key in common),
        "build identity changed across source arms",
    )
    require(
        comparable_manifest(records["base"]) == comparable_manifest(records["fix"]),
        "dependency manifest changed across source arms",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    records = {}
    for arm in TREES:
        path = args.result_dir / f"stage1-build-{arm}.json"
        require(path.is_file(), f"missing {path.name}")
        records[arm] = json.loads(path.read_text(encoding="utf-8"))
        verify(records[arm], arm)
    verify_pair(records)
    print("PASS: paired Stage 1 build and dependency identities verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
