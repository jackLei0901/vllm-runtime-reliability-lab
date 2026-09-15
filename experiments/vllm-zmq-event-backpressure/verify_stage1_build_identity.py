#!/usr/bin/env python3
"""Verify the paired Stage 1 build identities without the private wheel."""

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
WHEEL_FILENAME = (
    "vllm-0.1.1.dev19+g22258a26b-cp38-abi3-manylinux_2_28_x86_64.whl"
)
WHEEL_SHA256 = "1d7b69a0bc85dacd9722e2af896b6a1b602e6b4712e1ea9e81b4b4ea9fcab978"
WHEEL_VERSION = "0.1.1.dev19+g22258a26b"
EXPECTED_EXTENSIONS = {
    "vllm._C_stable_libtorch": (
        "vllm/_C_stable_libtorch.abi3.so",
        "0086011da596463c9e1d2ee8c591e9d7a9522737f29455d62fc34d8ec9b3f980",
    ),
    "vllm._moe_C_stable_libtorch": (
        "vllm/_moe_C_stable_libtorch.abi3.so",
        "bbf2462d63bac8de096d1cd1f48a979f5b39cb394238e4c8a7aebc179b13b7f7",
    ),
    "vllm.vllm_flash_attn._vllm_fa2_C": (
        "vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so",
        "f49b6ac53ef96d5bc457ef231b7cef46c8aff833431d820c2a27463a14c82dcf",
    ),
    "vllm.vllm_flash_attn._vllm_fa3_C": (
        "vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so",
        "8759be32480e0a4b0d964bd880da8b108d5647f6a46ff00472597f700d70c70e",
    ),
}
EXPECTED_KEYS = {
    "cuda_variant",
    "driver_version",
    "extensions",
    "generator_sha256",
    "gpu_capability",
    "gpu_name",
    "python",
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


def verify(record: dict[str, Any], arm: str) -> None:
    require(set(record) == EXPECTED_KEYS, f"{arm}: unexpected identity fields")
    require(record["schema_version"] == 1, f"{arm}: schema mismatch")
    require(record["source_tree"] == TREES[arm], f"{arm}: source tree mismatch")
    require(record["torch"] == "2.13.0+cu130", f"{arm}: torch mismatch")
    require(record["torch_cuda"] == "13.0", f"{arm}: CUDA mismatch")
    require(record["cuda_variant"] == "cu130", f"{arm}: wheel variant mismatch")
    require(
        record["vllm_relative_file"] == "vllm/__init__.py",
        f"{arm}: import mismatch",
    )
    require(record["wheel_filename"] == WHEEL_FILENAME, f"{arm}: wheel filename")
    require(record["wheel_sha256"] == WHEEL_SHA256, f"{arm}: wheel hash")
    require(record["wheel_version"] == WHEEL_VERSION, f"{arm}: wheel version")
    require(record["driver_version"] == "580.105.08", f"{arm}: driver mismatch")
    require(record["gpu_name"] == "NVIDIA GeForce RTX 4090", f"{arm}: GPU mismatch")
    require(record["gpu_capability"] == "8.9", f"{arm}: capability mismatch")
    require(
        record["generator_sha256"] == sha256(HERE / "stage1_build_identity.py"),
        f"{arm}: generator hash mismatch",
    )
    extensions = record["extensions"]
    require(set(extensions) == set(EXPECTED_EXTENSIONS), f"{arm}: extensions")
    for module, value in extensions.items():
        require(
            isinstance(value, dict) and set(value) == {"relative_file", "sha256"},
            f"{arm}: invalid extension identity",
        )
        expected_file, expected_hash = EXPECTED_EXTENSIONS[module]
        require(value["relative_file"] == expected_file, f"{arm}: extension path")
        require(value["sha256"] == expected_hash, f"{arm}: extension hash")


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
    common = {
        "cuda_variant",
        "driver_version",
        "extensions",
        "gpu_capability",
        "gpu_name",
        "python",
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
    print("PASS: paired Stage 1 exact-commit build identities verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
