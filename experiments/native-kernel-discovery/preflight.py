"""CPU-only public freeze verification, reusable offline during GPU admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

FILES = (
    "docs/kernel/NATIVE_KERNEL_PROFILE_DECISION_2026-10-03.md",
    "docs/kernel/NATIVE_KERNEL_PROFILE_DECISION_2026-10-03.zh-CN.md",
    "experiments/native-kernel-discovery/collect.py",
    "experiments/native-kernel-discovery/preflight.py",
    "experiments/native-kernel-discovery/admit.py",
    "tests/test_native_kernel_discovery.py",
)


def sha(path):
    with Path(path).open("rb") as stream:
        digest = (
            hashlib.file_digest(stream, "sha256")
            if hasattr(hashlib, "file_digest")
            else None
        )
        if digest is None:  # Python 3.10 CI
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


def validate_identity(receipt, expected_count):
    """Read the actual comparison receipt schema, rejecting missing fields."""
    if (
        type(expected_count) is not int
        or expected_count <= 0
        or type(receipt.get("wheel_members_checked")) is not int
        or receipt["wheel_members_checked"] != expected_count
        or type(receipt.get("mismatch_count")) is not int
        or receipt["mismatch_count"] != 0
        or receipt.get("mismatches") != []
    ):
        raise ValueError("invalid or failed installed-wheel comparison receipt")


def write_new(path, data):
    # A failure or successful verification is never silently overwritten.
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def validate_commit(commit):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("full lowercase commit SHA required")


def prepare_public(root, commit, receipt, fetch=None):
    """Only this function uses the network; call BEFORE booking."""
    root, receipt = Path(root), Path(receipt)
    validate_commit(commit)
    if receipt.exists():
        return check_public(root, commit, receipt)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    verified = {}
    for relative in FILES:
        expected = sha(root / relative)
        # Each completed file survives a later network failure.
        cache = receipt.parent / (
            receipt.name
            + "."
            + hashlib.sha256(relative.encode()).hexdigest()[:16]
            + ".verified.json"
        )
        if cache.exists():
            row = json.loads(cache.read_text(encoding="utf-8"))
            if row != {"commit": commit, "path": relative, "sha256": expected}:
                raise ValueError("partial verification differs; use a new receipt")
        else:
            url = (
                "https://raw.githubusercontent.com/jackLei0901/"
                f"vllm-runtime-reliability-lab/{commit}/{relative}"
            )
            try:
                if fetch is None:
                    with urllib.request.urlopen(url, timeout=20) as response:
                        data = response.read()
                else:
                    data = fetch(url)
                if hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError(f"public file mismatch: {relative}")
            except Exception as exc:
                write_new(
                    receipt.parent / (receipt.name + f".failure-{time.time_ns()}.json"),
                    {
                        "commit": commit,
                        "path": relative,
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
                raise
            write_new(cache, {"commit": commit, "path": relative, "sha256": expected})
        verified[relative] = expected
    write_new(
        receipt,
        {
            "schema": 1,
            "commit": commit,
            "verified_utc_ns": time.time_ns(),
            "files": verified,
        },
    )
    return check_public(root, commit, receipt)


def check_public(root, commit, receipt):
    """Offline only: no fallback to network or automatic receipt creation."""
    validate_commit(commit)
    data = json.loads(Path(receipt).read_text(encoding="utf-8"))
    expected = {relative: sha(Path(root) / relative) for relative in FILES}
    if (
        data.get("schema") != 1
        or data.get("commit") != commit
        or (data.get("files") != expected)
    ):
        raise ValueError("public receipt or local frozen files differ")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "check"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    function = prepare_public if args.mode == "prepare" else check_public
    function(args.root, args.freeze, args.receipt)
    print("PUBLIC_FREEZE_MATCHES")


if __name__ == "__main__":
    main()
