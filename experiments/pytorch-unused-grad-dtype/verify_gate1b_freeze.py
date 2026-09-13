"""Verify Gate 1b pre-execution files against their frozen hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST = HERE / "GATE1B_FREEZE.json"


def verify() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("freeze_id") != "gate1b-pre-execution-2026-09-12":
        raise AssertionError("unexpected Gate 1b freeze identity")
    if manifest.get("status") != "pre-execution":
        raise AssertionError("Gate 1b freeze is no longer pre-execution")
    if manifest.get("expected_torch_version") != "2.13.0+cu130":
        raise AssertionError("unexpected Gate 1b torch contract")
    if manifest.get("expected_classifications") != {
        "control": "completed_symmetric_fp32",
        "affected": "rank1_assertion_rank0_wait",
    }:
        raise AssertionError("unexpected Gate 1b classification contract")
    if manifest.get("timing_seconds") != {
        "stack_capture": 20,
        "process_group_timeout": 30,
        "wall_timeout": 45,
    }:
        raise AssertionError("unexpected Gate 1b timing contract")
    files = manifest.get("files")
    if not isinstance(files, dict) or len(files) != 7:
        raise AssertionError("Gate 1b freeze must contain exactly seven files")
    for relative_path, expected_hash in sorted(files.items()):
        path = ROOT / relative_path
        if not path.is_file():
            raise AssertionError(f"frozen file missing: {relative_path}")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise AssertionError(f"frozen file changed: {relative_path}: {actual_hash}")
    print(f"PASS ({len(files)} frozen Gate 1b files)")


if __name__ == "__main__":
    verify()
