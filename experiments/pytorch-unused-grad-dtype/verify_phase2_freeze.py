"""Verify the pre-execution Phase 2 files against their frozen hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST = HERE / "PHASE2_FREEZE.json"


def verify() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("freeze_id") != "phase2-pre-execution-2026-09-12":
        raise AssertionError("unexpected Phase 2 freeze identity")
    if manifest.get("status") != "pre-execution":
        raise AssertionError("Phase 2 freeze is no longer pre-execution")
    files = manifest.get("files")
    if not isinstance(files, dict) or len(files) != 15:
        raise AssertionError("Phase 2 freeze must contain exactly 15 files")
    for relative_path, expected_hash in sorted(files.items()):
        path = ROOT / relative_path
        if not path.is_file():
            raise AssertionError(f"frozen file missing: {relative_path}")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise AssertionError(f"frozen file changed: {relative_path}: {actual_hash}")
    print(f"PASS ({len(files)} frozen Phase 2 files)")


if __name__ == "__main__":
    verify()
