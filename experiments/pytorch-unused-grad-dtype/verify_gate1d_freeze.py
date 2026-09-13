"""Verify that the pre-execution Gate 1d inputs match their frozen hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "GATE1D_FREEZE.json"


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("freeze_id") != "gate1d-pre-execution-2026-09-12":
        raise AssertionError("unexpected Gate 1d freeze identifier")
    if manifest.get("status") != "pre-execution":
        raise AssertionError("Gate 1d freeze is not marked pre-execution")
    if manifest.get("supersedes") != "gate1c-pre-execution-2026-09-12":
        raise AssertionError("Gate 1d does not supersede the withdrawn Gate 1c")
    if manifest.get("superseded_gate_status") != "withdrawn-before-execution":
        raise AssertionError("Gate 1c withdrawal status is inconsistent")
    if manifest.get("timing_seconds") != {
        "stack_capture": 20,
        "process_group_timeout": 30,
        "wall_bound": 60,
    }:
        raise AssertionError("Gate 1d frozen timing is inconsistent")
    if manifest.get("termination_mismatch_policy") != "record-and-continue":
        raise AssertionError("Gate 1d termination mismatch policy is inconsistent")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise AssertionError("Gate 1d freeze file map is empty")
    expected_names = {
        "GATE1D_PROTOCOL.md",
        "gate1c_campaign.py",
        "gate1d_campaign.py",
        "gate1d_reproducer.py",
        "normalize_flight_recorder.py",
        "process_lifecycle.py",
        "verify_gate1c.py",
        "verify_gate1d.py",
        "verify_gate1d_freeze.py",
    }
    if set(files) != expected_names:
        raise AssertionError("Gate 1d freeze file set is wrong")
    for name, expected_hash in sorted(files.items()):
        path = HERE / name
        if not path.is_file():
            path = HERE.parent / "organic-hang" / name
        if not path.is_file():
            raise AssertionError(f"frozen file is missing: {name}")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise AssertionError(f"frozen file changed: {name}")
    print(f"PASS ({len(files)} frozen Gate 1d files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
