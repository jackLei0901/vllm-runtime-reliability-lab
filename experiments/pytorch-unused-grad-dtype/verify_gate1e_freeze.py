"""Verify that the pre-execution Gate 1e inputs match their frozen hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "GATE1E_FREEZE.json"


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("freeze_id") != "gate1e-pre-execution-2026-09-13":
        raise AssertionError("unexpected Gate 1e freeze identifier")
    if manifest.get("status") != "pre-execution":
        raise AssertionError("Gate 1e freeze is not marked pre-execution")
    if manifest.get("supersedes") != "gate1d-pre-execution-2026-09-12":
        raise AssertionError("Gate 1e does not supersede the executed Gate 1d")
    if manifest.get("superseded_gate_status") != "stopped-during-control":
        raise AssertionError("Gate 1d execution status is inconsistent")
    if manifest.get("timing_seconds") != {
        "stack_capture": 20,
        "process_group_timeout": 30,
        "wall_bound": 60,
    }:
        raise AssertionError("Gate 1e frozen timing is inconsistent")
    if manifest.get("control_mismatch_policy") != "stop-and-retain":
        raise AssertionError("Gate 1e control mismatch policy is inconsistent")
    if manifest.get("affected_termination_mismatch_policy") != "record-and-continue":
        raise AssertionError("Gate 1e affected termination policy is inconsistent")
    files = manifest.get("files")
    expected_names = {
        "GATE1E_PROTOCOL.md",
        "gate1c_campaign.py",
        "gate1d_campaign.py",
        "gate1d_reproducer.py",
        "gate1e_campaign.py",
        "normalize_flight_recorder.py",
        "process_lifecycle.py",
        "verify_gate1c.py",
        "verify_gate1d.py",
        "verify_gate1e_freeze.py",
    }
    if not isinstance(files, dict) or set(files) != expected_names:
        raise AssertionError("Gate 1e freeze file set is wrong")
    for name, expected_hash in sorted(files.items()):
        path = HERE / name
        if not path.is_file():
            path = HERE.parent / "organic-hang" / name
        if not path.is_file():
            raise AssertionError(f"frozen file is missing: {name}")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise AssertionError(f"frozen file changed: {name}")
    print(f"PASS ({len(files)} frozen Gate 1e files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
