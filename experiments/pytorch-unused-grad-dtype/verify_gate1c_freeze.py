"""Verify that the pre-execution Gate 1c inputs match their frozen hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "GATE1C_FREEZE.json"


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("freeze_id") != "gate1c-pre-execution-2026-09-12":
        raise AssertionError("unexpected Gate 1c freeze identifier")
    if manifest.get("status") != "pre-execution":
        raise AssertionError("Gate 1c freeze is not marked pre-execution")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise AssertionError("Gate 1c freeze file map is empty")
    expected_names = {
        "GATE1C_PROTOCOL.md",
        "gate1c_campaign.py",
        "gate1c_reproducer.py",
        "normalize_flight_recorder.py",
        "process_lifecycle.py",
        "verify_gate1c.py",
        "verify_gate1c_freeze.py",
    }
    if set(files) != expected_names:
        raise AssertionError("Gate 1c freeze file set is wrong")
    for name, expected_hash in sorted(files.items()):
        path = HERE / name
        if not path.is_file():
            # Shared helpers live in the sibling organic-hang experiment.
            path = HERE.parent / "organic-hang" / name
        if not path.is_file():
            raise AssertionError(f"frozen file is missing: {name}")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise AssertionError(f"frozen file changed: {name}")
    print(f"PASS ({len(files)} frozen Gate 1c files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
