"""Verify that the pre-execution Gate 1f inputs match their frozen hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "GATE1F_FREEZE.json"


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("freeze_id") != "gate1f-pre-execution-2026-09-13":
        raise AssertionError("unexpected Gate 1f freeze identifier")
    if manifest.get("status") != "pre-execution":
        raise AssertionError("Gate 1f freeze is not marked pre-execution")
    if manifest.get("follows") != "gate1e-failed-closed-2026-09-13":
        raise AssertionError("Gate 1f does not preserve the Gate 1e result")
    if manifest.get("diagnostic_change") != "TORCH_CPP_LOG_LEVEL=INFO":
        raise AssertionError("Gate 1f diagnostic change is inconsistent")
    if manifest.get("expected_nccl_version") != [2, 29, 7]:
        raise AssertionError("Gate 1f NCCL version is inconsistent")
    if manifest.get("trials") != {"affected": 1, "control": 0}:
        raise AssertionError("Gate 1f trial count is inconsistent")
    if manifest.get("timing_seconds") != {
        "stack_capture": 20,
        "process_group_timeout": 30,
        "wall_bound": 60,
    }:
        raise AssertionError("Gate 1f timing is inconsistent")
    if manifest.get("expected_rank1_library_log_flags") != {
        "shutdown_start": True,
        "operations_flushed": True,
        "watchdog_joined_destroying": True,
        "destroy_complete": False,
        "dump_signal_observed": False,
        "dump_signal_broadcast": False,
        "dump_signal_broadcast_failed": False,
        "dump_success": False,
    }:
        raise AssertionError("Gate 1f rank-1 prediction is inconsistent")
    if manifest.get("expected_rank0_library_log_flags") != {
        "shutdown_start": False,
        "operations_flushed": False,
        "watchdog_joined_destroying": False,
        "destroy_complete": False,
        "dump_signal_observed": False,
        "dump_signal_broadcast": True,
        "dump_signal_broadcast_failed": False,
        "dump_success": True,
    }:
        raise AssertionError("Gate 1f rank-0 prediction is inconsistent")
    if manifest.get("library_log_scan_error_policy") != "retain-and-fail-closed":
        raise AssertionError("Gate 1f scanner-error policy is inconsistent")
    files = manifest.get("files")
    expected_names = {
        "GATE1F_PROTOCOL.md",
        "gate1c_campaign.py",
        "gate1d_campaign.py",
        "gate1d_reproducer.py",
        "gate1e_campaign.py",
        "gate1f_campaign.py",
        "normalize_flight_recorder.py",
        "process_lifecycle.py",
        "verify_gate1c.py",
        "verify_gate1d.py",
        "verify_gate1f.py",
        "verify_gate1f_freeze.py",
    }
    if not isinstance(files, dict) or set(files) != expected_names:
        raise AssertionError("Gate 1f freeze file set is wrong")
    for name, expected_hash in sorted(files.items()):
        path = HERE / name
        if not path.is_file():
            path = HERE.parent / "organic-hang" / name
        if not path.is_file():
            raise AssertionError(f"frozen file is missing: {name}")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise AssertionError(f"frozen file changed: {name}")
    print(f"PASS ({len(files)} frozen Gate 1f files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
