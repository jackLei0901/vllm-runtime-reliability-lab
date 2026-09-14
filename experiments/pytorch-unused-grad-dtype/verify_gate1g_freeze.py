"""Verify that Gate 1g runtime inputs match their pre-execution freeze."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "GATE1G_FREEZE.json"


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("freeze_id") != "gate1g-pre-execution-2026-09-13":
        raise AssertionError("unexpected Gate 1g freeze identifier")
    if manifest.get("status") != "pre-execution":
        raise AssertionError("Gate 1g freeze is not marked pre-execution")
    if manifest.get("follows") != "gate1f-confirmed-2026-09-13":
        raise AssertionError("Gate 1g does not preserve the Gate 1f result")
    if manifest.get("scope") != "fsdp-free-process-group-nccl":
        raise AssertionError("Gate 1g scope changed")
    if manifest.get("expected_torch") != "2.13.0+cu130":
        raise AssertionError("Gate 1g torch version changed")
    if manifest.get("expected_nccl_version") != [2, 29, 7]:
        raise AssertionError("Gate 1g NCCL version changed")
    if manifest.get("trials") != {"affected": 1, "control": 1}:
        raise AssertionError("Gate 1g trial count changed")
    if manifest.get("timing_seconds") != {
        "stack_capture": 20,
        "process_group_timeout": 30,
        "wall_bound": 60,
        "work_wait_timeout": 180,
    }:
        raise AssertionError("Gate 1g timing changed")
    if manifest.get("communicator_precondition") != (
        "two-rank-all-reduce-plus-cuda-synchronize"
    ):
        raise AssertionError("Gate 1g communicator precondition changed")
    if manifest.get("pending_all_reduce_candidates") != "exactly-one-rank0-local":
        raise AssertionError("Gate 1g pending-entry cardinality changed")
    if manifest.get("raw_retention") != "derived-summaries-only":
        raise AssertionError("Gate 1g privacy boundary changed")
    if manifest.get("peer_participation_inference") != "forbidden":
        raise AssertionError("Gate 1g local-evidence boundary changed")
    files = manifest.get("files")
    expected_names = {
        "GATE1G_PROTOCOL.md",
        "gate1c_campaign.py",
        "gate1d_campaign.py",
        "gate1e_campaign.py",
        "gate1f_campaign.py",
        "gate1g_campaign.py",
        "gate1g_reproducer.py",
        "normalize_flight_recorder.py",
        "process_lifecycle.py",
        "verify_gate1g.py",
        "verify_gate1g_freeze.py",
    }
    if not isinstance(files, dict) or set(files) != expected_names:
        raise AssertionError("Gate 1g freeze file set is wrong")
    for name, expected_hash in sorted(files.items()):
        path = HERE / name
        if not path.is_file():
            path = HERE.parent / "organic-hang" / name
        if not path.is_file():
            raise AssertionError(f"frozen file is missing: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise AssertionError(f"frozen file changed: {name}")
    print(f"PASS ({len(files)} frozen Gate 1g files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
