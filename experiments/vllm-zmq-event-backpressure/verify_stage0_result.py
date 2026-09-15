#!/usr/bin/env python3
"""Fail-closed verifier for the public vLLM #53859 Stage 0 summary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SUMMARY = ROOT / "results" / "vllm-zmq-backpressure-stage0-20260915" / "summary.json"

EXPECTED_BASE_TREE = "b7061e73a6ed4773e16bd2ae3acf47aebfd1342d"
EXPECTED_FIX_TREE = "46bc6e191b14ce12a04827454b4588ea5d3a435f"
EXPECTED_FIX_PATCH_SHA256 = (
    "ebf0e35f53e6e3a74c79d608f6a2656d3f7537bcb3c648ce6885a8eac3423dfc"
)
EXPECTED_FILES = {
    "tests/distributed/test_events.py",
    "vllm/distributed/kv_events.py",
}
EXPECTED_STACK = [
    "publish_second (stage0_backpressure.py)",
    "publish (vllm/distributed/kv_events.py)",
    "put (queue.py)",
    "wait (threading.py)",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    require(data["verdict"] == "PASS", "summary verdict is not PASS")

    source = data["source"]
    require(source["base_tree"] == EXPECTED_BASE_TREE, "base tree mismatch")
    require(source["base_tree_matches_github"] is True, "base tree not verified")
    require(source["fix_tree"] == EXPECTED_FIX_TREE, "fix tree mismatch")
    require(
        source["fix_patch_sha256"] == EXPECTED_FIX_PATCH_SHA256,
        "fix patch hash mismatch",
    )
    require(set(source["fix_changed_files"]) == EXPECTED_FILES, "fix file set mismatch")

    protocol = data["protocol"]
    require(
        sha256(HERE / protocol["executed_script"]) == protocol["script_sha256"],
        "executed runner hash mismatch",
    )
    require(
        sha256(HERE / protocol["hardened_script"])
        == protocol["hardened_script_sha256"],
        "hardened runner hash mismatch",
    )
    require(
        sha256(HERE / "STAGE0_PROTOCOL.md") == protocol["sha256"],
        "protocol hash mismatch",
    )
    require(protocol["queue_size"] == 1, "queue size changed")
    require(protocol["observation_seconds"] == 1.0, "observation bound changed")
    require(protocol["base_hold_seconds"] == 20.0, "base hold changed")

    base = data["arms"]["base"]
    require(base["return_code"] == 0, "base runner failed")
    require(base["consumer_paused"] is True, "base consumer was not paused")
    require(
        base["second_publish_returned_inside_bound"] is False,
        "base publish did not block",
    )
    require(base["first_batch_retained_before_release"] is True, "base queue changed")
    require(base["recovered_after_release"] is True, "base did not recover")
    require(
        base["second_publish_completed_after_release"] is True,
        "base second publish did not complete after release",
    )
    require(base["queue_drained_after_release"] is True, "base queue did not drain")
    require(base["stack_observed"] is True, "base stack was not observed")
    require(base["stack_call_chain_outer_to_inner"] == EXPECTED_STACK, "stack mismatch")

    fix = data["arms"]["fix"]
    require(fix["return_code"] == 0, "fix runner failed")
    require(fix["consumer_paused"] is True, "fix consumer was not paused")
    require(
        fix["second_publish_returned_inside_bound"] is True,
        "fix publish still blocked",
    )
    require(fix["first_batch_retained_before_release"] is True, "fix replaced batch")
    require(fix["new_batch_dropped"] is True, "fix did not record event loss")
    require(fix["queue_full_warning_observed"] is True, "fix warning absent")
    require(fix["queue_drained_after_release"] is True, "fix queue did not drain")

    unsupported = " ".join(data["interpretation"]["not_supported"])
    require("EngineCore" in unsupported, "EngineCore limitation missing")
    require("sequence gap" in unsupported, "undetectable-loss limitation missing")
    print("PASS: vLLM #53859 Stage 0 base/fix evidence matches the frozen contract")


if __name__ == "__main__":
    main()
