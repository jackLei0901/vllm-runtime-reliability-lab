#!/usr/bin/env python3
"""Verify the two real-publisher Stage 1a CPU preflight summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from stage1_campaign import BASE_TREE, FIX_TREE, PLUGIN_ROOT, sha256_file

HERE = Path(__file__).resolve().parent
EXPECTED_FILES = {"base": "stage1a-base.json", "fix": "stage1a-fix.json"}
SUMMARY_KEYS = {
    "cleanup",
    "counts_after_release",
    "counts_before_release",
    "engine_core_kv_events_sha256",
    "engine_core_mapped_worktree_binaries",
    "environment",
    "implementation_sha256",
    "ready_authorization",
    "schema_version",
    "source_arm",
    "source_tree",
    "stack",
    "stack_attempt_count",
    "subject_return_code",
    "verdict",
    "yama_ptrace_scope",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_counts(value: Any, label: str) -> None:
    require(
        isinstance(value, dict)
        and set(value) == {"accepted_batch_count", "dropped_batch_count"},
        f"{label}: queue-count shape mismatch",
    )
    require(
        all(
            isinstance(count, int) and not isinstance(count, bool) and count >= 0
            for count in value.values()
        ),
        f"{label}: invalid queue count",
    )


def verify_record(record: dict[str, Any], source_arm: str) -> None:
    require(set(record) == SUMMARY_KEYS, f"{source_arm}: summary shape mismatch")
    require(record["schema_version"] == 1, f"{source_arm}: schema mismatch")
    require(record["source_arm"] == source_arm, f"{source_arm}: arm mismatch")
    expected_tree = BASE_TREE if source_arm == "base" else FIX_TREE
    require(record["source_tree"] == expected_tree, f"{source_arm}: tree mismatch")
    require(record["verdict"] == "PASS", f"{source_arm}: preflight failed")
    require(record["subject_return_code"] == 0, f"{source_arm}: subject failed")

    plugin = PLUGIN_ROOT / "src" / "dfx_stage1_backpressure" / "__init__.py"
    expected_implementation = {
        "preflight": sha256_file(HERE / "stage1a_cpu_preflight.py"),
        "campaign": sha256_file(HERE / "stage1_campaign.py"),
        "contract": sha256_file(HERE / "stage1_contract.py"),
        "plugin": sha256_file(plugin),
    }
    require(
        record["implementation_sha256"] == expected_implementation,
        f"{source_arm}: implementation hash mismatch",
    )
    require(
        record["ready_authorization"] == "pr_set_ptracer_observer",
        f"{source_arm}: observer authorization mismatch",
    )
    scope = record["yama_ptrace_scope"]
    require(
        isinstance(scope, int) and not isinstance(scope, bool) and scope >= 1,
        f"{source_arm}: Yama was not restrictive",
    )
    cleanup = record["cleanup"]
    require(
        isinstance(cleanup, dict)
        and set(cleanup) == {"engine_core_gone", "process_group_gone", "termination"},
        f"{source_arm}: cleanup shape mismatch",
    )
    require(cleanup["engine_core_gone"] is True, f"{source_arm}: subject survived")
    require(
        cleanup["process_group_gone"] is True,
        f"{source_arm}: process group survived",
    )

    before = record["counts_before_release"]
    after = record["counts_after_release"]
    verify_counts(before, f"{source_arm} before release")
    verify_counts(after, f"{source_arm} after release")
    if source_arm == "base":
        require(
            before == {"accepted_batch_count": 1, "dropped_batch_count": 0},
            "base: unexpected counts before release",
        )
        require(after["accepted_batch_count"] >= 2, "base: blocked batch not accepted")
        require(after["dropped_batch_count"] == 0, "base: unexpected drop")
        stack = record["stack"]
        require(stack["available"] is True, "base: py-spy unavailable")
        require(stack["match"] is True, "base: stack chain mismatch")
        require(record["stack_attempt_count"] >= 1, "base: no stack attempt")
    else:
        require(before["accepted_batch_count"] >= 1, "fix: no accepted batch")
        require(before["dropped_batch_count"] >= 1, "fix: no dropped batch")
        require(after == before, "fix: queue counts changed after release")
        require(
            record["stack"] == {"available": None, "match": None, "raw_sha256": None},
            "fix: unexpected stack claim",
        )
        require(record["stack_attempt_count"] == 0, "fix: unexpected stack attempt")

    environment = record["environment"]
    require(
        environment["torch"] == "2.13.0+cu130",
        f"{source_arm}: Stage 1 runtime torch mismatch",
    )
    require(
        environment["import_matches_tree"] is True, f"{source_arm}: import mismatch"
    )
    require(
        environment["kv_events_sha256"] == environment["tree_kv_events_sha256"],
        f"{source_arm}: imported kv_events mismatch",
    )
    require(
        record["engine_core_kv_events_sha256"] == environment["tree_kv_events_sha256"],
        f"{source_arm}: EngineCore kv_events mismatch",
    )
    require(
        isinstance(record["engine_core_mapped_worktree_binaries"], dict),
        f"{source_arm}: mapped binary evidence is invalid",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    records = {}
    for source_arm, filename in EXPECTED_FILES.items():
        path = args.result_dir / filename
        require(path.is_file(), f"missing {filename}")
        record = json.loads(path.read_text(encoding="utf-8"))
        verify_record(record, source_arm)
        records[source_arm] = record

    shared_keys = {"python", "torch", "cuda", "cuda_available", "gpu_count"}
    base_environment = records["base"]["environment"]
    fix_environment = records["fix"]["environment"]
    require(
        all(base_environment[key] == fix_environment[key] for key in shared_keys),
        "runtime changed across source arms",
    )
    print("PASS: Stage 1a real plugin/publisher preflight verified on both trees")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
