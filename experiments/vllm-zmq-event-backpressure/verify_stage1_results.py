#!/usr/bin/env python3
"""Fail-closed verifier for the four-cell vLLM #53859 Stage 1 result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BASE_COMMIT = "22258a26bc090bccf5473cf681bbe9bac41bd035"
FIX_HEAD = "1a2b85306b6d13033bbecc693e6cb776acb4bcaa"
FIX_PATCH_SHA256 = "ebf0e35f53e6e3a74c79d608f6a2656d3f7537bcb3c648ce6885a8eac3423dfc"
BASE_TREE = "b7061e73a6ed4773e16bd2ae3acf47aebfd1342d"
FIX_TREE = "46bc6e191b14ce12a04827454b4588ea5d3a435f"
KV_EVENTS_SHA256 = {
    "base": "de08f01e8736881560256c350a3b6418999dd4aa30f9c6ec9159527f7ed6899a",
    "fix": "15c3038f1bf97e785c3f1960451e10b54bfa3d87ab456ce2550a2fe98ba1b49a",
}
CELLS = {
    1: ("base", "control"),
    2: ("fix", "control"),
    3: ("base", "pause"),
    4: ("fix", "pause"),
}
FILES = {
    index: f"cell-{index}-{source}-{trigger}.json"
    for index, (source, trigger) in CELLS.items()
}
HASHED_IMPLEMENTATION = {
    "campaign": HERE / "stage1_campaign.py",
    "contract": HERE / "stage1_contract.py",
    "plugin": (
        HERE / "stage1_plugin" / "src" / "dfx_stage1_backpressure" / "__init__.py"
    ),
    "plugin_package": HERE / "stage1_plugin" / "pyproject.toml",
    "protocol": HERE / "STAGE1_PROTOCOL_DRAFT.md",
}
SUMMARY_KEYS = {
    "accepted_batch_count",
    "build_identity_sha256",
    "campaign_error_kind",
    "cell_index",
    "classification",
    "completion_tokens",
    "dropped_batch_count",
    "engine_core_bound",
    "engine_core_kv_events_sha256",
    "engine_core_mapped_worktree_binaries",
    "environment",
    "health_during_stall",
    "hook_ready",
    "hook_error_kind",
    "hook_instance_count",
    "implementation_sha256",
    "observer_authorization",
    "private_server_log_sha256",
    "progress",
    "progress_count_at_release",
    "prompt_tokens",
    "recovered_after_release",
    "release_observed",
    "release_offset_seconds",
    "request_sha256",
    "schema_version",
    "server_command_sha256",
    "cleanup",
    "source",
    "source_arm",
    "stack",
    "stalled_before_release",
    "stream_error_kind",
    "trigger",
    "yama_ptrace_scope",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def maximum_gap(offsets: list[float]) -> float | None:
    if len(offsets) < 2:
        return None
    return max(b - a for a, b in zip(offsets, offsets[1:], strict=False))


def maximum_no_progress_gap(progress: dict[str, Any]) -> float | None:
    completed = progress["completed_offset_seconds"]
    offsets = progress["progress_offsets_seconds"]
    if completed is None or not offsets:
        return None
    boundaries = [0.0, *offsets, completed]
    return max(b - a for a, b in zip(boundaries, boundaries[1:], strict=False))


def verify_progress(record: dict[str, Any], label: str) -> None:
    require(
        set(record)
        == {
            "completed_offset_seconds",
            "progress_count",
            "progress_offsets_seconds",
        },
        f"{label}: unexpected progress fields",
    )
    offsets = record["progress_offsets_seconds"]
    require(isinstance(offsets, list), f"{label}: offsets are not a list")
    require(record["progress_count"] == len(offsets), f"{label}: count mismatch")
    require(len(offsets) >= 2, f"{label}: insufficient progress samples")
    require(
        all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value >= 0
            for value in offsets
        ),
        f"{label}: invalid progress offset",
    )
    require(offsets == sorted(offsets), f"{label}: offsets are not monotonic")
    completed = record["completed_offset_seconds"]
    require(
        isinstance(completed, (int, float))
        and not isinstance(completed, bool)
        and completed >= offsets[-1],
        f"{label}: invalid completion offset",
    )


def verify_cell(record: dict[str, Any], index: int) -> None:
    source_arm, trigger = CELLS[index]
    label = f"cell {index}"
    require(set(record) == SUMMARY_KEYS, f"{label}: unexpected summary fields")
    require(record["schema_version"] == 1, f"{label}: schema mismatch")
    require(record["cell_index"] == index, f"{label}: index mismatch")
    require(record["source_arm"] == source_arm, f"{label}: source mismatch")
    require(record["trigger"] == trigger, f"{label}: trigger mismatch")
    require(record["classification"] == "pass", f"{label}: not a pass")
    require(record["campaign_error_kind"] is None, f"{label}: campaign failed")
    require(record["engine_core_bound"] is True, f"{label}: PID not bound")
    require(record["hook_ready"] is True, f"{label}: hook not ready")
    require(record["hook_error_kind"] is None, f"{label}: hook error recorded")
    require(record["hook_instance_count"] == 1, f"{label}: hook count mismatch")
    require(record["release_observed"] is True, f"{label}: release not observed")
    require(record["stream_error_kind"] is None, f"{label}: stream failed")
    require(
        isinstance(record["prompt_tokens"], int)
        and not isinstance(record["prompt_tokens"], bool)
        and record["prompt_tokens"] >= 16,
        f"{label}: prompt did not reach one full block",
    )
    require(
        isinstance(record["completion_tokens"], int)
        and not isinstance(record["completion_tokens"], bool)
        and record["completion_tokens"] >= 32,
        f"{label}: output did not reach two blocks",
    )
    require(
        isinstance(record["accepted_batch_count"], int)
        and not isinstance(record["accepted_batch_count"], bool)
        and record["accepted_batch_count"] >= 1,
        f"{label}: trigger path was not reached",
    )
    require(
        isinstance(record["dropped_batch_count"], int)
        and not isinstance(record["dropped_batch_count"], bool)
        and record["dropped_batch_count"] >= 0,
        f"{label}: invalid drop count",
    )
    require(is_sha256(record["request_sha256"]), f"{label}: request hash invalid")
    require(
        is_sha256(record["build_identity_sha256"]),
        f"{label}: build identity hash invalid",
    )
    require(
        is_sha256(record["server_command_sha256"]),
        f"{label}: command hash invalid",
    )
    require(
        is_sha256(record["private_server_log_sha256"]),
        f"{label}: private log hash invalid",
    )
    cleanup = record["cleanup"]
    require(
        set(cleanup) == {"engine_core_gone", "process_group_gone", "termination"},
        f"{label}: cleanup fields mismatch",
    )
    require(cleanup["engine_core_gone"] is True, f"{label}: EngineCore survived")
    require(cleanup["process_group_gone"] is True, f"{label}: process group survived")
    require(
        cleanup["termination"] in {"exited", "terminated", "killed"},
        f"{label}: termination class invalid",
    )

    source = record["source"]
    require(
        source
        == {
            "base_commit": BASE_COMMIT,
            "fix_head": FIX_HEAD,
            "fix_patch_sha256": FIX_PATCH_SHA256,
            "tree": BASE_TREE if source_arm == "base" else FIX_TREE,
        },
        f"{label}: source identity mismatch",
    )
    implementation = record["implementation_sha256"]
    require(
        set(implementation) == set(HASHED_IMPLEMENTATION),
        f"{label}: implementation file set mismatch",
    )
    for name, path in HASHED_IMPLEMENTATION.items():
        require(implementation[name] == sha256(path), f"{label}: {name} hash mismatch")
    require(
        record["observer_authorization"] in {"pr_set_ptracer_observer", "yama_absent"},
        f"{label}: observer authorization mismatch",
    )
    scope = record["yama_ptrace_scope"]
    if record["observer_authorization"] == "yama_absent":
        require(scope is None, f"{label}: unexpected Yama scope")
    else:
        require(
            isinstance(scope, int) and not isinstance(scope, bool) and 0 <= scope <= 3,
            f"{label}: invalid Yama scope",
        )

    environment = record["environment"]
    require(
        set(environment)
        == {
            "cuda",
            "cuda_available",
            "gpu_capability",
            "gpu_count",
            "gpu_name",
            "import_matches_tree",
            "kv_events_relative_file",
            "kv_events_sha256",
            "python",
            "torch",
            "tree_kv_events_sha256",
            "vllm",
            "vllm_relative_file",
        },
        f"{label}: environment fields mismatch",
    )
    require(environment["cuda_available"] is True, f"{label}: CUDA unavailable")
    require(environment["gpu_count"] == 1, f"{label}: GPU count is not one")
    require(environment["import_matches_tree"] is True, f"{label}: import mismatch")
    require(
        environment["vllm_relative_file"] == "vllm/__init__.py",
        f"{label}: vLLM import path mismatch",
    )
    require(
        environment["kv_events_relative_file"] == "vllm/distributed/kv_events.py",
        f"{label}: kv_events import path mismatch",
    )
    require(
        is_sha256(environment["kv_events_sha256"])
        and environment["kv_events_sha256"] == environment["tree_kv_events_sha256"],
        f"{label}: imported kv_events hash mismatch",
    )
    require(
        environment["tree_kv_events_sha256"] == KV_EVENTS_SHA256[source_arm],
        f"{label}: source-specific kv_events hash mismatch",
    )
    require(
        record["engine_core_kv_events_sha256"] == environment["tree_kv_events_sha256"],
        f"{label}: EngineCore kv_events hash mismatch",
    )
    mapped = record["engine_core_mapped_worktree_binaries"]
    require(isinstance(mapped, dict) and mapped, f"{label}: no mapped binaries")
    require(
        all(
            isinstance(path, str) and path.startswith("vllm/") and is_sha256(digest)
            for path, digest in mapped.items()
        ),
        f"{label}: mapped binary identity invalid",
    )

    progress = record["progress"]
    verify_progress(progress, label)
    if trigger == "control":
        gap = maximum_gap(progress["progress_offsets_seconds"])
        require(gap is not None and gap < 2.0, f"{label}: control gap too large")
        require(record["dropped_batch_count"] == 0, f"{label}: control dropped events")
        require(record["stalled_before_release"] is False, f"{label}: control stalled")
        require(
            record["recovered_after_release"] is None,
            f"{label}: unexpected recovery claim",
        )
        require(record["health_during_stall"] is None, f"{label}: false stall health")
        require(record["release_offset_seconds"] is None, f"{label}: false release")
        require(record["progress_count_at_release"] == 0, f"{label}: false release")
    elif source_arm == "base":
        require(record["dropped_batch_count"] == 0, f"{label}: base recorded a drop")
        require(record["stalled_before_release"] is True, f"{label}: no stall")
        require(record["recovered_after_release"] is True, f"{label}: no recovery")
        require(
            record["health_during_stall"] in {"2xx", "non_2xx", "unavailable"},
            f"{label}: stall health missing",
        )
        require(
            isinstance(record["release_offset_seconds"], (int, float))
            and not isinstance(record["release_offset_seconds"], bool),
            f"{label}: release offset missing",
        )
        require(
            progress["completed_offset_seconds"] > record["release_offset_seconds"],
            f"{label}: completion did not follow release",
        )
        require(
            progress["progress_count"] > record["progress_count_at_release"],
            f"{label}: no progress after release",
        )
    else:
        require(record["dropped_batch_count"] > 0, f"{label}: no measured drops")
        require(record["stalled_before_release"] is False, f"{label}: fix stalled")
        require(
            record["recovered_after_release"] is None,
            f"{label}: unexpected recovery claim",
        )
        gap = maximum_no_progress_gap(progress)
        require(gap is not None and gap < 10.0, f"{label}: fix lost progress")
        require(record["health_during_stall"] is None, f"{label}: false stall health")
        require(
            isinstance(record["release_offset_seconds"], (int, float))
            and not isinstance(record["release_offset_seconds"], bool),
            f"{label}: release offset missing",
        )

    stack = record["stack"]
    require(
        set(stack) == {"available", "match", "raw_sha256"},
        f"{label}: stack fields mismatch",
    )
    if index == 3:
        require(stack["available"] is True, f"{label}: stack unavailable")
        require(stack["match"] is True, f"{label}: stack mismatch")
        require(is_sha256(stack["raw_sha256"]), f"{label}: stack hash invalid")
    else:
        require(
            stack == {"available": None, "match": None, "raw_sha256": None},
            f"{label}: unexpected stack claim",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    return parser.parse_args()


def verify_cross_cell(records: list[dict[str, Any]]) -> None:
    require(
        len({record["request_sha256"] for record in records}) == 1,
        "request changed across cells",
    )
    require(
        len({record["server_command_sha256"] for record in records}) == 1,
        "server command changed across cells",
    )
    shared_runtime_keys = {
        "cuda",
        "cuda_available",
        "gpu_capability",
        "gpu_count",
        "gpu_name",
        "python",
        "torch",
    }
    shared_runtimes = {
        json.dumps(
            {key: record["environment"][key] for key in shared_runtime_keys},
            sort_keys=True,
        )
        for record in records
    }
    require(len(shared_runtimes) == 1, "runtime environment changed across cells")
    for source_arm in ("base", "fix"):
        source_records = [
            record for record in records if record["source_arm"] == source_arm
        ]
        build_identities = {
            record["build_identity_sha256"] for record in source_records
        }
        require(
            len(build_identities) == 1,
            f"{source_arm} build identity changed across cells",
        )
        source_environments = {
            json.dumps(record["environment"], sort_keys=True)
            for record in source_records
        }
        require(
            len(source_environments) == 1,
            f"{source_arm} import identity changed across cells",
        )


def main() -> int:
    args = parse_args()
    records = []
    for index, filename in FILES.items():
        path = args.result_dir / filename
        require(path.is_file(), f"missing {filename}")
        record = json.loads(path.read_text(encoding="utf-8"))
        verify_cell(record, index)
        records.append(record)

    verify_cross_cell(records)
    print("PASS: vLLM #53859 Stage 1 four-cell contract verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
