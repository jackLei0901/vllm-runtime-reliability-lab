from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

MANIFEST_NAME = "replay-manifest.json"
MANIFEST_KEYS = {
    "case_id",
    "files",
    "kind",
    "schema_version",
    "source_links",
    "title",
}
RECORD_KEYS = {
    "accepted_batch_count",
    "build_identity_sha256",
    "campaign_error_kind",
    "cell_index",
    "classification",
    "cleanup",
    "completion_tokens",
    "dropped_batch_count",
    "engine_core_bound",
    "engine_core_kv_events_sha256",
    "engine_core_mapped_worktree_binaries",
    "environment",
    "health_during_stall",
    "hook_error_kind",
    "hook_instance_count",
    "hook_ready",
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
    "source",
    "source_arm",
    "stack",
    "stalled_before_release",
    "stream_error_kind",
    "trigger",
    "yama_ptrace_scope",
}
CELLS = {
    "base_control": (1, "base", "control"),
    "fix_control": (2, "fix", "control"),
    "base_pause": (3, "base", "pause"),
    "fix_pause": (4, "fix", "pause"),
}


class ReplayError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReplayError(f"cannot read {path.name}: {error}") from error
    require(isinstance(value, dict), f"{path.name}: top level must be an object")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_progress(record: dict[str, Any], label: str) -> None:
    progress = record["progress"]
    require(
        isinstance(progress, dict)
        and set(progress)
        == {"completed_offset_seconds", "progress_count", "progress_offsets_seconds"},
        f"{label}: progress shape changed",
    )
    offsets = progress["progress_offsets_seconds"]
    require(isinstance(offsets, list) and offsets, f"{label}: progress is empty")
    require(
        all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in offsets
        ),
        f"{label}: progress offsets are invalid",
    )
    require(offsets == sorted(offsets), f"{label}: progress is not monotonic")
    require(
        progress["progress_count"] == len(offsets), f"{label}: progress count mismatch"
    )
    require(
        progress["completed_offset_seconds"] >= offsets[-1],
        f"{label}: completion precedes final progress",
    )


def verify_record(record: dict[str, Any], label: str) -> None:
    index, source_arm, trigger = CELLS[label]
    require(set(record) == RECORD_KEYS, f"{label}: record shape changed")
    require(record["schema_version"] == 1, f"{label}: schema mismatch")
    require(record["cell_index"] == index, f"{label}: cell index mismatch")
    require(record["source_arm"] == source_arm, f"{label}: source arm mismatch")
    require(record["trigger"] == trigger, f"{label}: trigger mismatch")
    require(record["classification"] == "pass", f"{label}: cell did not pass")
    require(record["campaign_error_kind"] is None, f"{label}: campaign error recorded")
    require(record["engine_core_bound"] is True, f"{label}: EngineCore was not bound")
    require(record["hook_ready"] is True, f"{label}: fault hook was not ready")
    require(record["hook_error_kind"] is None, f"{label}: fault hook error recorded")
    require(record["release_observed"] is True, f"{label}: trigger release missing")
    require(record["stream_error_kind"] is None, f"{label}: request stream failed")
    require(record["completion_tokens"] >= 32, f"{label}: request did not complete")
    verify_progress(record, label)

    cleanup = record["cleanup"]
    require(
        isinstance(cleanup, dict)
        and set(cleanup) == {"engine_core_gone", "process_group_gone", "termination"}
        and cleanup["engine_core_gone"] is True
        and cleanup["process_group_gone"] is True,
        f"{label}: cleanup contract failed",
    )

    if trigger == "control":
        require(record["stalled_before_release"] is False, f"{label}: control stalled")
        require(record["dropped_batch_count"] == 0, f"{label}: control dropped events")
    elif source_arm == "base":
        require(record["stalled_before_release"] is True, "base_pause: stall missing")
        require(
            record["health_during_stall"] == "2xx", "base_pause: health was not green"
        )
        require(
            record["recovered_after_release"] is True, "base_pause: recovery missing"
        )
        stack = record["stack"]
        require(
            stack
            == {
                "available": True,
                "match": True,
                "raw_sha256": stack.get("raw_sha256"),
            }
            and isinstance(stack["raw_sha256"], str)
            and len(stack["raw_sha256"]) == 64,
            "base_pause: blocking stack evidence missing",
        )
    else:
        require(
            record["stalled_before_release"] is False, "fix_pause: progress stalled"
        )
        require(record["dropped_batch_count"] > 0, "fix_pause: trade-off not measured")
        require(record["health_during_stall"] is None, "fix_pause: false stall health")


def replay(result_dir: Path) -> list[str]:
    require(result_dir.is_dir(), f"not a result directory: {result_dir}")
    manifest_path = result_dir / MANIFEST_NAME
    require(manifest_path.is_file(), f"missing {MANIFEST_NAME}")
    manifest = load_json(manifest_path)
    require(set(manifest) == MANIFEST_KEYS, "manifest shape changed")
    require(manifest["schema_version"] == 1, "unsupported replay manifest")
    require(
        manifest["kind"] == "vllm_zmq_backpressure_four_cell_v1",
        "unsupported replay kind",
    )
    require(manifest["case_id"] == "vllm-53859-stage1-r3", "case ID mismatch")
    require(
        manifest["title"]
        == "vLLM health-green no-progress under KV-event backpressure",
        "case title mismatch",
    )
    require(
        manifest["source_links"]
        == [
            "https://github.com/vllm-project/vllm/issues/53859",
            "https://github.com/vllm-project/vllm/pull/53883",
        ],
        "source links changed",
    )
    files = manifest["files"]
    require(
        isinstance(files, dict) and set(files) == set(CELLS),
        "manifest cell set changed",
    )

    expected_json_files = {MANIFEST_NAME}
    records: dict[str, dict[str, Any]] = {}
    for label in CELLS:
        entry = files[label]
        require(
            isinstance(entry, dict) and set(entry) == {"path", "sha256"},
            f"{label}: manifest entry shape changed",
        )
        require(
            isinstance(entry["path"], str) and isinstance(entry["sha256"], str),
            f"{label}: manifest entry types changed",
        )
        path = result_dir / entry["path"]
        require(
            path.parent == result_dir,
            f"{label}: path must stay inside result directory",
        )
        require(path.is_file(), f"{label}: evidence file missing")
        require(sha256(path) == entry["sha256"], f"{label}: evidence hash mismatch")
        expected_json_files.add(path.name)
        record = load_json(path)
        verify_record(record, label)
        records[label] = record

    actual_json_files = {path.name for path in result_dir.glob("*.json")}
    require(
        actual_json_files == expected_json_files, "unexpected JSON evidence file set"
    )
    require(
        len({record["request_sha256"] for record in records.values()}) == 1,
        "request identity changed across cells",
    )
    require(
        len({record["server_command_sha256"] for record in records.values()}) == 1,
        "server command changed across cells",
    )
    for arm in ("base", "fix"):
        arm_records = [
            record for record in records.values() if record["source_arm"] == arm
        ]
        require(
            len({record["build_identity_sha256"] for record in arm_records}) == 1,
            f"{arm}: build identity changed between control and pause",
        )

    base = records["base_pause"]
    fixed = records["fix_pause"]
    return [
        f"CASE: {manifest['title']}",
        "PASS: evidence file set and SHA-256 identities verified",
        "PASS: controls completed without a stall or dropped event batch",
        "PASS: EngineCore process remained alive during the injected stall",
        (
            f"PASS: /health remained {base['health_during_stall']} "
            "while token progress stopped"
        ),
        "STACK: EngineCore -> ZmqEventPublisher.publish -> Queue.put",
        "FIX ARM: token progress completed under the same trigger",
        f"TRADE-OFF: {fixed['dropped_batch_count']} event batches dropped",
        (
            "BOUNDARY: replay verifies archived evidence; "
            "it does not rerun the GPU experiment"
        ),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed, CPU-only replay of a published lab result."
    )
    parser.add_argument("result_dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        lines = replay(args.result_dir.resolve())
    except ReplayError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
