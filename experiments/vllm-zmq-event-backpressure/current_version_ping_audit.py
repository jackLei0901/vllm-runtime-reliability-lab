"""Compact, privacy-bounded audit of the four current-wheel test cells."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--suffix", choices=("1", "2"), default="1")
    parser.add_argument("--cell-prefix", default="dfx-current-cell-")
    parser.add_argument("--source-equivalent-root", type=Path)
    args = parser.parse_args()
    cells = {}
    for letter in ("A", "B", "C", "D"):
        name = f"{letter}{args.suffix}"
        cell = args.root / f"{args.cell_prefix}{name}"
        summary_path = cell / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        before = summary.get("progress_before_release") or {}
        after = summary.get("progress_after_release") or {}
        health = summary.get("health_during_stall") or summary.get(
            "health_during_control"
        )
        counts = summary.get("queue_counts") or {}
        cells[name] = {
            "result": summary["result"],
            "ping": summary["ping"],
            "trigger": summary["trigger"],
            "health": health,
            "stall_confirmed": summary.get("stall_confirmed"),
            "identity_stable": summary["engine_identity_stable_at_end"],
            "source_mode": summary.get("source_mode", "test_plugin"),
            "source_sha256": summary.get("source_sha256"),
            "plugin_marker_count": summary.get("ping_plugin_marker_count", 0),
            "accepted_batches": counts.get("accepted_batch_count"),
            "progress_before_release": len(before.get("progress_offsets_seconds", [])),
            "progress_after_release": len(after.get("progress_offsets_seconds", [])),
            "completed_after_release": after.get("complete"),
            "usage": after.get("usage"),
            "summary_sha256": digest(summary_path),
            "private_server_log_sha256": digest(cell / "server.log"),
        }
    expected = {
        f"A{args.suffix}": ("off", "control", 200),
        f"B{args.suffix}": ("on", "control", 200),
        f"C{args.suffix}": ("off", "pause", 200),
        f"D{args.suffix}": ("on", "pause", 503),
    }
    for name, (ping, trigger, status) in expected.items():
        record = cells[name]
        if (
            record["ping"] != ping
            or record["trigger"] != trigger
            or record["health"].get("kind") != "http"
            or record["health"].get("status") != status
            or record["result"] != "expected_health_response_observed"
            or record["identity_stable"] is not True
            or (args.source_equivalent_root is None and record["plugin_marker_count"] < 2)
            or (args.source_equivalent_root is not None and record["source_mode"] != "pr36451_python_source_equivalent")
        ):
            raise SystemExit(f"cell {name} did not meet its stated checks")
    for name in (f"C{args.suffix}", f"D{args.suffix}"):
        record = cells[name]
        if (
            record["stall_confirmed"] is not True
            or record["accepted_batches"] is None
            or record["accepted_batches"] < 1
            or record["progress_before_release"] < 2
            or record["progress_after_release"] <= record["progress_before_release"]
            or record["completed_after_release"] is not True
        ):
            raise SystemExit(f"pause cell {name} lacks mechanism/recovery evidence")
    sources = {
        "stage1_plugin_sha256": digest(
            args.packages / "dfx_stage1_backpressure" / "__init__.py"
        ),
        "model_config_sha256": digest(args.model / "config.json"),
    }
    if args.source_equivalent_root is None:
        sources["ping_plugin_sha256"] = digest(
            args.packages / "dfx_health_ping_poc" / "__init__.py"
        )
    else:
        sources["source_sha256"] = {
            name: digest(args.source_equivalent_root / "vllm" / name)
            for name in (
                "envs.py",
                "v1/engine/async_llm.py",
                "v1/engine/core.py",
                "v1/engine/core_client.py",
            )
        }
        if any(record["source_sha256"] != sources["source_sha256"] for record in cells.values()):
            raise SystemExit("source identity differs between cells or from overlay")
    print(json.dumps({"audit": "pass", "cells": cells, "sources": sources}, sort_keys=True))


if __name__ == "__main__":
    main()
