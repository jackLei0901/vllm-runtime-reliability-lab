#!/usr/bin/env python3
"""Capture the Block 5 A/B/A2 producer sequence against a held Linux target."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dfxlab.native_evidence import validate_capture, validate_target_runtime
from dfxlab.native_producers import capture_native_stack

SEQUENCE = (
    ("py-spy-a", "py-spy"),
    ("pystack-b", "pystack"),
    ("py-spy-a2", "py-spy"),
)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
    if os.name != "nt":
        path.chmod(0o600)


def run(args: argparse.Namespace) -> dict[str, Any]:
    target_runtime = validate_target_runtime(
        {
            "vllm_version": args.vllm_version,
            "pytorch_version": args.pytorch_version,
            "pytorch_backend": args.pytorch_backend,
            "nccl_version": args.nccl_version,
            "topology": args.topology,
        }
    )
    if args.output.exists():
        raise ValueError("output directory already exists")
    private = args.output / "private"
    private.mkdir(parents=True, mode=0o700)
    captures = []
    for label, implementation in SEQUENCE:
        capture = capture_native_stack(
            implementation=implementation,
            pid=args.pid,
            expected_start_ticks=args.start_ticks,
            private_output=private / f"{label}.txt",
            declared_role=args.role,
            declared_rank=args.rank,
            timeout=args.timeout,
            max_output_bytes=args.max_output_kib * 1024,
        )
        validate_capture(capture)
        captures.append({"label": label, "capture": capture})
    record = {
        "schema_version": "native-producer-pair-capture-v0",
        "target_runtime": target_runtime,
        "sequence": [label for label, _ in SEQUENCE],
        "captures": captures,
        "pairing_status": "normalization_pending",
        "claim": None,
    }
    atomic_json(args.output / "capture-record.json", record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--start-ticks", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", default="engine_core")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--vllm-version", required=True)
    parser.add_argument("--pytorch-version", required=True)
    parser.add_argument("--pytorch-backend", required=True)
    parser.add_argument("--nccl-version", required=True)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--max-output-kib", type=int, default=1024)
    args = parser.parse_args()
    try:
        record = run(args)
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    outcomes = [
        f"{item['label']}={item['capture']['outcome']['attempt_stage']}/"
        f"{item['capture']['outcome']['outcome_code']}"
        for item in record["captures"]
    ]
    print("CAPTURED: " + ", ".join(outcomes))
    print("NO CLAIM: normalization and stability evaluation are still required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
