#!/usr/bin/env python3
"""Normalize and score one Stage B A/B/A2 capture without publishing raw stacks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from dfxlab.native_evidence import compare_capture_triplet, evaluate_attribution

FRAME_CLASSES = (
    "python:publisher",
    "python:queue-put",
    "native:condition-wait",
)
ANCHORS = {
    "python:publisher": re.compile(
        r"(?:\bpublish\b.*vllm/distributed/kv_events\.py|"
        r"vllm/distributed/kv_events\.py.*\bin publish\b)"
    ),
    "python:queue-put": re.compile(r"(?:\bput \(queue\.py:140\)|queue\.py\", line 140, in put)"),
    "native:condition-wait": re.compile(r"\bPyThread_acquire_lock_timed\b"),
}
RAW_NAMES = {
    "py-spy-a": "py-spy-a.txt",
    "pystack-b": "pystack-b.txt",
    "py-spy-a2": "py-spy-a2.txt",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_stack(raw: str) -> dict[str, Any]:
    blocks = re.split(r"(?m)^(?=Thread \d+|Traceback for thread \d+)", raw)
    matches: list[dict[str, int]] = []
    for block in blocks:
        positions = {
            name: match.start()
            for name, pattern in ANCHORS.items()
            if (match := pattern.search(block)) is not None
        }
        if set(positions) == set(FRAME_CLASSES):
            order = [name for name, _ in sorted(positions.items(), key=lambda x: x[1])]
            if order in [list(FRAME_CLASSES), list(reversed(FRAME_CLASSES))]:
                matches.append(positions)
    if len(matches) > 1:
        raise ValueError("multiple threads satisfy the queue-wait predicates")
    frames = list(FRAME_CLASSES) if matches else []
    return {
        "thread_ref": "queue-wait-thread" if frames else "no-matching-thread",
        "execution_domain": "mixed" if frames else "unknown",
        "gil_state": "unknown",
        "ordered_frame_classes": frames,
        "lifecycle_facts": [],
    }


def score(
    capture_record: dict[str, Any],
    private_dir: Path,
    rule: dict[str, Any],
    cell: str,
) -> dict[str, Any]:
    by_label = {item["label"]: item["capture"] for item in capture_record["captures"]}
    if set(by_label) != set(RAW_NAMES):
        raise ValueError("capture labels do not match A/B/A2")
    attributions = []
    for label in capture_record["sequence"]:
        raw = (private_dir / RAW_NAMES[label]).read_bytes()
        expected = by_label[label]["outcome"]["raw_output_sha256"]
        if sha256_bytes(raw) != expected:
            raise ValueError(f"raw digest mismatch: {label}")
        observation = normalize_stack(raw.decode("utf-8", errors="replace"))
        attribution = evaluate_attribution(
            by_label[label], observation, capture_record["target_runtime"], [rule]
        )
        attributions.append({"label": label, "attribution": attribution})
    pairing = compare_capture_triplet(
        *(item["attribution"] for item in attributions)
    )
    matches = [item["attribution"]["rule_match"] for item in attributions]
    if cell == "control":
        passed = matches == ["unmatched", "unmatched", "unmatched"] and pairing == {
            "pairing_result": "not_scorable",
            "not_scorable_reason": "no_admitted_rule",
            "stability_control_passed": True,
            "frame_sequence_compared": False,
            "coverage_equal": True,
        }
        result = "negative_control_passed" if passed else "negative_control_failed"
    else:
        passed = pairing["pairing_result"] == "interchangeable"
        result = "fault_pair_passed" if passed else "fault_pair_failed"
    return {
        "schema_version": "stage-b-v2-scored-pair-v0",
        "cell": cell,
        "result": result,
        "rule_set_id": rule["rule_set_id"],
        "attributions": attributions,
        "pairing": pairing,
        "raw_stack_published": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-record", type=Path, required=True)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--rule", type=Path, required=True)
    parser.add_argument("--cell", choices=("control", "fault"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = score(
            json.loads(args.capture_record.read_text(encoding="utf-8")),
            args.private_dir,
            json.loads(args.rule.read_text(encoding="utf-8")),
            args.cell,
        )
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(result["result"])
    return 0 if result["result"].endswith("_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
