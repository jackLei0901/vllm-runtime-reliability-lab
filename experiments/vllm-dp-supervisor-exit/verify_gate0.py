#!/usr/bin/env python3
"""Fail-closed verifier for the DP supervisor exit Gate 0."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_SOURCES = {
    "0bfc7a15d095fe83ecc82b50561a93c177fece2d": (
        "d10e59910d1ae934ed41fcd5593deb1c95c494a793fd20dc59ca736f7a4b9a97"
    ),
    "61950f9589938ffc73e4f4eaf4181c39f66afbaf": (
        "7c80f64bfe37feaebaaadadf5630cfa825035c034267951db0ca9a5a1e74588b"
    ),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    summary = json.loads(args.summary.read_text())
    require(summary["schema_version"] == 1, "unexpected schema version")
    require(
        summary["gate"] == "vllm-dp-supervisor-exit-gate0",
        "unexpected gate identifier",
    )
    require(summary["timeout_seconds"] == 90.0, "unexpected timeout")
    require(summary["platform"]["system"] == "Linux", "expected Linux")
    require(
        summary["protocol_sha256"] == sha256_file(root / "GATE0_PROTOCOL.md"),
        "protocol hash mismatch",
    )
    require(
        summary["runner_sha256"]
        == sha256_file(root / "gate0_dp_supervisor_exit.py"),
        "runner hash mismatch",
    )
    require(
        summary["vllm_commit"] in EXPECTED_SOURCES,
        "unexpected vLLM commit",
    )
    require(
        summary["subject_source_sha256"]
        == EXPECTED_SOURCES[summary["vllm_commit"]],
        "subject source hash mismatch",
    )
    require(
        summary["fault_tolerance_enabled"] is False,
        "fault tolerance must be explicitly disabled",
    )

    require(len(summary["cells"]) == 4, "expected exactly four cells")
    cells = {cell["case"]: cell for cell in summary["cells"]}
    require(
        set(cells)
        == {
            "intentional-sigterm-after-ready",
            "abnormal-child-exit-before-ready",
            "abnormal-child-exit-after-ready",
            "probe-failure-after-ready",
        },
        "unexpected cell set",
    )
    for cell in cells.values():
        require(cell["timed_out"] is False, f"{cell['case']} timed out")
        require(
            cell["process_returncode"] == 0,
            f"{cell['case']} process returned non-zero",
        )
        require(cell["subject"] is not None, f"{cell['case']} has no subject")
        require(
            cell["subject"]["entry_returned"] is True,
            f"{cell['case']} entry point did not return",
        )
        require(
            cell["subject"]["error"] is None,
            f"{cell['case']} retained an error",
        )

    intentional = cells["intentional-sigterm-after-ready"]["subject"]
    require(intentional["server_started"] is True, "signal server not started")
    require(intentional["signal_sent"] is True, "SIGTERM not sent")
    require(
        intentional["handled_signals"] == ["SIGTERM"],
        "registered handler did not handle exactly one SIGTERM",
    )
    require(intentional["child_exitcode"] == -15, "unexpected signal child exit")

    before_ready = cells["abnormal-child-exit-before-ready"]["subject"]
    require(before_ready["server_started"] is False, "early server started")
    require(before_ready["child_exitcode"] == 17, "unexpected early child exit")
    require(before_ready["handled_signals"] == [], "early failure handled signal")

    after_ready = cells["abnormal-child-exit-after-ready"]["subject"]
    require(after_ready["server_started"] is True, "late server not started")
    require(after_ready["child_exitcode"] == 17, "unexpected late child exit")
    require(after_ready["handled_signals"] == [], "late failure handled signal")

    probe_failure = cells["probe-failure-after-ready"]["subject"]
    require(probe_failure["server_started"] is True, "probe server not started")
    require(
        probe_failure["probe_failure_triggered"] is True,
        "probe failure was not observed",
    )
    require(
        probe_failure["child_exitcode"] == -15,
        "unexpected probe child exit",
    )
    require(
        probe_failure["handled_signals"] == [],
        "probe failure handled signal",
    )
    print(
        "PASS: real SIGTERM, child failure before/after readiness, and a post-ready "
        "probe failure all return supervisor process status 0."
    )


if __name__ == "__main__":
    main()
