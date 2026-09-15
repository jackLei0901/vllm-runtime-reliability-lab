"""Verify the Stage 1 R2 freeze after the dependency qualification failure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST = HERE / "STAGE1_R2_FREEZE.json"
EXPECTED_FILES = {
    "experiments/vllm-zmq-event-backpressure/STAGE1_FREEZE.json",
    "experiments/vllm-zmq-event-backpressure/STAGE1_PROTOCOL_DRAFT.md",
    "experiments/vllm-zmq-event-backpressure/STAGE1_R1_CONTROL_FAILURE_2026-09-15.md",
    "experiments/vllm-zmq-event-backpressure/STAGE1_R2_RUNBOOK.md",
    "experiments/vllm-zmq-event-backpressure/stage1_build_identity.py",
    "experiments/vllm-zmq-event-backpressure/stage1_campaign.py",
    "experiments/vllm-zmq-event-backpressure/stage1_contract.py",
    "experiments/vllm-zmq-event-backpressure/stage1_plugin/pyproject.toml",
    (
        "experiments/vllm-zmq-event-backpressure/stage1_plugin/src/"
        "dfx_stage1_backpressure/__init__.py"
    ),
    "experiments/vllm-zmq-event-backpressure/stage1a_cpu_preflight.py",
    "experiments/vllm-zmq-event-backpressure/verify_stage1_build_identity.py",
    "experiments/vllm-zmq-event-backpressure/verify_stage1_r2_freeze.py",
    "experiments/vllm-zmq-event-backpressure/verify_stage1_results.py",
    "experiments/vllm-zmq-event-backpressure/verify_stage1a.py",
    (
        "results/vllm-zmq-backpressure-stage1-build-r2-20260915/"
        "stage1-build-base.json"
    ),
    (
        "results/vllm-zmq-backpressure-stage1-build-r2-20260915/"
        "stage1-build-fix.json"
    ),
    (
        "results/vllm-zmq-backpressure-stage1-r1-control-failure-20260915/"
        "cell-1-base-control.json"
    ),
    (
        "results/vllm-zmq-backpressure-stage1a-r2-20260915/"
        "stage1a-base.json"
    ),
    (
        "results/vllm-zmq-backpressure-stage1a-r2-20260915/"
        "stage1a-fix.json"
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--private-dir",
        type=Path,
        default=HERE / ".stage1-private",
    )
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    require(
        manifest.get("freeze_id")
        == "vllm-zmq-stage1-r2-pre-execution-2026-09-15",
        "unexpected R2 freeze identifier",
    )
    require(manifest.get("status") == "pre-execution", "R2 status changed")
    require(
        manifest.get("supersedes")
        == "vllm-zmq-stage1-pre-execution-2026-09-15",
        "R2 predecessor changed",
    )
    require(
        manifest.get("change_scope") == "add-nvidia-ml-py-runtime-dependency",
        "R2 change scope changed",
    )
    require(
        manifest.get("cell_order")
        == ["base/control", "fix/control", "base/pause", "fix/pause"],
        "R2 cell order changed",
    )
    require(
        manifest.get("build_checkpoint_policy")
        == "byte-identical-before-first-and-after-last",
        "R2 checkpoint policy changed",
    )

    files = manifest.get("files")
    require(isinstance(files, dict), "R2 file map is missing")
    require(set(files) == EXPECTED_FILES, "R2 frozen file set changed")
    for relative, expected in sorted(files.items()):
        path = ROOT / relative
        require(path.is_file(), f"R2 frozen file is missing: {relative}")
        require(sha256(path) == expected, f"R2 frozen file changed: {relative}")

    private = manifest.get("private_inputs")
    require(
        isinstance(private, dict)
        and set(private) == {"request.json", "server-command.json"},
        "R2 private input set changed",
    )
    for name, expected in sorted(private.items()):
        path = args.private_dir / name
        require(path.is_file(), f"R2 private input is missing: {name}")
        require(sha256(path) == expected, f"R2 private input changed: {name}")

    print(f"PASS ({len(files)} R2 frozen files and {len(private)} private inputs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
