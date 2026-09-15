"""Verify the Stage 1 pre-execution freeze and private input hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST = HERE / "STAGE1_FREEZE.json"

EXPECTED_FILES = {
    "experiments/vllm-zmq-event-backpressure/STAGE1_PROTOCOL_DRAFT.md",
    "experiments/vllm-zmq-event-backpressure/STAGE1_RUNBOOK_DRAFT.md",
    "experiments/vllm-zmq-event-backpressure/stage1_build_identity.py",
    "experiments/vllm-zmq-event-backpressure/stage1_campaign.py",
    "experiments/vllm-zmq-event-backpressure/stage1_contract.py",
    "experiments/vllm-zmq-event-backpressure/stage1_plugin/pyproject.toml",
    (
        "experiments/vllm-zmq-event-backpressure/stage1_plugin/src/"
        "dfx_stage1_backpressure/__init__.py"
    ),
    "experiments/vllm-zmq-event-backpressure/verify_stage1_build_identity.py",
    "experiments/vllm-zmq-event-backpressure/verify_stage1_freeze.py",
    "experiments/vllm-zmq-event-backpressure/verify_stage1_results.py",
    (
        "results/vllm-zmq-backpressure-stage1-build-20260915/"
        "stage1-build-base.json"
    ),
    (
        "results/vllm-zmq-backpressure-stage1-build-20260915/"
        "stage1-build-fix.json"
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
        manifest.get("freeze_id") == "vllm-zmq-stage1-pre-execution-2026-09-15",
        "unexpected Stage 1 freeze identifier",
    )
    require(manifest.get("status") == "pre-execution", "freeze status changed")
    require(
        manifest.get("cell_order")
        == ["base/control", "fix/control", "base/pause", "fix/pause"],
        "cell order changed",
    )
    require(
        manifest.get("build_checkpoint_policy")
        == "byte-identical-before-first-and-after-last",
        "build checkpoint policy changed",
    )
    require(
        manifest.get("source_trees")
        == {
            "base": "b7061e73a6ed4773e16bd2ae3acf47aebfd1342d",
            "fix": "46bc6e191b14ce12a04827454b4588ea5d3a435f",
        },
        "source trees changed",
    )

    files = manifest.get("files")
    require(isinstance(files, dict), "frozen file map is missing")
    require(set(files) == EXPECTED_FILES, "frozen file set changed")
    for relative, expected in sorted(files.items()):
        path = ROOT / relative
        require(path.is_file(), f"frozen file is missing: {relative}")
        require(sha256(path) == expected, f"frozen file changed: {relative}")

    private = manifest.get("private_inputs")
    require(
        isinstance(private, dict)
        and set(private) == {"request.json", "server-command.json"},
        "private input set changed",
    )
    for name, expected in sorted(private.items()):
        path = args.private_dir / name
        require(path.is_file(), f"private input is missing: {name}")
        require(sha256(path) == expected, f"private input changed: {name}")

    print(f"PASS ({len(files)} frozen files and {len(private)} private inputs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
