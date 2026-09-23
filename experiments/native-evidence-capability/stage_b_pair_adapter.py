#!/usr/bin/env python3
"""Adapt the Stage 1 py-spy slot to the bounded Stage B A/B/A2 sequence."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dfxlab.native_producers import linux_start_ticks

REQUIRED_ENV = {
    "DFX_STAGE_B_NCCL_VERSION",
    "DFX_STAGE_B_OUTPUT",
    "DFX_STAGE_B_PYTORCH_BACKEND",
    "DFX_STAGE_B_PYTORCH_VERSION",
    "DFX_STAGE_B_REPO",
    "DFX_STAGE_B_TOPOLOGY",
    "DFX_STAGE_B_VLLM_VERSION",
}


def parse_campaign_invocation(argv: list[str]) -> int:
    if len(argv) != 3 or argv[0] != "dump" or argv[1] != "--pid":
        raise ValueError("expected: dump --pid PID")
    try:
        pid = int(argv[2])
    except ValueError as exc:
        raise ValueError("PID must be an integer") from exc
    if pid <= 0:
        raise ValueError("PID must be positive")
    return pid


def required_environment(environ: dict[str, str]) -> dict[str, str]:
    missing = sorted(name for name in REQUIRED_ENV if not environ.get(name))
    if missing:
        raise ValueError("missing environment: " + ",".join(missing))
    return {name: environ[name] for name in REQUIRED_ENV}


def main(argv: list[str] | None = None) -> int:
    try:
        pid = parse_campaign_invocation(list(sys.argv[1:] if argv is None else argv))
        values = required_environment(dict(os.environ))
        start_ticks = linux_start_ticks(pid)
        if start_ticks is None:
            raise ValueError("target identity is not live")
        output = Path(values["DFX_STAGE_B_OUTPUT"])
        if output.exists():
            raise ValueError("pair output already exists")
        runner = (
            Path(values["DFX_STAGE_B_REPO"])
            / "experiments"
            / "native-evidence-capability"
            / "run_stack_pair.py"
        )
        command = [
            sys.executable,
            str(runner),
            "--pid",
            str(pid),
            "--start-ticks",
            start_ticks,
            "--output",
            str(output),
            "--vllm-version",
            values["DFX_STAGE_B_VLLM_VERSION"],
            "--pytorch-version",
            values["DFX_STAGE_B_PYTORCH_VERSION"],
            "--pytorch-backend",
            values["DFX_STAGE_B_PYTORCH_BACKEND"],
            "--nccl-version",
            values["DFX_STAGE_B_NCCL_VERSION"],
            "--topology",
            values["DFX_STAGE_B_TOPOLOGY"],
            "--timeout",
            "5",
            "--max-output-kib",
            "1024",
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            timeout=25,
        )
        if completed.returncode != 0:
            sys.stderr.buffer.write(completed.stdout + completed.stderr)
            return completed.returncode or 1
        first_raw = output / "private" / "py-spy-a.txt"
        if not first_raw.is_file() or first_raw.stat().st_size == 0:
            raise ValueError("first py-spy capture is missing")
        sys.stdout.buffer.write(first_raw.read_bytes())
        return 0
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"stage-b-pair-adapter: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
