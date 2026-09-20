from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

STACK_ERROR_KINDS = {
    "binary_missing",
    "timeout",
    "permission_denied",
    "execution_failed",
    "empty_output",
}


def disabled_stack() -> dict[str, Any]:
    return {"state": "disabled", "producer": None}


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def _version(binary: str) -> str | None:
    try:
        completed = subprocess.run(
            [binary, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    first_line = completed.stdout.splitlines()[0] if completed.stdout else ""
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,63}", first_line)
    return tokens[-1] if tokens else None


def _yama_scope() -> int | None:
    path = Path("/proc/sys/kernel/yama/ptrace_scope")
    try:
        return int(path.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return None


def _producer(
    *,
    version: str | None,
    binary_sha256: str | None,
    exit_status: int | None,
    output_produced: bool,
    raw_output_sha256: str | None,
    error_kind: str | None,
) -> dict[str, Any]:
    return {
        "sampler_name": "py-spy",
        "sampler_version": version,
        "binary_sha256": binary_sha256,
        "platform": platform.system(),
        "yama_ptrace_scope": _yama_scope(),
        "exit_status": exit_status,
        "output_produced": output_produced,
        "raw_output_sha256": raw_output_sha256,
        "error_kind": error_kind,
    }


def capture_stack(
    pid: int, private_output: Path, timeout: float = 5.0
) -> dict[str, Any]:
    """Run one fixed py-spy command and return only bounded producer identity."""

    if pid <= 0 or timeout <= 0:
        raise ValueError("pid and timeout must be positive")
    binary = shutil.which("py-spy")
    if binary is None:
        return {
            "state": "unavailable",
            "producer": _producer(
                version=None,
                binary_sha256=None,
                exit_status=None,
                output_produced=False,
                raw_output_sha256=None,
                error_kind="binary_missing",
            ),
        }
    version = _version(binary)
    binary_sha256 = _sha256(Path(binary))
    private_output.parent.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        private_output.parent.chmod(0o700)
    try:
        private_output.unlink(missing_ok=True)
    except OSError:
        return {
            "state": "unavailable",
            "producer": _producer(
                version=version,
                binary_sha256=binary_sha256,
                exit_status=None,
                output_produced=False,
                raw_output_sha256=None,
                error_kind="execution_failed",
            ),
        }
    try:
        completed = subprocess.run(
            [binary, "dump", "--pid", str(pid), "--output", str(private_output)],
            check=False,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        error_kind = "timeout"
        exit_status = None
    except PermissionError:
        error_kind = "permission_denied"
        exit_status = None
    except (OSError, subprocess.SubprocessError):
        error_kind = "execution_failed"
        exit_status = None
    else:
        exit_status = completed.returncode
        error_kind = None if completed.returncode == 0 else "execution_failed"
    output_produced = private_output.is_file() and private_output.stat().st_size > 0
    if output_produced and os.name != "nt":
        private_output.chmod(0o600)
    raw_digest = _sha256(private_output) if output_produced else None
    if error_kind is None and not output_produced:
        error_kind = "empty_output"
    state = "produced" if error_kind is None and output_produced else "unavailable"
    return {
        "state": state,
        "producer": _producer(
            version=version,
            binary_sha256=binary_sha256,
            exit_status=exit_status,
            output_produced=output_produced,
            raw_output_sha256=raw_digest,
            error_kind=error_kind,
        ),
    }
