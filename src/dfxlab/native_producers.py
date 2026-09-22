from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from dfxlab.native_evidence import CAPTURE_SCHEMA, validate_capture

IMPLEMENTATIONS = {"py-spy", "pystack"}
_PERMISSION_MARKERS = (
    b"permission denied",
    b"operation not permitted",
)


def linux_start_ticks(pid: int) -> str | None:
    """Read Linux process start ticks without trusting command-line identity."""

    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        close = raw.rfind(")")
        fields = raw[close + 2 :].split()
        value = fields[19]
    except (OSError, UnicodeError, IndexError, ValueError):
        return None
    return value if value.isdigit() and int(value) > 0 else None


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
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = completed.stdout or completed.stderr
    tokens = (
        re.findall(rb"[A-Za-z0-9][A-Za-z0-9._+-]{0,63}", output.splitlines()[0])
        if output
        else []
    )
    return tokens[-1].decode("ascii", errors="ignore") if tokens else None


def _command(implementation: str, binary: str, pid: int) -> list[str]:
    if implementation == "pystack":
        return [binary, "remote", "--no-color", "--native-all", str(pid)]
    if implementation == "py-spy":
        return [binary, "dump", "--pid", str(pid), "--native", "--idle"]
    raise ValueError("unknown stack producer implementation")


def _limit_output(max_output_bytes: int):
    def apply_limit() -> None:
        import resource

        resource.setrlimit(resource.RLIMIT_FSIZE, (max_output_bytes, max_output_bytes))

    return apply_limit


def _record(
    *,
    identity: str,
    post_identity: str | None,
    role: str,
    rank: int | None,
    start_ns: int,
    end_ns: int,
    producer_timeout_ns: int,
    coordinator_timeout_ns: int,
    max_output_bytes: int,
    implementation: str,
    version: str | None,
    binary_digest: str | None,
    stage: str,
    outcome: str,
    raw_digest: str | None,
) -> dict[str, Any]:
    value = {
        "schema_version": CAPTURE_SCHEMA,
        "subject": {
            "process_identity_kind": "linux_proc_start_ticks",
            "process_identity_value": identity,
            "post_capture_identity_value": post_identity,
            "declared_role": role,
            "declared_rank": rank,
        },
        "window": {
            "start_monotonic_ns": start_ns,
            "end_monotonic_ns": end_ns,
            "producer_timeout_ns": producer_timeout_ns,
            "coordinator_timeout_ns": coordinator_timeout_ns,
            "max_output_bytes": max_output_bytes,
        },
        "producer": {
            "kind": "stack_snapshot",
            "implementation_name": implementation,
            "implementation_version": version,
            "binary_sha256": binary_digest,
            "platform": platform.system().lower(),
        },
        "outcome": {
            "attempt_stage": stage,
            "outcome_code": outcome,
            "raw_output_sha256": raw_digest,
        },
    }
    return validate_capture(value)


def capture_native_stack(
    *,
    implementation: str,
    pid: int,
    expected_start_ticks: str,
    private_output: Path,
    declared_role: str,
    declared_rank: int | None = None,
    timeout: float = 5.0,
    max_output_bytes: int = 1024 * 1024,
) -> dict[str, Any]:
    """Run one Linux attach attempt and return a closed experimental envelope."""

    if implementation not in IMPLEMENTATIONS:
        raise ValueError("unknown stack producer implementation")
    if pid <= 0 or timeout <= 0 or max_output_bytes <= 0:
        raise ValueError("pid, timeout, and output budget must be positive")

    start_ns = time.monotonic_ns()
    producer_timeout_ns = int(timeout * 1_000_000_000)
    cleanup_timeout = 1.0
    coordinator_timeout_ns = producer_timeout_ns + 2_000_000_000
    if platform.system() != "Linux":
        return _record(
            identity=expected_start_ticks,
            post_identity=None,
            role=declared_role,
            rank=declared_rank,
            start_ns=start_ns,
            end_ns=time.monotonic_ns(),
            producer_timeout_ns=producer_timeout_ns,
            coordinator_timeout_ns=coordinator_timeout_ns,
            max_output_bytes=max_output_bytes,
            implementation=implementation,
            version=None,
            binary_digest=None,
            stage="preflight",
            outcome="unsupported",
            raw_digest=None,
        )

    if linux_start_ticks(pid) != expected_start_ticks:
        raise ValueError("subject identity is not live before capture")

    binary = shutil.which(implementation)
    if binary is None:
        return _record(
            identity=expected_start_ticks,
            post_identity=None,
            role=declared_role,
            rank=declared_rank,
            start_ns=start_ns,
            end_ns=time.monotonic_ns(),
            producer_timeout_ns=producer_timeout_ns,
            coordinator_timeout_ns=coordinator_timeout_ns,
            max_output_bytes=max_output_bytes,
            implementation=implementation,
            version=None,
            binary_digest=None,
            stage="preflight",
            outcome="binary_missing",
            raw_digest=None,
        )

    version = _version(binary)
    binary_digest = _sha256(Path(binary))
    private_output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    private_output.unlink(missing_ok=True)
    outcome = "execution_failed"
    with private_output.open("wb") as stream:
        if os.name != "nt":
            private_output.chmod(0o600)
        try:
            process = subprocess.Popen(
                _command(implementation, binary, pid),
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                preexec_fn=_limit_output(max_output_bytes),
            )
        except OSError:
            process = None
        if process is not None:
            try:
                return_code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=cleanup_timeout)
                except subprocess.TimeoutExpired:
                    outcome = "execution_failed"
                else:
                    outcome = "timeout"
            else:
                output_size = private_output.stat().st_size
                if return_code == 0 and output_size > 0:
                    outcome = "produced"
                elif return_code == 0:
                    outcome = "empty_output"
                else:
                    stream.flush()
                    private_bytes = private_output.read_bytes().lower()
                    if any(marker in private_bytes for marker in _PERMISSION_MARKERS):
                        outcome = "permission_denied"

    end_ns = time.monotonic_ns()
    post_identity = linux_start_ticks(pid)
    raw_digest = _sha256(private_output) if private_output.stat().st_size else None
    return _record(
        identity=expected_start_ticks,
        post_identity=post_identity,
        role=declared_role,
        rank=declared_rank,
        start_ns=start_ns,
        end_ns=end_ns,
        producer_timeout_ns=producer_timeout_ns,
        coordinator_timeout_ns=coordinator_timeout_ns,
        max_output_bytes=max_output_bytes,
        implementation=implementation,
        version=version,
        binary_digest=binary_digest,
        stage="execution",
        outcome=outcome,
        raw_digest=raw_digest,
    )
