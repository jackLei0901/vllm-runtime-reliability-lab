"""Run and classify the two-rank FSDP2 experiment without retaining raw logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ORGANIC_DIR = ROOT / "experiments" / "organic-hang"
sys.path.insert(0, str(ORGANIC_DIR))

from process_lifecycle import (  # noqa: E402
    cleanup_process_group,
    identity_is_live,
    start_campaign_process,
    wait_for_rank_identities,
)

ASSERTION_MARKER = "FSDP reduce-scatter expects uniform gradient dtype"
SUCCESS_MARKER = "DFX_RESULT="


def _success_payload(output: str) -> dict | None:
    matches = [
        line.removeprefix(SUCCESS_MARKER)
        for line in output.splitlines()
        if line.startswith(SUCCESS_MARKER)
    ]
    if len(matches) != 1:
        return None
    try:
        payload = json.loads(matches[0])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def classify_output(arm: str, return_code: int, output: str) -> str:
    if return_code == 124:
        return "timeout"
    has_assertion = ASSERTION_MARKER in output
    if return_code != 0 and has_assertion:
        return "mixed_gradient_dtype_assertion"
    payload = _success_payload(output)
    if return_code == 0 and payload is not None and payload.get("arm") == arm:
        return "completed"
    return "unexpected_failure" if return_code else "unverified_completion"


def reproducer_path(execution_mode: str) -> Path:
    filename = (
        "reproducer.py" if execution_mode == "single" else "accumulation_reproducer.py"
    )
    return HERE / filename


def build_command(arm: str, work_units: int, execution_mode: str) -> list[str]:
    count_flag = "--steps" if execution_mode == "single" else "--microbatches"
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        str(reproducer_path(execution_mode)),
        "--arm",
        arm,
        count_flag,
        str(work_units),
    ]


def run_trial(
    arm: str,
    trial: int,
    timeout_seconds: int,
    work_units: int,
    execution_mode: str,
) -> dict:
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="dfx-unused-grad-") as temporary:
        state_dir = Path(temporary) / "state"
        environment = dict(os.environ)
        environment["DFX_STATE_DIR"] = str(state_dir)
        process, parent_identity = start_campaign_process(
            build_command(arm, work_units, execution_mode), environment
        )
        rank_identities = []
        timed_out = False
        cleanup = None
        output_bytes = b""
        try:
            rank_identities = wait_for_rank_identities(
                state_dir,
                world_size=2,
                deadline_monotonic=time.monotonic() + min(30, timeout_seconds),
                parent_poll=process.poll,
            )
            try:
                stdout, stderr = process.communicate(timeout=timeout_seconds)
                output_bytes = (stdout or b"") + (stderr or b"")
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                output_bytes = (exc.stdout or b"") + (exc.stderr or b"")
        finally:
            tracked_live = identity_is_live(parent_identity) or any(
                identity_is_live(item) for item in rank_identities
            )
            if timed_out or tracked_live:
                cleanup = cleanup_process_group(
                    process, parent_identity, rank_identities
                )

        return_code = 124 if timed_out else int(process.returncode or 0)
        output = output_bytes.decode("utf-8", errors="replace")
        no_orphans = cleanup["no_orphans"] if cleanup else True
        return {
            "schema_version": 1,
            "arm": arm,
            "trial": trial,
            "torch_version": _torch_version(),
            "torch_git_version": _torch_git_version(),
            "cuda_runtime_version": _cuda_runtime_version(),
            "cuda_visible_devices": _visible_device_count(),
            "gpu_names": _gpu_names(),
            "reproducer_sha256": _sha256_file(reproducer_path(execution_mode)),
            "execution_mode": execution_mode,
            "work_units_requested": work_units,
            "return_code": return_code,
            "classification": classify_output(arm, return_code, output),
            "assertion_marker_seen": ASSERTION_MARKER in output,
            "success_marker_seen": _success_payload(output) is not None,
            "duration_seconds": round(time.monotonic() - started, 3),
            "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
            "raw_output_persisted": False,
            "rank_identities_recorded": len(rank_identities) == 2,
            "no_tracked_orphans": no_orphans,
        }


def _torch_version() -> str:
    import torch

    return torch.__version__


def _torch_git_version() -> str | None:
    import torch

    return getattr(torch.version, "git_version", None)


def _cuda_runtime_version() -> str | None:
    import torch

    return torch.version.cuda


def _visible_device_count() -> int:
    import torch

    return torch.cuda.device_count()


def _gpu_names() -> list[str]:
    import torch

    return [torch.cuda.get_device_name(index) for index in range(2)]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight() -> None:
    import torch
    from torch.distributed.fsdp import FSDPModule

    if os.name != "posix" or not Path("/proc").is_dir():
        raise RuntimeError("campaign requires Linux /proc for PID-safe cleanup")
    if torch.cuda.device_count() != 2:
        raise RuntimeError("campaign requires exactly two visible CUDA devices")
    if not torch.distributed.is_nccl_available():
        raise RuntimeError("PyTorch NCCL support is required")
    if not hasattr(FSDPModule, "set_reduce_scatter_unused_params"):
        raise RuntimeError("installed PyTorch lacks the required FSDP2 API")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--work-units", type=int, default=4)
    parser.add_argument(
        "--execution-mode", choices=("single", "accumulated"), default="single"
    )
    args = parser.parse_args()
    if args.trials < 1 or args.timeout_seconds < 1 or args.work_units < 1:
        parser.error("trials, timeout and work units must be positive")
    if args.execution_mode == "accumulated" and args.work_units < 2:
        parser.error("accumulated mode requires at least two microbatches")

    preflight()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        for arm in ("control", "affected"):
            for trial in range(1, args.trials + 1):
                result = run_trial(
                    arm,
                    trial,
                    args.timeout_seconds,
                    args.work_units,
                    args.execution_mode,
                )
                target = args.output / f"{arm}-trial-{trial}.json"
                target.write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                print(f"{arm} trial {trial}: {result['classification']}")
    except BaseException:
        shutil.rmtree(args.output, ignore_errors=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
