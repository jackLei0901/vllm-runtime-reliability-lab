"""Execute the frozen four-GPU topology/divergence matrix."""

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
sys.path.insert(0, str(ROOT / "experiments" / "organic-hang"))

from prepare_four_gpu_matrix import MATRIX  # noqa: E402
from process_lifecycle import (  # noqa: E402
    cleanup_process_group,
    identity_is_live,
    start_campaign_process,
    wait_for_rank_identities,
)

ASSERTION_MARKER = "FSDP reduce-scatter expects uniform gradient dtype"


def classify(return_code: int, output: str) -> str:
    if return_code == 124:
        return "timeout"
    if return_code != 0 and ASSERTION_MARKER in output:
        return "mixed_gradient_dtype_assertion"
    if return_code == 0:
        return "completed"
    return "unexpected_failure"


def run_trial(
    case: str,
    prepared: Path,
    trial: int,
    timeout_seconds: int,
    port: int,
) -> dict:
    import torch

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="dfx-four-gpu-") as temporary:
        state_dir = Path(temporary) / "state"
        environment = dict(os.environ)
        environment.update(
            {
                "DFX_STATE_DIR": str(state_dir),
                "DFX_ORGANIC_MASTER_PORT": str(port),
                "DFX_ORGANIC_TIMEOUT_S": "60",
                "DFX_ORGANIC_TRAINING_STEPS": "200",
                "DFX_ORGANIC_DEBUG_DETAIL": "0",
            }
        )
        process, parent_identity = start_campaign_process(
            [sys.executable, str(prepared)], environment
        )
        rank_identities = []
        timed_out = False
        cleanup = None
        output_bytes = b""
        try:
            rank_identities = wait_for_rank_identities(
                state_dir,
                world_size=4,
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
        return {
            "schema_version": 1,
            "case": case,
            "trial": trial,
            "expected_classification": MATRIX[case]["expected"],
            "classification": classify(return_code, output),
            "return_code": return_code,
            "assertion_marker_seen": ASSERTION_MARKER in output,
            "duration_seconds": round(time.monotonic() - started, 3),
            "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
            "raw_output_persisted": False,
            "prepared_sha256": hashlib.sha256(prepared.read_bytes()).hexdigest(),
            "torch_version": torch.__version__,
            "torch_git_version": getattr(torch.version, "git_version", None),
            "cuda_runtime_version": torch.version.cuda,
            "cuda_visible_devices": torch.cuda.device_count(),
            "gpu_names": [torch.cuda.get_device_name(index) for index in range(4)],
            "rank_identities_recorded": len(rank_identities) == 4,
            "no_tracked_orphans": (
                cleanup["no_orphans"] if cleanup is not None else True
            ),
            "rendezvous_port": port,
            "training_steps": 200,
            "detail_enabled": False,
        }


def preflight(prepared_dir: Path) -> None:
    import torch
    from torch.distributed.fsdp import FSDPModule

    if os.name != "posix" or not Path("/proc").is_dir():
        raise RuntimeError("campaign requires Linux /proc")
    if torch.cuda.device_count() != 4:
        raise RuntimeError("exactly four visible CUDA devices are required")
    names = [torch.cuda.get_device_name(index) for index in range(4)]
    if len(set(names)) != 1:
        raise RuntimeError(f"four identical GPUs are required: {names}")
    if not torch.distributed.is_nccl_available():
        raise RuntimeError("PyTorch NCCL support is required")
    if not hasattr(FSDPModule, "set_reduce_scatter_unused_params"):
        raise RuntimeError("installed PyTorch lacks the required FSDP2 API")
    for case, contract in MATRIX.items():
        path = prepared_dir / f"{case}.py"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != contract["sha256"]:
            raise RuntimeError(f"prepared source mismatch for {case}: {digest}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--base-port", type=int, default=29600)
    args = parser.parse_args()
    if args.trials < 1 or args.timeout_seconds < 1:
        parser.error("trials and timeout must be positive")

    preflight(args.prepared_dir)
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        ordinal = 0
        for case in MATRIX:
            for trial in range(1, args.trials + 1):
                port = args.base_port + ordinal
                ordinal += 1
                result = run_trial(
                    case,
                    args.prepared_dir / f"{case}.py",
                    trial,
                    args.timeout_seconds,
                    port,
                )
                target = args.output / f"{case}-trial-{trial}.json"
                target.write_text(
                    json.dumps(result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                print(f"{case} trial {trial}: {result['classification']}")
    except BaseException:
        shutil.rmtree(args.output, ignore_errors=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
