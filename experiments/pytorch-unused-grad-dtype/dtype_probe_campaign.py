"""Run the fail-closed two-rank dtype mechanism probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "organic-hang"))

from process_lifecycle import (  # noqa: E402
    cleanup_process_group,
    identity_is_live,
    start_campaign_process,
    wait_for_rank_identities,
)

ASSERTION_MARKER = "FSDP reduce-scatter expects uniform gradient dtype"
PROBE_PATTERN = re.compile(r"DFX_REDUCE_DTYPES=(\{[^\r\n]+\})")
CASES = {
    "unused-parameter": "completed_uniform_bf16",
    "forced-mixed-gradient": "mixed_gradient_dtype_assertion",
}
UNIFORM_BF16 = ("torch.bfloat16",)
MIXED_BF16_FP32 = ("torch.bfloat16", "torch.float32")


def parse_dtype_records(output: str) -> list[dict]:
    records = [json.loads(match) for match in PROBE_PATTERN.findall(output)]
    records.sort(key=lambda item: item["rank"])
    return records


def _per_rank_dtype_sets(records: list[dict]) -> list[tuple[str, ...]]:
    return [tuple(sorted(set(item["grad_dtypes"]))) for item in records]


def classify(case: str, return_code: int, output: str) -> str:
    if return_code == 124:
        return "timeout"
    records = parse_dtype_records(output)
    complete_ranks = len(records) == 2 and [item["rank"] for item in records] == [0, 1]
    dtype_sets = _per_rank_dtype_sets(records)
    if (
        case == "forced-mixed-gradient"
        and return_code != 0
        and ASSERTION_MARKER in output
        and complete_ranks
        and dtype_sets == [MIXED_BF16_FP32, MIXED_BF16_FP32]
    ):
        return "mixed_gradient_dtype_assertion"
    if return_code == 0 and complete_ranks:
        if case == "unused-parameter" and dtype_sets == [UNIFORM_BF16, UNIFORM_BF16]:
            return "completed_uniform_bf16"
        return "completed_unexpected_dtype_set"
    return "unexpected_failure" if return_code else "unverified_completion"


def run_trial(case: str, trial: int, timeout_seconds: int) -> dict:
    import torch

    started = time.monotonic()
    reproducer = HERE / "dtype_probe_reproducer.py"
    with tempfile.TemporaryDirectory(prefix="dfx-dtype-probe-") as temporary:
        state_dir = Path(temporary) / "state"
        environment = dict(os.environ)
        environment["DFX_STATE_DIR"] = str(state_dir)
        process, parent_identity = start_campaign_process(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=2",
                str(reproducer),
                "--case",
                case,
            ],
            environment,
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
        return {
            "schema_version": 1,
            "case": case,
            "trial": trial,
            "expected_classification": CASES[case],
            "classification": classify(case, return_code, output),
            "dtype_records": parse_dtype_records(output),
            "return_code": return_code,
            "assertion_marker_seen": ASSERTION_MARKER in output,
            "duration_seconds": round(time.monotonic() - started, 3),
            "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
            "raw_output_persisted": False,
            "reproducer_sha256": hashlib.sha256(reproducer.read_bytes()).hexdigest(),
            "torch_version": torch.__version__,
            "torch_git_version": getattr(torch.version, "git_version", None),
            "cuda_runtime_version": torch.version.cuda,
            "cuda_visible_devices": torch.cuda.device_count(),
            "gpu_names": [torch.cuda.get_device_name(index) for index in range(2)],
            "rank_identities_recorded": len(rank_identities) == 2,
            "no_tracked_orphans": (
                cleanup["no_orphans"] if cleanup is not None else True
            ),
        }


def preflight() -> None:
    import torch
    from torch.distributed.fsdp import FSDPModule

    if os.name != "posix" or not Path("/proc").is_dir():
        raise RuntimeError("dtype probe requires Linux /proc")
    if torch.cuda.device_count() != 2:
        raise RuntimeError("dtype probe requires exactly two visible GPUs")
    if not torch.distributed.is_nccl_available():
        raise RuntimeError("PyTorch NCCL support is required")
    if not hasattr(FSDPModule, "set_reduce_scatter_unused_params"):
        raise RuntimeError("installed PyTorch lacks the required FSDP2 API")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    args = parser.parse_args()
    preflight()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        for case in CASES:
            for trial in range(1, args.trials + 1):
                result = run_trial(case, trial, args.timeout_seconds)
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
