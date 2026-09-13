"""Run Gate 1d without stopping on a termination-prediction mismatch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import gate1c_campaign as gate1c

HERE = Path(__file__).resolve().parent
MARKERS = {
    **gate1c.MARKERS,
    "ptrace": re.compile(r"DFX_RANK_PTRACE=(\{[^\r\n]+\})"),
}
EXPECTED_MECHANISM = gate1c.EXPECTED_MECHANISM
EXPECTED_TERMINATION = gate1c.EXPECTED_TERMINATION
WATCHDOG_MARKERS = gate1c.WATCHDOG_MARKERS
WORLD_SIZE = gate1c.WORLD_SIZE
YAMA_SCOPE = Path("/proc/sys/kernel/yama/ptrace_scope")


def parse_records(output: str, kind: str) -> list[dict]:
    records = [json.loads(value) for value in MARKERS[kind].findall(output)]
    records.sort(key=lambda item: (item["rank"], item.get("call_index", 0)))
    return records


def preflight(py_spy: str) -> dict:
    import torch
    from torch.distributed.fsdp import FSDPModule

    if os.name != "posix" or not Path("/proc").is_dir():
        raise RuntimeError("Gate 1d requires Linux /proc")
    if torch.__version__ != "2.13.0+cu130":
        raise RuntimeError(f"expected torch 2.13.0+cu130, got {torch.__version__}")
    if torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError("Gate 1d requires exactly two visible GPUs")
    names = [torch.cuda.get_device_name(index) for index in range(WORLD_SIZE)]
    if len(set(names)) != 1:
        raise RuntimeError(f"two identical GPUs are required: {names}")
    if not torch.distributed.is_nccl_available():
        raise RuntimeError("NCCL support is required")
    if not hasattr(FSDPModule, "set_reduce_scatter_unused_params"):
        raise RuntimeError("required FSDP2 API is unavailable")
    result = subprocess.run(
        [py_spy, "--version"], check=False, capture_output=True, text=True, timeout=10
    )
    if result.returncode != 0:
        raise RuntimeError("py-spy is required for Gate 1d")
    if YAMA_SCOPE.exists():
        ptrace_scope = int(YAMA_SCOPE.read_text(encoding="utf-8").strip())
        ptrace_mode = "pr_set_ptracer_parent"
    else:
        ptrace_scope = None
        ptrace_mode = "yama_absent"
    return {
        "effective_uid": os.geteuid(),
        "ptrace_scope": ptrace_scope,
        "ptrace_mode": ptrace_mode,
        "py_spy_version_sha256": hashlib.sha256(
            (result.stdout or result.stderr).strip().encode()
        ).hexdigest(),
    }


def run_trial(
    arm: str,
    trial: int,
    *,
    wall_timeout: int,
    stack_capture_after: int,
    py_spy: str,
    preflight_record: dict,
) -> dict:
    import torch

    started = time.monotonic()
    reproducer = HERE / "gate1d_reproducer.py"
    with tempfile.TemporaryDirectory(prefix="dfx-gate1d-") as temporary:
        state_dir = Path(temporary) / "state"
        state_dir.mkdir()
        raw_path = Path(temporary) / "private-output.log"
        environment = dict(os.environ)
        environment.update(
            {
                "DFX_STATE_DIR": str(state_dir),
                "DFX_OBSERVER_PID": str(os.getpid()),
                "TORCH_NCCL_TRACE_BUFFER_SIZE": "2000",
                "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
                "TORCH_NCCL_ASYNC_ERROR_HANDLING": "3",
                "TORCH_FR_DUMP_TEMP_FILE": str(state_dir / "trace_"),
                "TORCH_NCCL_DEBUG_INFO_TEMP_FILE": str(state_dir / "trace_"),
            }
        )
        process = None
        parent_identity = None
        ranks = []
        stacks = []
        captured = False
        timed_out = False
        natural_return_code = None
        cleanup = None
        runner_error = None
        deadline = started + wall_timeout
        with raw_path.open("wb") as raw_stream:
            try:
                process, parent_identity = gate1c.start_campaign_process(
                    [
                        sys.executable,
                        "-m",
                        "torch.distributed.run",
                        "--standalone",
                        "--nproc-per-node=2",
                        str(reproducer),
                        "--arm",
                        arm,
                        "--microbatches",
                        "4",
                    ],
                    environment,
                    stdout=raw_stream,
                    stderr=subprocess.STDOUT,
                )
                ranks = gate1c.wait_for_rank_identities(
                    state_dir,
                    world_size=WORLD_SIZE,
                    deadline_monotonic=min(deadline, time.monotonic() + 20),
                    parent_poll=process.poll,
                )
                while process.poll() is None and time.monotonic() < deadline:
                    elapsed_seconds = time.monotonic() - started
                    if (
                        arm == "affected"
                        and not captured
                        and elapsed_seconds >= stack_capture_after
                    ):
                        stacks = gate1c.capture_rank_stacks(py_spy, ranks, reproducer)
                        captured = True
                    time.sleep(0.05)
                natural_return_code = process.poll()
                timed_out = natural_return_code is None and time.monotonic() >= deadline
            except BaseException as exc:
                runner_error = type(exc).__name__
            finally:
                if process is not None and parent_identity is not None:
                    if process.poll() is None or any(
                        gate1c.identity_is_live(item) for item in ranks
                    ):
                        try:
                            cleanup = gate1c.cleanup_process_group(
                                process, parent_identity, ranks
                            )
                        except BaseException as exc:
                            runner_error = runner_error or type(exc).__name__
        output_bytes = raw_path.read_bytes()
        output = output_bytes.decode("utf-8", errors="replace")
        records = {kind: parse_records(output, kind) for kind in MARKERS}
        watchdog_seen = any(value in output for value in WATCHDOG_MARKERS)
        mechanism = gate1c.classify_mechanism(
            arm,
            reduce_records=records["reduce"],
            reduce_return_records=records["reduce_return"],
            barrier_enter_records=records["barrier_enter"],
            barrier_return_records=records["barrier_return"],
            outcomes=records["outcome"],
        )
        termination = gate1c.classify_termination(
            arm,
            timed_out=timed_out,
            natural_return_code=natural_return_code,
            teardown_enter_records=records["teardown_enter"],
            teardown_return_records=records["teardown_return"],
            watchdog_marker_seen=watchdog_seen,
        )
        if cleanup is not None:
            no_tracked_orphans = cleanup["no_orphans"]
        elif process is None:
            no_tracked_orphans = False
        else:
            no_tracked_orphans = process.poll() is not None and not any(
                gate1c.identity_is_live(item) for item in ranks
            )
        return {
            "schema_version": 3,
            "arm": arm,
            "trial": trial,
            "configuration": {
                "stack_capture_after_seconds": stack_capture_after,
                "process_group_timeout_seconds": 30,
                "wall_timeout_seconds": wall_timeout,
                "async_error_handling": 3,
            },
            "expected_mechanism": EXPECTED_MECHANISM[arm],
            "mechanism_classification": mechanism,
            "expected_termination": EXPECTED_TERMINATION[arm],
            "termination_classification": termination,
            "termination_prediction_matched": termination == EXPECTED_TERMINATION[arm],
            "timed_out": timed_out,
            "natural_return_code": natural_return_code,
            "reported_return_code": 124 if timed_out else natural_return_code,
            "reduce_records": records["reduce"],
            "reduce_return_records": records["reduce_return"],
            "barrier_enter_records": records["barrier_enter"],
            "barrier_return_records": records["barrier_return"],
            "teardown_enter_records": records["teardown_enter"],
            "teardown_return_records": records["teardown_return"],
            "rank_outcomes": records["outcome"],
            "rank_ptrace_records": records["ptrace"],
            "watchdog_marker_seen": watchdog_seen,
            "stack_capture": stacks,
            "flight_recorder": gate1c.summarize_flight_recorder(state_dir),
            "preflight": preflight_record,
            "duration_seconds": round(time.monotonic() - started, 3),
            "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
            "raw_output_persisted": False,
            "runner_error_type": runner_error,
            "reproducer_sha256": hashlib.sha256(reproducer.read_bytes()).hexdigest(),
            "torch_version": torch.__version__,
            "torch_git_version": getattr(torch.version, "git_version", None),
            "cuda_runtime_version": torch.version.cuda,
            "cuda_visible_devices": torch.cuda.device_count(),
            "gpu_names": [
                torch.cuda.get_device_name(index) for index in range(WORLD_SIZE)
            ],
            "rank_identities_recorded": len(ranks) == WORLD_SIZE,
            "no_tracked_orphans": no_tracked_orphans,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--wall-timeout", type=int, default=60)
    parser.add_argument("--stack-capture-after", type=int, default=20)
    parser.add_argument("--py-spy", default="py-spy")
    args = parser.parse_args()
    if not 0 < args.stack_capture_after < args.wall_timeout:
        parser.error("stack capture must occur before the wall timeout")
    if args.stack_capture_after >= 30:
        parser.error("stack capture must precede the 30-second process-group timeout")
    if args.wall_timeout < 60:
        parser.error("Gate 1d requires at least a 60-second wall bound")
    preflight_record = preflight(args.py_spy)
    args.output.mkdir(parents=True, exist_ok=False)
    termination_mismatches = 0
    for arm in ("control", "affected"):
        for trial in range(1, args.trials + 1):
            result = run_trial(
                arm,
                trial,
                wall_timeout=args.wall_timeout,
                stack_capture_after=args.stack_capture_after,
                py_spy=args.py_spy,
                preflight_record=preflight_record,
            )
            path = args.output / f"{arm}-trial-{trial}.json"
            path.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                f"{arm} trial {trial}: mechanism="
                f"{result['mechanism_classification']}; termination="
                f"{result['termination_classification']}"
            )
            if result["mechanism_classification"] != EXPECTED_MECHANISM[arm]:
                print("STOP: first mechanism mismatch retained", file=sys.stderr)
                return 2
            if not result["termination_prediction_matched"]:
                termination_mismatches += 1
                print("NOTE: termination prediction mismatch retained", file=sys.stderr)
    print(f"termination prediction mismatches: {termination_mismatches}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
