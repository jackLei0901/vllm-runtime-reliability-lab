"""Run Gate 1b with per-rank outcomes and bounded incident capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ORGANIC = ROOT / "experiments" / "organic-hang"
sys.path.insert(0, str(ORGANIC))

from normalize_flight_recorder import normalize_rank_artifacts  # noqa: E402
from process_lifecycle import (  # noqa: E402
    cleanup_process_group,
    identity_is_live,
    start_campaign_process,
    wait_for_rank_identities,
)

MARKERS = {
    "reduce": re.compile(r"DFX_RANK_REDUCE=(\{[^\r\n]+\})"),
    "reduce_return": re.compile(r"DFX_RANK_REDUCE_RETURN=(\{[^\r\n]+\})"),
    "outcome": re.compile(r"DFX_RANK_OUTCOME=(\{[^\r\n]+\})"),
}
WATCHDOG_MARKERS = (
    "Watchdog caught collective operation timeout",
    "watchdog caught collective operation timeout",
    "NCCL watchdog",
)
EXPECTED = {
    "control": "completed_symmetric_fp32",
    "affected": "rank1_assertion_rank0_wait",
}
WORLD_SIZE = 2


def parse_records(output: str, kind: str) -> list[dict]:
    records = [json.loads(value) for value in MARKERS[kind].findall(output)]
    records.sort(key=lambda item: (item["rank"], item.get("call_index", 0)))
    return records


def dtype_set(record: dict) -> tuple[str, ...]:
    return tuple(sorted(set(record["grad_dtypes"])))


def exact_rank_records(records: list[dict]) -> dict[int, dict] | None:
    grouped = {rank: [] for rank in range(WORLD_SIZE)}
    for record in records:
        rank = record.get("rank")
        if rank not in grouped:
            return None
        grouped[rank].append(record)
    if any(len(values) != 1 for values in grouped.values()):
        return None
    return {rank: values[0] for rank, values in grouped.items()}


def classify(
    arm: str,
    *,
    timed_out: bool,
    reduce_records: list[dict],
    reduce_return_records: list[dict],
    outcomes: list[dict],
) -> str:
    by_rank = exact_rank_records(reduce_records)
    if by_rank is None:
        return "unexpected_pattern"
    if any(
        item.get("call_index") != 1
        or item.get("grad_count") != len(item.get("grad_dtypes", []))
        for item in by_rank.values()
    ):
        return "unexpected_pattern"
    returns_by_rank = {item["rank"] for item in reduce_return_records}
    if len(returns_by_rank) != len(reduce_return_records):
        return "unexpected_pattern"
    outcome_by_rank = {item["rank"]: item["outcome"] for item in outcomes}
    if len(outcome_by_rank) != len(outcomes):
        return "unexpected_pattern"
    if arm == "control":
        if (
            not timed_out
            and all(dtype_set(by_rank[rank]) == ("torch.float32",) for rank in (0, 1))
            and returns_by_rank == {0, 1}
            and outcome_by_rank == {0: "completed", 1: "completed"}
        ):
            return "completed_symmetric_fp32"
    elif (
        timed_out
        and dtype_set(by_rank[0]) == ("torch.float32",)
        and dtype_set(by_rank[1]) == ("torch.bfloat16", "torch.float32")
        and not returns_by_rank
        and outcome_by_rank == {1: "local_uniformity_assertion"}
    ):
        return "rank1_assertion_rank0_wait"
    return "unexpected_pattern"


def capture_stack(py_spy: str, rank: int, pid: int, target: Path) -> dict:
    started = time.monotonic()
    try:
        result = subprocess.run(
            [py_spy, "dump", "--pid", str(pid), "--native", "--json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except subprocess.TimeoutExpired:
        return {"rank": rank, "status": "timeout", "elapsed_seconds": 8}
    raw = result.stdout or result.stderr
    summary = {
        "rank": rank,
        "status": "captured" if result.returncode == 0 else "failed",
        "return_code": result.returncode,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "raw_persisted": False,
    }
    try:
        payload = json.loads(result.stdout)
        threads = payload if isinstance(payload, list) else payload.get("threads", [])
        frames = []
        for thread in threads:
            for frame in thread.get("frames", []):
                filename = frame.get("filename")
                if filename and Path(filename).resolve() == target.resolve():
                    frames.append(
                        {
                            "function": str(frame.get("name", "unknown"))[:128],
                            "line": int(frame.get("line", 0)),
                        }
                    )
                    break
        summary["thread_count"] = len(threads)
        summary["project_frames"] = frames[:8]
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
        summary["status"] = "unparsed" if result.returncode == 0 else "failed"
    return summary


def summarize_flight_recorder(state_dir: Path) -> dict:
    artifacts = []
    files = []
    for rank in (0, 1):
        candidates = [
            state_dir / f"trace_{rank}",
            state_dir / f"trace_{rank}.pickle",
        ]
        path = next((item for item in candidates if item.is_file()), None)
        if path is None:
            continue
        raw = path.read_bytes()
        entry = {
            "rank": rank,
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        try:
            decoded = pickle.loads(raw)
            if not isinstance(decoded, dict):
                raise TypeError("Flight Recorder root is not a dictionary")
            decoded["rank"] = rank
            artifacts.append(decoded)
            entry["decoded"] = True
        except Exception as exc:
            entry["decoded"] = False
            entry["decode_error_type"] = type(exc).__name__
        files.append(entry)
    normalized = None
    if artifacts:
        try:
            normalized = normalize_rank_artifacts(artifacts)
        except Exception as exc:
            normalized = {
                "normalization": "failed_closed",
                "error_type": type(exc).__name__,
            }
    return {
        "file_count": len(files),
        "files": files,
        "normalized": normalized,
        "raw_persisted": False,
    }


def run_trial(
    arm: str,
    trial: int,
    *,
    wall_timeout: int,
    stack_capture_after: int,
    py_spy: str,
) -> dict:
    import torch

    started = time.monotonic()
    reproducer = HERE / "gate1b_reproducer.py"
    with tempfile.TemporaryDirectory(prefix="dfx-gate1b-") as temporary:
        state_dir = Path(temporary) / "state"
        state_dir.mkdir()
        raw_path = Path(temporary) / "private-output.log"
        environment = dict(os.environ)
        environment.update(
            {
                "DFX_STATE_DIR": str(state_dir),
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
        cleanup = None
        runner_error = None
        deadline = started + wall_timeout
        with raw_path.open("wb") as raw_stream:
            try:
                process, parent_identity = start_campaign_process(
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
                ranks = wait_for_rank_identities(
                    state_dir,
                    world_size=WORLD_SIZE,
                    deadline_monotonic=min(deadline, time.monotonic() + 20),
                    parent_poll=process.poll,
                )
                while process.poll() is None and time.monotonic() < deadline:
                    elapsed = time.monotonic() - started
                    if (
                        arm == "affected"
                        and not captured
                        and elapsed >= stack_capture_after
                    ):
                        stacks = [
                            capture_stack(py_spy, rank, identity.pid, reproducer)
                            for rank, identity in enumerate(ranks)
                            if identity_is_live(identity)
                        ]
                        captured = True
                    time.sleep(0.05)
                timed_out = process.poll() is None and time.monotonic() >= deadline
            except BaseException as exc:
                runner_error = type(exc).__name__
            finally:
                if process is not None and parent_identity is not None:
                    if process.poll() is None or any(
                        identity_is_live(item) for item in ranks
                    ):
                        try:
                            cleanup = cleanup_process_group(
                                process, parent_identity, ranks
                            )
                        except BaseException as exc:
                            runner_error = runner_error or type(exc).__name__
        output_bytes = raw_path.read_bytes()
        output = output_bytes.decode("utf-8", errors="replace")
        reduce_records = parse_records(output, "reduce")
        outcomes = parse_records(output, "outcome")
        reduce_return_records = parse_records(output, "reduce_return")
        classification = classify(
            arm,
            timed_out=timed_out,
            reduce_records=reduce_records,
            reduce_return_records=reduce_return_records,
            outcomes=outcomes,
        )
        return_code = None if process is None else process.returncode
        if cleanup is not None:
            no_tracked_orphans = cleanup["no_orphans"]
        elif process is None:
            no_tracked_orphans = False
        else:
            no_tracked_orphans = process.poll() is not None and not any(
                identity_is_live(item) for item in ranks
            )
        return {
            "schema_version": 1,
            "arm": arm,
            "trial": trial,
            "expected_classification": EXPECTED[arm],
            "classification": classification,
            "timed_out": timed_out,
            "return_code": 124 if timed_out else return_code,
            "reduce_records": reduce_records,
            "reduce_return_records": reduce_return_records,
            "rank_outcomes": outcomes,
            "watchdog_marker_seen": any(value in output for value in WATCHDOG_MARKERS),
            "stack_capture": stacks,
            "flight_recorder": summarize_flight_recorder(state_dir),
            "duration_seconds": round(time.monotonic() - started, 3),
            "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
            "raw_output_persisted": False,
            "runner_error_type": runner_error,
            "reproducer_sha256": hashlib.sha256(reproducer.read_bytes()).hexdigest(),
            "torch_version": torch.__version__,
            "torch_git_version": getattr(torch.version, "git_version", None),
            "cuda_runtime_version": torch.version.cuda,
            "cuda_visible_devices": torch.cuda.device_count(),
            "gpu_names": [torch.cuda.get_device_name(index) for index in range(2)],
            "rank_identities_recorded": len(ranks) == WORLD_SIZE,
            "no_tracked_orphans": no_tracked_orphans,
        }


def preflight(py_spy: str) -> None:
    import torch
    from torch.distributed.fsdp import FSDPModule

    if os.name != "posix" or not Path("/proc").is_dir():
        raise RuntimeError("Gate 1b requires Linux /proc")
    if torch.__version__ != "2.13.0+cu130":
        raise RuntimeError(f"expected torch 2.13.0+cu130, got {torch.__version__}")
    if torch.cuda.device_count() != 2:
        raise RuntimeError("Gate 1b requires exactly two visible GPUs")
    names = [torch.cuda.get_device_name(index) for index in range(2)]
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
        raise RuntimeError("py-spy is required for Gate 1b")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--wall-timeout", type=int, default=45)
    parser.add_argument("--stack-capture-after", type=int, default=20)
    parser.add_argument("--py-spy", default="py-spy")
    args = parser.parse_args()
    if not 0 < args.stack_capture_after < args.wall_timeout:
        parser.error("stack capture must occur before the wall timeout")
    if args.stack_capture_after >= 30:
        parser.error("stack capture must precede the 30-second process-group timeout")
    preflight(args.py_spy)
    args.output.mkdir(parents=True, exist_ok=False)
    for arm in ("control", "affected"):
        for trial in range(1, args.trials + 1):
            result = run_trial(
                arm,
                trial,
                wall_timeout=args.wall_timeout,
                stack_capture_after=args.stack_capture_after,
                py_spy=args.py_spy,
            )
            path = args.output / f"{arm}-trial-{trial}.json"
            path.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"{arm} trial {trial}: {result['classification']}")
            if result["classification"] != EXPECTED[arm]:
                print("STOP: first expectation mismatch retained", file=sys.stderr)
                return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
