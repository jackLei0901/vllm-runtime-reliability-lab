"""Run Gate 1c with independent mechanism, termination and capture gates."""

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
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

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
    "barrier_enter": re.compile(r"DFX_RANK_BARRIER_ENTER=(\{[^\r\n]+\})"),
    "barrier_return": re.compile(r"DFX_RANK_BARRIER_RETURN=(\{[^\r\n]+\})"),
    "teardown_enter": re.compile(r"DFX_RANK_TEARDOWN_ENTER=(\{[^\r\n]+\})"),
    "teardown_return": re.compile(r"DFX_RANK_TEARDOWN_RETURN=(\{[^\r\n]+\})"),
    "outcome": re.compile(r"DFX_RANK_OUTCOME=(\{[^\r\n]+\})"),
}
WATCHDOG_MARKERS = (
    "Watchdog caught collective operation timeout",
    "watchdog caught collective operation timeout",
    "NCCL watchdog",
)
EXPECTED_MECHANISM = {
    "control": "completed_symmetric_fp32",
    "affected": "rank1_assertion_rank0_barrier_wait",
}
EXPECTED_TERMINATION = {
    "control": "normal_completion",
    "affected": "wall_bound_rank1_teardown_wait",
}
WORLD_SIZE = 2


def parse_records(output: str, kind: str) -> list[dict]:
    records = [json.loads(value) for value in MARKERS[kind].findall(output)]
    records.sort(key=lambda item: (item["rank"], item.get("call_index", 0)))
    return records


def dtype_set(record: dict) -> tuple[str, ...]:
    return tuple(sorted(set(record["grad_dtypes"])))


def exact_rank_records(records: list[dict], ranks: set[int]) -> dict[int, dict] | None:
    grouped = {rank: [] for rank in ranks}
    for record in records:
        rank = record.get("rank")
        if rank not in grouped:
            return None
        grouped[rank].append(record)
    if any(len(values) != 1 for values in grouped.values()):
        return None
    return {rank: values[0] for rank, values in grouped.items()}


def _rank_set(records: list[dict]) -> set[int] | None:
    ranks = [item.get("rank") for item in records]
    if any(
        not isinstance(rank, int) or rank not in range(WORLD_SIZE) for rank in ranks
    ):
        return None
    if len(ranks) != len(set(ranks)):
        return None
    return set(ranks)


def classify_mechanism(
    arm: str,
    *,
    reduce_records: list[dict],
    reduce_return_records: list[dict],
    barrier_enter_records: list[dict],
    barrier_return_records: list[dict],
    outcomes: list[dict],
) -> str:
    by_rank = exact_rank_records(reduce_records, {0, 1})
    if by_rank is None:
        return "unexpected_pattern"
    if any(
        item.get("call_index") != 1
        or item.get("grad_count") != len(item.get("grad_dtypes", []))
        for item in by_rank.values()
    ):
        return "unexpected_pattern"
    returned = _rank_set(reduce_return_records)
    barrier_entered = _rank_set(barrier_enter_records)
    barrier_returned = _rank_set(barrier_return_records)
    outcome_by_rank = {item.get("rank"): item.get("outcome") for item in outcomes}
    if (
        returned is None
        or barrier_entered is None
        or barrier_returned is None
        or len(outcome_by_rank) != len(outcomes)
    ):
        return "unexpected_pattern"
    if arm == "control":
        if (
            all(dtype_set(by_rank[rank]) == ("torch.float32",) for rank in (0, 1))
            and returned == {0, 1}
            and barrier_entered == {0, 1}
            and barrier_returned == {0, 1}
            and outcome_by_rank == {0: "completed", 1: "completed"}
        ):
            return "completed_symmetric_fp32"
    elif (
        dtype_set(by_rank[0]) == ("torch.float32",)
        and dtype_set(by_rank[1]) == ("torch.bfloat16", "torch.float32")
        and returned == {0}
        and barrier_entered == {0}
        and not barrier_returned
        and outcome_by_rank == {1: "local_uniformity_assertion"}
    ):
        return "rank1_assertion_rank0_barrier_wait"
    return "unexpected_pattern"


def classify_termination(
    arm: str,
    *,
    timed_out: bool,
    natural_return_code: int | None,
    teardown_enter_records: list[dict],
    teardown_return_records: list[dict],
    watchdog_marker_seen: bool,
) -> str:
    entered = _rank_set(teardown_enter_records)
    returned = _rank_set(teardown_return_records)
    if entered is None or returned is None:
        return "unexpected_termination"
    if arm == "control":
        if (
            not timed_out
            and natural_return_code == 0
            and entered == {0, 1}
            and returned == {0, 1}
        ):
            return "normal_completion"
        return "unexpected_termination"
    if timed_out:
        if entered == {1} and not returned:
            return "wall_bound_rank1_teardown_wait"
        return "wall_bound_other_teardown_pattern"
    if watchdog_marker_seen:
        return "watchdog_teardown"
    return "torchrun_teardown"


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
        return {
            "rank": rank,
            "status": "timeout",
            "elapsed_seconds": 8,
            "raw_persisted": False,
        }
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


def capture_rank_stacks(py_spy: str, ranks: list, target: Path) -> list[dict]:
    """Start both rank captures together so both precede the PG timeout."""
    summaries = []
    live = []
    for rank, identity in enumerate(ranks):
        if identity_is_live(identity):
            live.append((rank, identity.pid))
        else:
            summaries.append(
                {"rank": rank, "status": "not_live", "raw_persisted": False}
            )
    with ThreadPoolExecutor(max_workers=WORLD_SIZE) as executor:
        futures = [
            executor.submit(capture_stack, py_spy, rank, pid, target)
            for rank, pid in live
        ]
        summaries.extend(future.result() for future in futures)
    summaries.sort(key=lambda item: item["rank"])
    return summaries


def _process_group_id(entry: dict[str, Any]) -> str:
    process_group = entry.get("process_group")
    if isinstance(process_group, (list, tuple)) and process_group:
        return str(process_group[0])
    return str(entry["pg_id"])


def _members(pg_config: dict[str, Any], pg_id: str) -> tuple[int, ...]:
    config = pg_config.get(str(pg_id), {})
    value = config.get("ranks", config.get("global_ranks", config.get("members")))
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or not value:
        raise ValueError(f"process group {pg_id} has no rank membership")
    return tuple(sorted(int(rank) for rank in value))


def _operation(entry: dict[str, Any]) -> str:
    return str(entry.get("profiling_name", "unknown")).rsplit(":", 1)[-1].upper()


def strict_pending_reduce_summary(artifacts: list[dict[str, Any]]) -> dict:
    dump_ranks = sorted(int(artifact["rank"]) for artifact in artifacts)
    missing_dump_ranks = sorted(set(range(WORLD_SIZE)) - set(dump_ranks))
    if dump_ranks != list(range(WORLD_SIZE)):
        return {
            "status": "incomplete_dump_set",
            "dump_ranks": dump_ranks,
            "missing_dump_ranks": missing_dump_ranks,
            "candidates": [],
        }
    grouped: dict[tuple[tuple[int, ...], int], list[dict[str, Any]]] = defaultdict(list)
    for artifact in artifacts:
        rank = int(artifact["rank"])
        pg_config = artifact["pg_config"]
        for entry in artifact["entries"]:
            operation = _operation(entry)
            if bool(entry.get("is_p2p", False)) or operation == "COALESCED":
                continue
            pg_id = _process_group_id(entry)
            members = _members(pg_config, pg_id)
            sequence = int(entry["collective_seq_id"])
            grouped[(members, sequence)].append(
                {
                    "rank": rank,
                    "operation": operation,
                    "state": str(entry.get("state", "unknown")),
                }
            )
    candidates = []
    for (members, sequence), participants in sorted(grouped.items()):
        present = sorted({item["rank"] for item in participants})
        if members != (0, 1) or present != [0] or len(participants) != 1:
            continue
        participant = participants[0]
        if (
            "REDUCE_SCATTER" in participant["operation"]
            and participant["state"] != "completed"
        ):
            candidates.append(
                {
                    "sequence_number": sequence,
                    "group_members": list(members),
                    "present_ranks": present,
                    "operation": participant["operation"],
                    "state_by_rank": [
                        {"rank": participant["rank"], "state": participant["state"]}
                    ],
                }
            )
    return {
        "status": "matched" if candidates else "no_pending_single_rank_reduce",
        "dump_ranks": dump_ranks,
        "missing_dump_ranks": missing_dump_ranks,
        "candidates": candidates[:8],
    }


def summarize_flight_recorder(state_dir: Path) -> dict:
    artifacts = []
    files = []
    for rank in range(WORLD_SIZE):
        candidates = [state_dir / f"trace_{rank}", state_dir / f"trace_{rank}.pickle"]
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
    try:
        strict_summary = strict_pending_reduce_summary(artifacts)
    except Exception as exc:
        strict_summary = {"status": "failed_closed", "error_type": type(exc).__name__}
    return {
        "file_count": len(files),
        "files": files,
        "normalized": normalized,
        "strict_pending_reduce": strict_summary,
        "raw_persisted": False,
    }


def _read_ptrace_scope() -> int | None:
    path = Path("/proc/sys/kernel/yama/ptrace_scope")
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, PermissionError, ValueError):
        return None


def preflight(py_spy: str) -> dict:
    import torch
    from torch.distributed.fsdp import FSDPModule

    if os.name != "posix" or not Path("/proc").is_dir():
        raise RuntimeError("Gate 1c requires Linux /proc")
    if torch.__version__ != "2.13.0+cu130":
        raise RuntimeError(f"expected torch 2.13.0+cu130, got {torch.__version__}")
    if torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError("Gate 1c requires exactly two visible GPUs")
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
        raise RuntimeError("py-spy is required for Gate 1c")
    return {
        "effective_uid": os.geteuid(),
        "ptrace_scope": _read_ptrace_scope(),
        "observer_authorization": "PR_SET_PTRACER campaign parent",
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
    reproducer = HERE / "gate1c_reproducer.py"
    with tempfile.TemporaryDirectory(prefix="dfx-gate1c-") as temporary:
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
                    elapsed_seconds = time.monotonic() - started
                    if (
                        arm == "affected"
                        and not captured
                        and elapsed_seconds >= stack_capture_after
                    ):
                        stacks = capture_rank_stacks(py_spy, ranks, reproducer)
                        captured = True
                    time.sleep(0.05)
                natural_return_code = process.poll()
                timed_out = natural_return_code is None and time.monotonic() >= deadline
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
        records = {kind: parse_records(output, kind) for kind in MARKERS}
        watchdog_seen = any(value in output for value in WATCHDOG_MARKERS)
        mechanism = classify_mechanism(
            arm,
            reduce_records=records["reduce"],
            reduce_return_records=records["reduce_return"],
            barrier_enter_records=records["barrier_enter"],
            barrier_return_records=records["barrier_return"],
            outcomes=records["outcome"],
        )
        termination = classify_termination(
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
                identity_is_live(item) for item in ranks
            )
        return {
            "schema_version": 2,
            "arm": arm,
            "trial": trial,
            "expected_mechanism": EXPECTED_MECHANISM[arm],
            "mechanism_classification": mechanism,
            "expected_termination": EXPECTED_TERMINATION[arm],
            "termination_classification": termination,
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
            "watchdog_marker_seen": watchdog_seen,
            "stack_capture": stacks,
            "flight_recorder": summarize_flight_recorder(state_dir),
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
    parser.add_argument("--wall-timeout", type=int, default=45)
    parser.add_argument("--stack-capture-after", type=int, default=20)
    parser.add_argument("--py-spy", default="py-spy")
    args = parser.parse_args()
    if not 0 < args.stack_capture_after < args.wall_timeout:
        parser.error("stack capture must occur before the wall timeout")
    if args.stack_capture_after >= 30:
        parser.error("stack capture must precede the 30-second process-group timeout")
    preflight_record = preflight(args.py_spy)
    args.output.mkdir(parents=True, exist_ok=False)
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
            if (
                result["mechanism_classification"] != EXPECTED_MECHANISM[arm]
                or result["termination_classification"] != EXPECTED_TERMINATION[arm]
            ):
                print("STOP: first expectation mismatch retained", file=sys.stderr)
                return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
