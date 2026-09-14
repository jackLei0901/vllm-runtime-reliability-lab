"""Run the frozen FSDP-free Gate 1g ProcessGroupNCCL experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import gate1c_campaign as gate1c
import gate1f_campaign as gate1f

HERE = Path(__file__).resolve().parent
REPRODUCER = HERE / "gate1g_reproducer.py"
WORLD_SIZE = 2
EXPECTED_TORCH = "2.13.0+cu130"
EXPECTED_NCCL = [2, 29, 7]
MARKER_PREFIXES = {
    "warmup_enter": "DFX_G1G_WARMUP_ENTER=",
    "warmup_return": "DFX_G1G_WARMUP_RETURN=",
    "allreduce_enqueued": "DFX_G1G_ALLREDUCE_ENQUEUED=",
    "allreduce_return": "DFX_G1G_ALLREDUCE_RETURN=",
    "ready_observed": "DFX_G1G_READY_OBSERVED=",
    "outcome": "DFX_G1G_OUTCOME=",
    "teardown_enter": "DFX_G1G_TEARDOWN_ENTER=",
    "teardown_return": "DFX_G1G_TEARDOWN_RETURN=",
    "ptrace": "DFX_G1G_PTRACE=",
}
EXPECTED_MECHANISM = {
    "control": "completed_symmetric_all_reduce",
    "affected": "rank1_failure_rank0_all_reduce_wait",
}
EXPECTED_TERMINATION = {
    "control": "normal_completion",
    "affected": "wall_bound_rank1_teardown_wait",
}


def parse_records(output: str, kind: str) -> list[dict]:
    """Decode complete marker objects, including adjacent rank writes."""
    prefix = MARKER_PREFIXES[kind]
    decoder = json.JSONDecoder()
    records = []
    search_from = 0
    while True:
        marker = output.find(prefix, search_from)
        if marker < 0:
            break
        payload_start = marker + len(prefix)
        try:
            payload, payload_end = decoder.raw_decode(output, payload_start)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed {kind} marker at byte {marker}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{kind} marker payload is not an object")
        records.append(payload)
        search_from = payload_end
    records.sort(key=lambda item: (item.get("rank", -1), item.get("call_index", 0)))
    return records


def parse_all_records_safe(output: str) -> tuple[dict[str, list[dict]], dict]:
    records = {}
    errors = {}
    for kind in MARKER_PREFIXES:
        try:
            records[kind] = parse_records(output, kind)
        except Exception as exc:
            records[kind] = []
            errors[kind] = type(exc).__name__
    return records, errors


def _rank_set(records: list[dict]) -> set[int] | None:
    ranks = [item.get("rank") for item in records]
    if any(not isinstance(rank, int) or rank not in (0, 1) for rank in ranks):
        return None
    if len(ranks) != len(set(ranks)):
        return None
    return set(ranks)


def _outcome_map(records: list[dict]) -> dict[int, str] | None:
    ranks = _rank_set(records)
    if ranks is None:
        return None
    return {int(item["rank"]): str(item.get("outcome")) for item in records}


def classify_mechanism(arm: str, records: dict[str, list[dict]]) -> str:
    warmup_entered = _rank_set(records["warmup_enter"])
    warmup_returned = _rank_set(records["warmup_return"])
    enqueued = _rank_set(records["allreduce_enqueued"])
    returned = _rank_set(records["allreduce_return"])
    ready = _rank_set(records["ready_observed"])
    outcomes = _outcome_map(records["outcome"])
    if warmup_entered != {0, 1} or warmup_returned != {0, 1}:
        return "unexpected_pattern"
    if arm == "control":
        if (
            enqueued == {0, 1}
            and returned == {0, 1}
            and ready == set()
            and outcomes == {0: "completed", 1: "completed"}
        ):
            return EXPECTED_MECHANISM[arm]
    elif (
        enqueued == {0}
        and returned == set()
        and ready == {1}
        and outcomes == {1: "injected_rank_failure"}
    ):
        return EXPECTED_MECHANISM[arm]
    return "unexpected_pattern"


def classify_termination(
    arm: str,
    *,
    timed_out: bool,
    natural_return_code: int | None,
    teardown_enter: list[dict],
    teardown_return: list[dict],
    watchdog_marker_seen: bool,
) -> str:
    entered = _rank_set(teardown_enter)
    returned = _rank_set(teardown_return)
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
    if timed_out and entered == {1} and not returned and watchdog_marker_seen:
        return "wall_bound_rank1_teardown_wait"
    if timed_out:
        return "wall_bound_other_teardown_pattern"
    if watchdog_marker_seen:
        return "watchdog_teardown"
    return "torchrun_teardown"


def preflight(py_spy: str) -> dict:
    import torch

    if os.name != "posix" or not Path("/proc").is_dir():
        raise RuntimeError("Gate 1g requires Linux /proc")
    if torch.__version__ != EXPECTED_TORCH:
        raise RuntimeError(f"expected {EXPECTED_TORCH}, got {torch.__version__}")
    if torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError("Gate 1g requires exactly two visible GPUs")
    names = [torch.cuda.get_device_name(index) for index in range(WORLD_SIZE)]
    if len(set(names)) != 1:
        raise RuntimeError(f"two identical GPUs are required: {names}")
    if not torch.distributed.is_nccl_available():
        raise RuntimeError("NCCL support is required")
    result = subprocess.run(
        [py_spy, "--version"], check=False, capture_output=True, text=True, timeout=10
    )
    if result.returncode != 0:
        raise RuntimeError("py-spy is required for Gate 1g")
    scope_path = Path("/proc/sys/kernel/yama/ptrace_scope")
    scope = (
        int(scope_path.read_text(encoding="utf-8").strip())
        if scope_path.exists()
        else None
    )
    mode = "pr_set_ptracer_parent" if scope_path.exists() else "yama_absent"
    nccl = gate1f.nccl_version_record()
    if nccl != EXPECTED_NCCL:
        raise RuntimeError(f"Gate 1g requires NCCL 2.29.7, got {nccl}")
    return {
        "effective_uid": os.geteuid(),
        "ptrace_scope": scope,
        "ptrace_mode": mode,
        "py_spy_version_sha256": hashlib.sha256(
            (result.stdout or result.stderr).strip().encode()
        ).hexdigest(),
        "nccl_version": nccl,
    }


def rank0_pending_allreduce_summary(artifacts: list[dict[str, Any]]) -> dict:
    """Summarize only rank 0's local pending all-reduce; infer no peer state."""
    ranks = sorted(int(item["rank"]) for item in artifacts)
    rank0 = next((item for item in artifacts if int(item["rank"]) == 0), None)
    candidates = []
    if rank0 is not None:
        for entry in rank0.get("entries", []):
            operation = gate1c._operation(entry)
            if "ALLREDUCE" not in operation.replace("_", ""):
                continue
            pg_id = gate1c._process_group_id(entry)
            members = gate1c._members(rank0.get("pg_config", {}), pg_id)
            state = str(entry.get("state", "unknown"))
            if members == (0, 1) and state != "completed":
                candidates.append(
                    {
                        "sequence_number": int(entry["collective_seq_id"]),
                        "group_members": list(members),
                        "observed_rank": 0,
                        "operation": operation,
                        "state": state,
                    }
                )
    if len(candidates) == 1:
        status = "matched"
    elif candidates:
        status = "unexpected_pending_all_reduce_count"
    else:
        status = "no_rank0_pending_all_reduce"
    return {
        "status": status,
        "dump_ranks": ranks,
        "peer_participation_inferred": False,
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
        item = {
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
            item["decoded"] = True
        except Exception as exc:
            item["decoded"] = False
            item["decode_error_type"] = type(exc).__name__
        files.append(item)
    try:
        local_summary = rank0_pending_allreduce_summary(artifacts)
    except Exception as exc:
        local_summary = {"status": "failed_closed", "error_type": type(exc).__name__}
    return {
        "file_count": len(files),
        "files": files,
        "rank0_pending_all_reduce": local_summary,
        "raw_persisted": False,
    }


def run_trial(
    arm: str,
    *,
    wall_timeout: int,
    stack_capture_after: int,
    py_spy: str,
    preflight_record: dict,
) -> dict:
    import torch

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="dfx-gate1g-") as temporary:
        state_dir = Path(temporary) / "state"
        state_dir.mkdir()
        raw_path = Path(temporary) / "private-output.log"
        environment = dict(os.environ)
        environment.update(
            {
                "DFX_STATE_DIR": str(state_dir),
                "DFX_OBSERVER_PID": str(os.getpid()),
                "TORCH_CPP_LOG_LEVEL": "INFO",
                "TORCH_NCCL_TRACE_BUFFER_SIZE": "2000",
                "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
                "TORCH_NCCL_ASYNC_ERROR_HANDLING": "3",
                "TORCH_FR_DUMP_TEMP_FILE": str(state_dir / "trace_"),
                "TORCH_NCCL_DEBUG_INFO_TEMP_FILE": str(state_dir / "trace_"),
            }
        )
        process = parent_identity = None
        ranks: list = []
        stacks: list[dict] = []
        captured = timed_out = False
        natural_return_code = cleanup = runner_error = None
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
                        str(REPRODUCER),
                        "--arm",
                        arm,
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
                    if (
                        arm == "affected"
                        and not captured
                        and time.monotonic() - started >= stack_capture_after
                    ):
                        stacks = gate1c.capture_rank_stacks(py_spy, ranks, REPRODUCER)
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
        records, marker_errors = parse_all_records_safe(output)
        flags, scan_error = gate1f.scan_library_log_flags_safe(output)
        watchdog = any(value in output for value in gate1c.WATCHDOG_MARKERS)
        mechanism = classify_mechanism(arm, records)
        termination = classify_termination(
            arm,
            timed_out=timed_out,
            natural_return_code=natural_return_code,
            teardown_enter=records["teardown_enter"],
            teardown_return=records["teardown_return"],
            watchdog_marker_seen=watchdog,
        )
        no_orphans = (
            cleanup["no_orphans"]
            if cleanup is not None
            else process is not None
            and process.poll() is not None
            and not any(gate1c.identity_is_live(item) for item in ranks)
        )
        return {
            "schema_version": 5,
            "arm": arm,
            "trial": 1,
            "configuration": {
                "stack_capture_after_seconds": stack_capture_after,
                "process_group_timeout_seconds": 30,
                "wall_timeout_seconds": wall_timeout,
                "work_wait_timeout_seconds": 180,
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
            "warmup_enter_records": records["warmup_enter"],
            "warmup_return_records": records["warmup_return"],
            "allreduce_enqueued_records": records["allreduce_enqueued"],
            "allreduce_return_records": records["allreduce_return"],
            "ready_observed_records": records["ready_observed"],
            "rank_outcomes": records["outcome"],
            "teardown_enter_records": records["teardown_enter"],
            "teardown_return_records": records["teardown_return"],
            "rank_ptrace_records": records["ptrace"],
            "marker_parse_errors": marker_errors,
            "watchdog_marker_seen": watchdog,
            "stack_capture": stacks,
            "flight_recorder": summarize_flight_recorder(state_dir),
            "library_log_flags": flags,
            "library_log_scan_error": scan_error,
            "preflight": preflight_record,
            "duration_seconds": round(time.monotonic() - started, 3),
            "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
            "raw_output_persisted": False,
            "runner_error_type": runner_error,
            "reproducer_sha256": hashlib.sha256(REPRODUCER.read_bytes()).hexdigest(),
            "torch_version": torch.__version__,
            "torch_git_version": getattr(torch.version, "git_version", None),
            "cuda_runtime_version": torch.version.cuda,
            "cuda_visible_devices": torch.cuda.device_count(),
            "gpu_names": [
                torch.cuda.get_device_name(index) for index in range(WORLD_SIZE)
            ],
            "rank_identities_recorded": len(ranks) == WORLD_SIZE,
            "no_tracked_orphans": no_orphans,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-timeout", type=int, default=60)
    parser.add_argument("--stack-capture-after", type=int, default=20)
    parser.add_argument("--py-spy", default="py-spy")
    args = parser.parse_args()
    if args.wall_timeout != 60 or args.stack_capture_after != 20:
        parser.error("Gate 1g freezes stack capture at 20 s and wall bound at 60 s")
    preflight_record = preflight(args.py_spy)
    args.output.mkdir(parents=True, exist_ok=False)
    for arm in ("control", "affected"):
        result = run_trial(
            arm,
            wall_timeout=args.wall_timeout,
            stack_capture_after=args.stack_capture_after,
            py_spy=args.py_spy,
            preflight_record=preflight_record,
        )
        path = args.output / f"{arm}-trial-1.json"
        path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"{arm}: mechanism={result['mechanism_classification']}; "
            f"termination={result['termination_classification']}"
        )
        if (
            result["marker_parse_errors"]
            or result["library_log_scan_error"] is not None
        ):
            print("STOP: bounded parser failure retained", file=sys.stderr)
            return 2
        if result["mechanism_classification"] != EXPECTED_MECHANISM[arm]:
            print("STOP: mechanism mismatch retained", file=sys.stderr)
            return 3
        if arm == "control" and not result["termination_prediction_matched"]:
            print("STOP: control termination mismatch retained", file=sys.stderr)
            return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
