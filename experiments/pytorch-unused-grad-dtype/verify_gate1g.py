"""Verify the frozen FSDP-free Gate 1g result pair."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import gate1f_campaign as gate1f
import gate1g_campaign as gate1g

HERE = Path(__file__).resolve().parent
REPRODUCER = HERE / "gate1g_reproducer.py"


def _call_line(attribute: str) -> int:
    tree = ast.parse(REPRODUCER.read_text(encoding="utf-8"))
    lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attribute
    ]
    if len(lines) != 1:
        raise AssertionError(f"expected one {attribute} call in Gate 1g reproducer")
    return lines[0]


def _rank_set(records: list[dict], label: str) -> set[int]:
    ranks = [item.get("rank") for item in records]
    if any(not isinstance(rank, int) or rank not in (0, 1) for rank in ranks):
        raise AssertionError(f"{label} has an unexpected rank")
    if len(ranks) != len(set(ranks)):
        raise AssertionError(f"{label} has duplicate ranks")
    return set(ranks)


def _outcome_map(records: list[dict]) -> dict[int, str]:
    _rank_set(records, "rank outcomes")
    return {int(item["rank"]): str(item.get("outcome")) for item in records}


def _verify_stack(record: dict) -> None:
    stacks = record.get("stack_capture")
    if not isinstance(stacks, list) or _rank_set(stacks, "stack evidence") != {0, 1}:
        raise AssertionError("both affected rank stacks are required")
    by_rank = {int(item["rank"]): item for item in stacks}
    expected = {
        0: ("run_all_reduce", _call_line("wait")),
        1: ("main", _call_line("destroy_process_group")),
    }
    for rank, (function, line) in expected.items():
        item = by_rank[rank]
        if item.get("status") != "captured" or item.get("raw_persisted") is not False:
            raise AssertionError(f"rank {rank} stack capture failed")
        frames = item.get("project_frames")
        if not isinstance(frames, list) or not any(
            frame.get("function") == function and frame.get("line") == line
            for frame in frames
        ):
            raise AssertionError(f"rank {rank} stack missed the frozen wait site")


def _verify_common(record: dict, arm: str) -> None:
    if (
        record.get("schema_version") != 5
        or record.get("arm") != arm
        or record.get("trial") != 1
    ):
        raise AssertionError("Gate 1g result identity is wrong")
    if record.get("configuration") != {
        "stack_capture_after_seconds": 20,
        "process_group_timeout_seconds": 30,
        "wall_timeout_seconds": 60,
        "work_wait_timeout_seconds": 180,
        "async_error_handling": 3,
    }:
        raise AssertionError("Gate 1g timing or NCCL configuration changed")
    if record.get("torch_version") != gate1g.EXPECTED_TORCH:
        raise AssertionError("unexpected torch version")
    if record.get("cuda_visible_devices") != 2:
        raise AssertionError("Gate 1g requires two visible GPUs")
    names = record.get("gpu_names")
    if not isinstance(names, list) or len(names) != 2 or len(set(names)) != 1:
        raise AssertionError("Gate 1g requires two identical GPUs")
    if record.get("runner_error_type") is not None:
        raise AssertionError("campaign runner reported an internal error")
    if record.get("marker_parse_errors") != {}:
        raise AssertionError("marker parser failed closed")
    if record.get("library_log_scan_error") is not None:
        raise AssertionError("library log scanner failed closed")
    if (
        record.get("rank_identities_recorded") is not True
        or record.get("no_tracked_orphans") is not True
    ):
        raise AssertionError("rank identity or lifecycle contract failed")
    if record.get("raw_output_persisted") is not False:
        raise AssertionError("raw launcher output was retained")
    if (
        record.get("reproducer_sha256")
        != hashlib.sha256(REPRODUCER.read_bytes()).hexdigest()
    ):
        raise AssertionError("result reproducer hash does not match current source")
    preflight = record.get("preflight")
    if (
        not isinstance(preflight, dict)
        or preflight.get("nccl_version") != gate1g.EXPECTED_NCCL
    ):
        raise AssertionError("NCCL provenance is missing")
    if preflight.get("ptrace_mode") not in {"yama_absent", "pr_set_ptracer_parent"}:
        raise AssertionError("ptrace mode is malformed")
    if (
        not isinstance(preflight.get("py_spy_version_sha256"), str)
        or len(preflight["py_spy_version_sha256"]) != 64
    ):
        raise AssertionError("py-spy version evidence is malformed")
    ptrace = record.get("rank_ptrace_records", [])
    if _rank_set(ptrace, "ptrace evidence") != {0, 1} or any(
        item.get("mode") != preflight.get("ptrace_mode") for item in ptrace
    ):
        raise AssertionError("rank ptrace evidence disagrees with preflight")
    output_hash = record.get("output_sha256")
    if not isinstance(output_hash, str) or len(output_hash) != 64:
        raise AssertionError("launcher output hash is malformed")


def _verify_control(record: dict) -> None:
    if (
        record.get("expected_mechanism") != gate1g.EXPECTED_MECHANISM["control"]
        or record.get("mechanism_classification")
        != gate1g.EXPECTED_MECHANISM["control"]
    ):
        raise AssertionError("control mechanism prediction failed")
    if _rank_set(record["warmup_enter_records"], "control warm-up entry") != {
        0,
        1,
    } or _rank_set(record["warmup_return_records"], "control warm-up return") != {0, 1}:
        raise AssertionError("control communicator warm-up did not complete")
    if _rank_set(record["allreduce_enqueued_records"], "control enqueue") != {
        0,
        1,
    } or _rank_set(record["allreduce_return_records"], "control return") != {0, 1}:
        raise AssertionError("control all-reduce did not complete on both ranks")
    if record.get("ready_observed_records") != [] or _outcome_map(
        record["rank_outcomes"]
    ) != {0: "completed", 1: "completed"}:
        raise AssertionError("control marker matrix is wrong")
    if (
        record.get("termination_classification") != "normal_completion"
        or record.get("termination_prediction_matched") is not True
    ):
        raise AssertionError("control did not terminate normally")
    if record.get("timed_out") or record.get("natural_return_code") != 0:
        raise AssertionError("control launcher return is wrong")
    if _rank_set(record["teardown_enter_records"], "control teardown entry") != {
        0,
        1,
    } or _rank_set(record["teardown_return_records"], "control teardown return") != {
        0,
        1,
    }:
        raise AssertionError("control teardown matrix is wrong")
    if (
        record.get("stack_capture") != []
        or record.get("flight_recorder", {}).get("file_count") != 0
    ):
        raise AssertionError("control must not trigger incident capture")
    expected_flags = {
        name: name
        in {
            "shutdown_start",
            "operations_flushed",
            "watchdog_joined_destroying",
            "destroy_complete",
        }
        for name in gate1f.LOG_MESSAGES
    }
    flags = record.get("library_log_flags")
    if flags != {"0": expected_flags, "1": expected_flags}:
        raise AssertionError("control ProcessGroupNCCL lifecycle flags are wrong")


def _verify_affected(record: dict) -> None:
    if (
        record.get("expected_mechanism") != gate1g.EXPECTED_MECHANISM["affected"]
        or record.get("mechanism_classification")
        != gate1g.EXPECTED_MECHANISM["affected"]
    ):
        raise AssertionError("affected mechanism prediction failed")
    if _rank_set(record["warmup_enter_records"], "affected warm-up entry") != {
        0,
        1,
    } or _rank_set(record["warmup_return_records"], "affected warm-up return") != {
        0,
        1,
    }:
        raise AssertionError("affected communicator warm-up did not complete")
    if (
        _rank_set(record["allreduce_enqueued_records"], "affected enqueue") != {0}
        or record.get("allreduce_return_records") != []
    ):
        raise AssertionError("rank 0 pending all-reduce markers are wrong")
    if _rank_set(record["ready_observed_records"], "ready observation") != {
        1
    } or _outcome_map(record["rank_outcomes"]) != {1: "injected_rank_failure"}:
        raise AssertionError("rank 1 injected-failure markers are wrong")
    if (
        record.get("termination_classification") != "wall_bound_rank1_teardown_wait"
        or record.get("termination_prediction_matched") is not True
    ):
        raise AssertionError("affected termination prediction failed")
    if (
        record.get("timed_out") is not True
        or record.get("natural_return_code") is not None
        or record.get("reported_return_code") != 124
    ):
        raise AssertionError("affected launcher did not reach the wall bound")
    if (
        _rank_set(record["teardown_enter_records"], "affected teardown entry") != {1}
        or record.get("teardown_return_records") != []
    ):
        raise AssertionError("affected teardown matrix is wrong")
    if record.get("watchdog_marker_seen") is not True:
        raise AssertionError("affected watchdog marker is missing")
    _verify_stack(record)
    flight = record.get("flight_recorder")
    if not isinstance(flight, dict) or flight.get("raw_persisted") is not False:
        raise AssertionError("Flight Recorder summary is malformed")
    files = flight.get("files")
    if (
        flight.get("file_count") != 1
        or not isinstance(files, list)
        or {item.get("rank") for item in files if item.get("decoded") is True} != {0}
    ):
        raise AssertionError("affected arm requires exactly one decodable rank-0 dump")
    local = flight.get("rank0_pending_all_reduce")
    if (
        not isinstance(local, dict)
        or local.get("status") != "matched"
        or local.get("dump_ranks") != [0]
        or local.get("peer_participation_inferred") is not False
    ):
        raise AssertionError("rank-0-local Flight Recorder contract failed")
    candidates = local.get("candidates")
    if (
        not isinstance(candidates, list)
        or len(candidates) != 1
        or any(
            item.get("group_members") != [0, 1]
            or item.get("observed_rank") != 0
            or "ALLREDUCE" not in str(item.get("operation", "")).replace("_", "")
            or item.get("state") == "completed"
            for item in candidates
        )
    ):
        raise AssertionError("pending rank-0 all-reduce candidate is malformed")
    if record.get("library_log_flags") != {
        "0": gate1f.EXPECTED_RANK0_FLAGS,
        "1": gate1f.EXPECTED_RANK1_FLAGS,
    }:
        raise AssertionError("affected ProcessGroupNCCL flags changed")


def verify(root: Path) -> None:
    expected = {"control-trial-1.json", "affected-trial-1.json"}
    if {path.name for path in root.glob("*.json")} != expected:
        raise AssertionError(
            "Gate 1g requires exactly one control and one affected result"
        )
    records = {}
    for arm in ("control", "affected"):
        record = json.loads((root / f"{arm}-trial-1.json").read_text(encoding="utf-8"))
        _verify_common(record, arm)
        records[arm] = record
    if records["control"]["preflight"] != records["affected"]["preflight"]:
        raise AssertionError("environment identity changed between arms")
    _verify_control(records["control"])
    _verify_affected(records["affected"])
    print("PASS (Gate 1g: FSDP-free ProcessGroupNCCL shutdown/dump gap reproduced)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    verify(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
