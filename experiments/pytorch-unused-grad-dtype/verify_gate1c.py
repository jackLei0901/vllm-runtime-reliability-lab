"""Verify the frozen Gate 1c mechanism, termination and capture contracts."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

EXPECTED_MECHANISM = {
    "control": "completed_symmetric_fp32",
    "affected": "rank1_assertion_rank0_barrier_wait",
}
EXPECTED_TERMINATION = {
    "control": "normal_completion",
    "affected": "wall_bound_rank1_teardown_wait",
}
EXPECTED_TORCH = "2.13.0+cu130"
HERE = Path(__file__).resolve().parent
REPRODUCER = HERE / "gate1c_reproducer.py"


def _rank_map(records: list[dict], label: str, ranks: set[int]) -> dict[int, dict]:
    grouped: dict[int, list[dict]] = {rank: [] for rank in ranks}
    for record in records:
        rank = record.get("rank")
        if rank not in grouped:
            raise AssertionError(f"{label} has an unexpected rank")
        grouped[rank].append(record)
    if any(len(items) != 1 for items in grouped.values()):
        raise AssertionError(
            f"{label} must contain exactly one record per expected rank"
        )
    return {rank: items[0] for rank, items in grouped.items()}


def _rank_set(records: list[dict], label: str) -> set[int]:
    ranks = [item.get("rank") for item in records]
    if any(not isinstance(rank, int) or rank not in (0, 1) for rank in ranks):
        raise AssertionError(f"{label} has an unexpected rank")
    if len(ranks) != len(set(ranks)):
        raise AssertionError(f"{label} has duplicate rank records")
    return set(ranks)


def _dtype_set(record: dict) -> tuple[str, ...]:
    values = record.get("grad_dtypes")
    if not isinstance(values, list) or not values:
        raise AssertionError("dtype evidence is missing")
    return tuple(sorted(set(values)))


def _call_line(function_name: str) -> int:
    tree = ast.parse(REPRODUCER.read_text(encoding="utf-8"))
    lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == function_name
    ]
    if len(lines) != 1:
        raise AssertionError(f"expected one {function_name} call in the reproducer")
    return lines[0]


def _verify_mechanism(record: dict, arm: str) -> None:
    if record.get("expected_mechanism") != EXPECTED_MECHANISM[arm]:
        raise AssertionError("frozen mechanism expectation is inconsistent")
    if record.get("mechanism_classification") != EXPECTED_MECHANISM[arm]:
        raise AssertionError("predeclared mechanism expectation failed")
    reduces = _rank_map(record.get("reduce_records", []), "reduce evidence", {0, 1})
    if any(
        item.get("call_index") != 1
        or item.get("grad_count") != len(item.get("grad_dtypes", []))
        for item in reduces.values()
    ):
        raise AssertionError("reduce evidence count or call index is wrong")
    returned = _rank_set(record.get("reduce_return_records", []), "reduce returns")
    barrier_entered = _rank_set(
        record.get("barrier_enter_records", []), "barrier entries"
    )
    barrier_returned = _rank_set(
        record.get("barrier_return_records", []), "barrier returns"
    )
    outcomes = record.get("rank_outcomes", [])
    if arm == "control":
        if any(_dtype_set(reduces[rank]) != ("torch.float32",) for rank in (0, 1)):
            raise AssertionError("control dtype matrix is wrong")
        if (
            returned != {0, 1}
            or barrier_entered != {0, 1}
            or barrier_returned != {0, 1}
        ):
            raise AssertionError("control reduce/barrier matrix is wrong")
        outcome_map = _rank_map(outcomes, "rank outcomes", {0, 1})
        if any(item.get("outcome") != "completed" for item in outcome_map.values()):
            raise AssertionError("control outcome matrix is wrong")
        return
    if _dtype_set(reduces[0]) != ("torch.float32",):
        raise AssertionError("rank 0 dtype matrix is wrong")
    if _dtype_set(reduces[1]) != ("torch.bfloat16", "torch.float32"):
        raise AssertionError("rank 1 dtype matrix is wrong")
    if returned != {0} or barrier_entered != {0} or barrier_returned:
        raise AssertionError("affected reduce/barrier matrix is wrong")
    outcome = _rank_map(outcomes, "rank outcomes", {1})[1]
    if outcome.get("outcome") != "local_uniformity_assertion":
        raise AssertionError("rank 1 did not emit the expected local assertion")


def _verify_termination(record: dict, arm: str) -> None:
    if record.get("expected_termination") != EXPECTED_TERMINATION[arm]:
        raise AssertionError("frozen termination expectation is inconsistent")
    if record.get("termination_classification") != EXPECTED_TERMINATION[arm]:
        raise AssertionError("predeclared termination expectation failed")
    entered = _rank_set(record.get("teardown_enter_records", []), "teardown entries")
    returned = _rank_set(record.get("teardown_return_records", []), "teardown returns")
    if arm == "control":
        if record.get("timed_out") or record.get("natural_return_code") != 0:
            raise AssertionError("control did not complete naturally")
        if entered != {0, 1} or returned != {0, 1}:
            raise AssertionError("control teardown matrix is wrong")
        return
    if not record.get("timed_out") or record.get("reported_return_code") != 124:
        raise AssertionError("affected arm did not reach the wall bound")
    if record.get("natural_return_code") is not None:
        raise AssertionError("affected launcher exited before cleanup")
    if entered != {1} or returned:
        raise AssertionError("affected teardown matrix is wrong")


def _verify_stack_frame(stack: dict, expected_line: int, label: str) -> None:
    if stack.get("status") != "captured":
        raise AssertionError(f"{label} stack was not captured")
    if stack.get("raw_persisted") is not False:
        raise AssertionError(f"{label} raw stack was retained")
    frames = stack.get("project_frames")
    if not isinstance(frames, list) or not any(
        item.get("function") == "main" and item.get("line") == expected_line
        for item in frames
    ):
        raise AssertionError(f"{label} stack did not show the expected wait site")


def _verify_capture(record: dict, arm: str) -> None:
    stacks = record.get("stack_capture")
    flight = record.get("flight_recorder")
    if not isinstance(stacks, list) or not isinstance(flight, dict):
        raise AssertionError("capture evidence is malformed")
    if flight.get("raw_persisted") is not False:
        raise AssertionError("raw Flight Recorder output was retained")
    if arm == "control":
        if stacks or flight.get("file_count") != 0:
            raise AssertionError("control must not trigger incident capture")
        return
    stack_map = _rank_map(stacks, "stack evidence", {0, 1})
    _verify_stack_frame(stack_map[0], _call_line("barrier"), "rank 0")
    _verify_stack_frame(stack_map[1], _call_line("destroy_process_group"), "rank 1")
    files = flight.get("files")
    if not isinstance(files, list) or flight.get("file_count") != len(files):
        raise AssertionError("Flight Recorder file inventory is inconsistent")
    if {item.get("rank") for item in files if item.get("decoded") is True} != {0, 1}:
        raise AssertionError(
            "both rank dumps must decode before participation is inferred"
        )
    normalized = flight.get("normalized")
    if (
        not isinstance(normalized, dict)
        or normalized.get("normalization") == "failed_closed"
    ):
        raise AssertionError("Flight Recorder normalization failed")
    strict = flight.get("strict_pending_reduce")
    if not isinstance(strict, dict) or strict.get("status") != "matched":
        raise AssertionError("strict pending reduce-scatter evidence did not match")
    if strict.get("dump_ranks") != [0, 1] or strict.get("missing_dump_ranks") != []:
        raise AssertionError(
            "dump absence was confused with collective non-participation"
        )
    candidates = strict.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise AssertionError("pending reduce-scatter candidate is missing")
    for candidate in candidates:
        if (
            candidate.get("group_members") != [0, 1]
            or candidate.get("present_ranks") != [0]
            or "REDUCE_SCATTER" not in candidate.get("operation", "")
            or not candidate.get("state_by_rank")
            or candidate["state_by_rank"][0].get("state") == "completed"
        ):
            raise AssertionError("strict pending reduce-scatter candidate is malformed")


def _verify_provenance(record: dict) -> None:
    preflight = record.get("preflight")
    if not isinstance(preflight, dict):
        raise AssertionError("preflight record is missing")
    if not isinstance(preflight.get("effective_uid"), int):
        raise AssertionError("effective UID was not recorded")
    if preflight.get("ptrace_scope") is not None and not isinstance(
        preflight.get("ptrace_scope"), int
    ):
        raise AssertionError("ptrace_scope is malformed")
    if preflight.get("observer_authorization") != "PR_SET_PTRACER campaign parent":
        raise AssertionError("observer authorization was not declared")
    version_hash = preflight.get("py_spy_version_sha256")
    if not isinstance(version_hash, str) or len(version_hash) != 64:
        raise AssertionError("py-spy version evidence is malformed")


def verify(root: Path, trials: int = 3) -> None:
    expected = {
        f"{arm}-trial-{trial}.json"
        for arm in ("control", "affected")
        for trial in range(1, trials + 1)
    }
    actual = {path.name for path in root.glob("*.json")}
    if actual != expected:
        raise AssertionError("Gate 1c result file set is incomplete or contains extras")
    source_hashes = set()
    torch_git_versions = set()
    gpu_names = set()
    preflight_records = set()
    for filename in sorted(expected):
        path = root / filename
        record = json.loads(path.read_text(encoding="utf-8"))
        arm, _, trial_text = path.stem.partition("-trial-")
        if record.get("schema_version") != 2:
            raise AssertionError("unexpected Gate 1c schema version")
        if record.get("arm") != arm or record.get("trial") != int(trial_text):
            raise AssertionError("result identity does not match filename")
        if record.get("runner_error_type") is not None:
            raise AssertionError("campaign runner reported an internal error")
        if record.get("torch_version") != EXPECTED_TORCH:
            raise AssertionError("unexpected torch version")
        if record.get("cuda_visible_devices") != 2:
            raise AssertionError("Gate 1c requires two visible GPUs")
        names = record.get("gpu_names")
        if not isinstance(names, list) or len(names) != 2 or len(set(names)) != 1:
            raise AssertionError("Gate 1c requires two identical GPUs")
        gpu_names.add(tuple(names))
        if record.get("rank_identities_recorded") is not True:
            raise AssertionError("rank identities were not recorded")
        if record.get("no_tracked_orphans") is not True:
            raise AssertionError("lifecycle contract failed")
        if record.get("raw_output_persisted") is not False:
            raise AssertionError("raw launcher output was retained")
        output_hash = record.get("output_sha256")
        if not isinstance(output_hash, str) or len(output_hash) != 64:
            raise AssertionError("launcher output hash is malformed")
        _verify_provenance(record)
        source_hashes.add(record.get("reproducer_sha256"))
        torch_git_versions.add(record.get("torch_git_version"))
        preflight_records.add(json.dumps(record["preflight"], sort_keys=True))
        _verify_mechanism(record, arm)
        _verify_termination(record, arm)
        _verify_capture(record, arm)
    if len(source_hashes) != 1 or None in source_hashes:
        raise AssertionError("reproducer source changed across trials")
    if len(torch_git_versions) != 1 or None in torch_git_versions:
        raise AssertionError("torch git revision changed across trials")
    if len(gpu_names) != 1 or len(preflight_records) != 1:
        raise AssertionError("environment identity changed across trials")
    print(
        f"PASS (Gate 1c: 2 arms x {trials} trials; mechanism, termination, "
        "stack and strict Flight Recorder join)"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    verify(args.root, args.trials)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
