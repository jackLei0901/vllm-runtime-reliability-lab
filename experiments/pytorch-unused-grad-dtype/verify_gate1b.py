"""Verify the frozen Gate 1b rank matrix and capture contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED_CLASSIFICATION = {
    "control": "completed_symmetric_fp32",
    "affected": "rank1_assertion_rank0_wait",
}
EXPECTED_TORCH = "2.13.0+cu130"


def _rank_map(records: list[dict], label: str) -> dict[int, dict]:
    grouped: dict[int, list[dict]] = {0: [], 1: []}
    for record in records:
        rank = record.get("rank")
        if rank not in grouped:
            raise AssertionError(f"{label} has an unexpected rank")
        grouped[rank].append(record)
    if any(len(items) != 1 for items in grouped.values()):
        raise AssertionError(f"{label} must contain exactly one record per rank")
    return {rank: items[0] for rank, items in grouped.items()}


def _dtype_set(record: dict) -> tuple[str, ...]:
    values = record.get("grad_dtypes")
    if not isinstance(values, list) or not values:
        raise AssertionError("dtype evidence is missing")
    return tuple(sorted(set(values)))


def _verify_mechanism(record: dict, arm: str) -> None:
    if record.get("expected_classification") != EXPECTED_CLASSIFICATION[arm]:
        raise AssertionError("frozen classification is inconsistent")
    if record.get("classification") != EXPECTED_CLASSIFICATION[arm]:
        raise AssertionError("predeclared mechanism expectation failed")
    reduces = _rank_map(record.get("reduce_records", []), "reduce evidence")
    if any(
        item.get("call_index") != 1
        or item.get("grad_count") != len(item.get("grad_dtypes", []))
        for item in reduces.values()
    ):
        raise AssertionError("reduce evidence count or call index is wrong")
    returns = record.get("reduce_return_records", [])
    outcomes = record.get("rank_outcomes", [])
    if arm == "control":
        if record.get("timed_out") or record.get("return_code") != 0:
            raise AssertionError("control did not complete")
        if any(_dtype_set(reduces[rank]) != ("torch.float32",) for rank in (0, 1)):
            raise AssertionError("control dtype matrix is wrong")
        return_map = _rank_map(returns, "reduce-return evidence")
        if {rank: item.get("call_index") for rank, item in return_map.items()} != {
            0: 1,
            1: 1,
        }:
            raise AssertionError("control reduce-return call index is wrong")
        outcome_map = _rank_map(outcomes, "rank outcomes")
        if any(item.get("outcome") != "completed" for item in outcome_map.values()):
            raise AssertionError("control rank outcome is wrong")
        return

    if not record.get("timed_out") or record.get("return_code") != 124:
        raise AssertionError("affected arm must reach the wall-clock bound")
    if _dtype_set(reduces[0]) != ("torch.float32",):
        raise AssertionError("rank 0 dtype matrix is wrong")
    if _dtype_set(reduces[1]) != ("torch.bfloat16", "torch.float32"):
        raise AssertionError("rank 1 dtype matrix is wrong")
    if returns:
        raise AssertionError("affected reduce must not return on either rank")
    if (
        len(outcomes) != 1
        or outcomes[0].get("rank") != 1
        or outcomes[0].get("outcome") != "local_uniformity_assertion"
    ):
        raise AssertionError("affected rank outcome matrix is wrong")


def _verify_capture(record: dict, arm: str) -> None:
    stacks = record.get("stack_capture")
    flight = record.get("flight_recorder")
    if not isinstance(stacks, list) or not isinstance(flight, dict):
        raise AssertionError("capture evidence is malformed")
    if any(item.get("raw_persisted") is not False for item in stacks):
        raise AssertionError("raw stack output was retained")
    if flight.get("raw_persisted") is not False:
        raise AssertionError("raw Flight Recorder output was retained")
    if arm == "control":
        if stacks:
            raise AssertionError("control must not trigger stack capture")
        return

    rank0 = [item for item in stacks if item.get("rank") == 0]
    if len(rank0) != 1 or rank0[0].get("status") != "captured":
        raise AssertionError("rank 0 stack was not captured before timeout")
    if not rank0[0].get("project_frames"):
        raise AssertionError("rank 0 stack has no reproducer frame")
    files = flight.get("files")
    if not isinstance(files, list) or flight.get("file_count") != len(files):
        raise AssertionError("Flight Recorder file inventory is inconsistent")
    if not files or not any(item.get("decoded") is True for item in files):
        raise AssertionError("no decodable Flight Recorder artifact was captured")
    normalized = flight.get("normalized")
    if (
        not isinstance(normalized, dict)
        or normalized.get("normalization") == "failed_closed"
    ):
        raise AssertionError("Flight Recorder normalization failed")
    divergences = normalized.get("secondary_divergences")
    if not isinstance(divergences, list) or not any(
        item.get("reason") in {"missing_member", "all_pending"} for item in divergences
    ):
        raise AssertionError("Flight Recorder did not expose a pending-rank divergence")


def verify(root: Path, trials: int = 3) -> None:
    expected = {
        f"{arm}-trial-{trial}.json"
        for arm in ("control", "affected")
        for trial in range(1, trials + 1)
    }
    actual = {path.name for path in root.glob("*.json")}
    if actual != expected:
        raise AssertionError("Gate 1b result file set is incomplete or contains extras")

    source_hashes = set()
    torch_git_versions = set()
    gpu_names = set()
    for filename in sorted(expected):
        path = root / filename
        record = json.loads(path.read_text(encoding="utf-8"))
        arm, _, trial_text = path.stem.partition("-trial-")
        if record.get("schema_version") != 1:
            raise AssertionError("unexpected Gate 1b schema version")
        if record.get("arm") != arm or record.get("trial") != int(trial_text):
            raise AssertionError("result identity does not match filename")
        if record.get("runner_error_type") is not None:
            raise AssertionError("campaign runner reported an internal error")
        if record.get("torch_version") != EXPECTED_TORCH:
            raise AssertionError("unexpected torch version")
        if record.get("cuda_visible_devices") != 2:
            raise AssertionError("Gate 1b requires two visible GPUs")
        names = record.get("gpu_names")
        if not isinstance(names, list) or len(names) != 2 or len(set(names)) != 1:
            raise AssertionError("Gate 1b requires two identical GPUs")
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
        source_hashes.add(record.get("reproducer_sha256"))
        torch_git_versions.add(record.get("torch_git_version"))
        _verify_mechanism(record, arm)
        _verify_capture(record, arm)

    if len(source_hashes) != 1 or None in source_hashes:
        raise AssertionError("reproducer source changed across trials")
    if len(torch_git_versions) != 1 or None in torch_git_versions:
        raise AssertionError("torch git revision changed across trials")
    if len(gpu_names) != 1:
        raise AssertionError("GPU identity changed across trials")
    print(f"PASS (Gate 1b: 2 arms x {trials} trials; mechanism, capture, lifecycle)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    verify(args.root, args.trials)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
