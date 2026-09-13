"""Verify Gate 1d while reporting termination prediction independently."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gate1c_campaign
import verify_gate1c as gate1c_verifier

EXPECTED_MECHANISM = gate1c_verifier.EXPECTED_MECHANISM
EXPECTED_TERMINATION = gate1c_verifier.EXPECTED_TERMINATION
EXPECTED_TORCH = gate1c_verifier.EXPECTED_TORCH
HERE = Path(__file__).resolve().parent
REPRODUCER = HERE / "gate1d_reproducer.py"
RECOGNIZED_TERMINATIONS = {
    "normal_completion",
    "wall_bound_rank1_teardown_wait",
    "wall_bound_other_teardown_pattern",
    "watchdog_teardown",
    "torchrun_teardown",
}


def _verify_configuration(record: dict) -> None:
    if record.get("configuration") != {
        "stack_capture_after_seconds": 20,
        "process_group_timeout_seconds": 30,
        "wall_timeout_seconds": 60,
        "async_error_handling": 3,
    }:
        raise AssertionError("Gate 1d timing or NCCL configuration changed")


def _verify_reproducer_hash(record: dict) -> None:
    actual_hash = hashlib.sha256(REPRODUCER.read_bytes()).hexdigest()
    if record.get("reproducer_sha256") != actual_hash:
        raise AssertionError("result reproducer hash does not match current source")


def _verify_ptrace(record: dict) -> None:
    preflight = record.get("preflight")
    if not isinstance(preflight, dict):
        raise AssertionError("preflight record is missing")
    if not isinstance(preflight.get("effective_uid"), int):
        raise AssertionError("effective UID was not recorded")
    mode = preflight.get("ptrace_mode")
    scope = preflight.get("ptrace_scope")
    if mode == "yama_absent":
        if scope is not None:
            raise AssertionError("Yama absence conflicts with ptrace_scope")
    elif mode == "pr_set_ptracer_parent":
        if not isinstance(scope, int):
            raise AssertionError("Yama ptrace_scope was not recorded")
    else:
        raise AssertionError("unknown ptrace authorization mode")
    version_hash = preflight.get("py_spy_version_sha256")
    if not isinstance(version_hash, str) or len(version_hash) != 64:
        raise AssertionError("py-spy version evidence is malformed")
    ptrace_records = gate1c_verifier._rank_map(
        record.get("rank_ptrace_records", []), "rank ptrace evidence", {0, 1}
    )
    if any(item.get("mode") != mode for item in ptrace_records.values()):
        raise AssertionError("rank ptrace mode disagrees with preflight")


def _verify_termination_observation(record: dict, arm: str) -> bool:
    if record.get("expected_termination") != EXPECTED_TERMINATION[arm]:
        raise AssertionError("frozen termination expectation is inconsistent")
    observed = record.get("termination_classification")
    if observed not in RECOGNIZED_TERMINATIONS:
        raise AssertionError("termination outcome is not recognized")
    recomputed = gate1c_campaign.classify_termination(
        arm,
        timed_out=record.get("timed_out") is True,
        natural_return_code=record.get("natural_return_code"),
        teardown_enter_records=record.get("teardown_enter_records", []),
        teardown_return_records=record.get("teardown_return_records", []),
        watchdog_marker_seen=record.get("watchdog_marker_seen") is True,
    )
    if observed != recomputed:
        raise AssertionError("termination classification does not match evidence")
    matched = observed == EXPECTED_TERMINATION[arm]
    if record.get("termination_prediction_matched") is not matched:
        raise AssertionError("termination match flag is inconsistent")
    return matched


def verify(root: Path, trials: int = 3) -> None:
    expected = {
        f"{arm}-trial-{trial}.json"
        for arm in ("control", "affected")
        for trial in range(1, trials + 1)
    }
    actual = {path.name for path in root.glob("*.json")}
    if actual != expected:
        raise AssertionError("Gate 1d result file set is incomplete or contains extras")
    source_hashes = set()
    torch_git_versions = set()
    gpu_names = set()
    preflight_records = set()
    termination_mismatches = []
    gate1c_verifier.REPRODUCER = REPRODUCER
    for filename in sorted(expected):
        path = root / filename
        record = json.loads(path.read_text(encoding="utf-8"))
        arm, _, trial_text = path.stem.partition("-trial-")
        if record.get("schema_version") != 3:
            raise AssertionError("unexpected Gate 1d schema version")
        if record.get("arm") != arm or record.get("trial") != int(trial_text):
            raise AssertionError("result identity does not match filename")
        if record.get("runner_error_type") is not None:
            raise AssertionError("campaign runner reported an internal error")
        if record.get("torch_version") != EXPECTED_TORCH:
            raise AssertionError("unexpected torch version")
        if record.get("cuda_visible_devices") != 2:
            raise AssertionError("Gate 1d requires two visible GPUs")
        names = record.get("gpu_names")
        if not isinstance(names, list) or len(names) != 2 or len(set(names)) != 1:
            raise AssertionError("Gate 1d requires two identical GPUs")
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
        _verify_configuration(record)
        _verify_reproducer_hash(record)
        _verify_ptrace(record)
        source_hashes.add(record.get("reproducer_sha256"))
        torch_git_versions.add(record.get("torch_git_version"))
        preflight_records.add(json.dumps(record["preflight"], sort_keys=True))
        gate1c_verifier._verify_mechanism(record, arm)
        if not _verify_termination_observation(record, arm):
            termination_mismatches.append(filename)
        gate1c_verifier._verify_capture(record, arm)
    if len(source_hashes) != 1 or None in source_hashes:
        raise AssertionError("reproducer source changed across trials")
    if len(torch_git_versions) != 1 or None in torch_git_versions:
        raise AssertionError("torch git revision changed across trials")
    if len(gpu_names) != 1 or len(preflight_records) != 1:
        raise AssertionError("environment identity changed across trials")
    print(
        f"PASS (Gate 1d: 2 arms x {trials} trials; mechanism, capture, lifecycle; "
        f"termination prediction mismatches={len(termination_mismatches)})"
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
