"""Verify the frozen Gate 1f shutdown-stage diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gate1f_campaign
import verify_gate1c as gate1c_verifier
import verify_gate1d as gate1d_verifier

HERE = Path(__file__).resolve().parent
REPRODUCER = HERE / "gate1d_reproducer.py"


def verify(root: Path) -> None:
    expected = {"affected-trial-1.json"}
    actual = {path.name for path in root.glob("*.json")}
    if actual != expected:
        raise AssertionError("Gate 1f requires exactly one affected result")
    record = json.loads((root / "affected-trial-1.json").read_text(encoding="utf-8"))
    if record.get("schema_version") != 4:
        raise AssertionError("unexpected Gate 1f schema version")
    if record.get("arm") != "affected" or record.get("trial") != 1:
        raise AssertionError("Gate 1f result identity is wrong")
    if record.get("runner_error_type") is not None:
        raise AssertionError("campaign runner reported an internal error")
    if record.get("diagnostic_change") != "TORCH_CPP_LOG_LEVEL=INFO":
        raise AssertionError("Gate 1f diagnostic change was not recorded")
    if record.get("torch_version") != gate1d_verifier.EXPECTED_TORCH:
        raise AssertionError("unexpected torch version")
    if record.get("cuda_visible_devices") != 2:
        raise AssertionError("Gate 1f requires two visible GPUs")
    names = record.get("gpu_names")
    if not isinstance(names, list) or len(names) != 2 or len(set(names)) != 1:
        raise AssertionError("Gate 1f requires two identical GPUs")
    if record.get("rank_identities_recorded") is not True:
        raise AssertionError("rank identities were not recorded")
    if record.get("no_tracked_orphans") is not True:
        raise AssertionError("lifecycle contract failed")
    if record.get("raw_output_persisted") is not False:
        raise AssertionError("raw launcher output was retained")
    if record.get("library_log_scan_error") is not None:
        raise AssertionError("library log scan failed closed")

    gate1d_verifier._verify_configuration(record)
    gate1d_verifier._verify_reproducer_hash(record)
    gate1d_verifier._verify_ptrace(record)
    gate1c_verifier.REPRODUCER = REPRODUCER
    gate1c_verifier._verify_mechanism(record, "affected")
    gate1d_verifier._verify_termination_observation(record, "affected")
    stacks = gate1c_verifier._rank_map(
        record.get("stack_capture", []), "stack evidence", {0, 1}
    )
    gate1c_verifier._verify_stack_frame(
        stacks[0], gate1c_verifier._call_line("barrier"), "rank 0"
    )
    gate1c_verifier._verify_stack_frame(
        stacks[1], gate1c_verifier._call_line("destroy_process_group"), "rank 1"
    )

    flight = record.get("flight_recorder")
    if not isinstance(flight, dict) or flight.get("raw_persisted") is not False:
        raise AssertionError("Flight Recorder evidence is malformed")
    files = flight.get("files")
    if (
        flight.get("file_count") != 1
        or not isinstance(files, list)
        or {item.get("rank") for item in files if item.get("decoded") is True} != {0}
    ):
        raise AssertionError("Gate 1f expected one decodable rank-0 dump")
    strict = flight.get("strict_pending_reduce")
    if not isinstance(strict, dict) or strict != {
        "status": "incomplete_dump_set",
        "dump_ranks": [0],
        "missing_dump_ranks": [1],
        "candidates": [],
    }:
        raise AssertionError("Gate 1f dump-set boundary changed")

    preflight = record.get("preflight")
    if not isinstance(preflight, dict) or preflight.get("nccl_version") != [2, 29, 7]:
        raise AssertionError("NCCL 2.29.7 provenance is missing")
    flags = record.get("library_log_flags")
    if not isinstance(flags, dict) or set(flags) != {"0", "1"}:
        raise AssertionError("per-rank library log flags are malformed")
    expected_rank0 = gate1f_campaign.EXPECTED_RANK0_FLAGS
    expected_rank1 = gate1f_campaign.EXPECTED_RANK1_FLAGS
    if record.get("expected_rank0_library_log_flags") != expected_rank0:
        raise AssertionError("frozen rank-0 log prediction changed")
    if record.get("expected_rank1_library_log_flags") != expected_rank1:
        raise AssertionError("frozen rank-1 log prediction changed")
    if flags["0"] != expected_rank0:
        raise AssertionError("predeclared rank-0 dump-signal prediction failed")
    if flags["1"] != expected_rank1:
        raise AssertionError("predeclared rank-1 shutdown-stage prediction failed")
    if record.get("rank0_library_log_prediction_matched") is not True:
        raise AssertionError("rank-0 prediction match flag is inconsistent")
    if record.get("rank1_library_log_prediction_matched") is not True:
        raise AssertionError("rank-1 prediction match flag is inconsistent")
    for rank_flags in flags.values():
        if set(rank_flags) != set(gate1f_campaign.LOG_MESSAGES) or not all(
            isinstance(value, bool) for value in rank_flags.values()
        ):
            raise AssertionError("library log allow-list is inconsistent")

    output_hash = record.get("output_sha256")
    if not isinstance(output_hash, str) or len(output_hash) != 64:
        raise AssertionError("launcher output hash is malformed")
    print(
        "PASS (Gate 1f: NCCL 2.29.7 shutdown-stage prediction confirmed; "
        "strict Gate 1e capture remains failed closed)"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    verify(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
