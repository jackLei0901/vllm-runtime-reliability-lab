"""Run one Gate 1f shutdown-stage diagnostic trial."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import gate1d_campaign as gate1d
import gate1e_campaign as gate1e

LOG_MESSAGES = {
    "shutdown_start": "Starting to destroy process group, flushing operations.",
    "operations_flushed": "Operations flushed, joining watchdog thread.",
    "watchdog_joined_destroying": ("Watchdog joined, destroying NCCL communicators."),
    "destroy_complete": "Destroy complete.",
    "dump_signal_observed": (
        "Observed flight recorder dump signal from another rank via TCPStore."
    ),
    "dump_signal_broadcast": (
        "Broadcasting signal exception_dump to other ranks via TCPStore."
    ),
    "dump_signal_broadcast_failed": (
        "Failed to broadcast signal exception_dump through TCPStore."
    ),
    "dump_success": "Flight Recorder trace successfully dumped.",
}
RANK_PREFIX = re.compile(r"\[PG ID [^\]\r\n]* Rank (?P<rank>\d+)\]\s*")
EXPECTED_RANK0_FLAGS = {
    "shutdown_start": False,
    "operations_flushed": False,
    "watchdog_joined_destroying": False,
    "destroy_complete": False,
    "dump_signal_observed": False,
    "dump_signal_broadcast": True,
    "dump_signal_broadcast_failed": False,
    "dump_success": True,
}
EXPECTED_RANK1_FLAGS = {
    "shutdown_start": True,
    "operations_flushed": True,
    "watchdog_joined_destroying": True,
    "destroy_complete": False,
    "dump_signal_observed": False,
    "dump_signal_broadcast": False,
    "dump_signal_broadcast_failed": False,
    "dump_success": False,
}


class LibraryLogScanError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def empty_library_log_flags() -> dict[str, dict[str, bool]]:
    return {str(rank): {name: False for name in LOG_MESSAGES} for rank in range(2)}


def scan_library_log_flags(output: str) -> dict[str, dict[str, bool]]:
    """Reduce allow-listed ProcessGroupNCCL messages to per-rank booleans."""
    flags = empty_library_log_flags()
    prefixes = list(RANK_PREFIX.finditer(output))
    for name, message in LOG_MESSAGES.items():
        search_from = 0
        while True:
            position = output.find(message, search_from)
            if position < 0:
                break
            preceding = [item for item in prefixes if item.end() <= position]
            if not preceding:
                raise LibraryLogScanError(f"{name}:missing_rank_prefix")
            prefix = preceding[-1]
            if output[prefix.end() : position].strip():
                raise LibraryLogScanError(f"{name}:ambiguous_rank")
            rank = int(prefix.group("rank"))
            if rank not in (0, 1):
                raise LibraryLogScanError(f"{name}:unexpected_rank")
            flags[str(rank)][name] = True
            search_from = position + len(message)
    return flags


def scan_library_log_flags_safe(
    output: str,
) -> tuple[dict[str, dict[str, bool]], str | None]:
    """Return a closed summary even when strict attribution fails."""
    try:
        return scan_library_log_flags(output), None
    except LibraryLogScanError as exc:
        return empty_library_log_flags(), exc.code
    except Exception as exc:  # pragma: no cover - defensive fail-closed path
        return empty_library_log_flags(), f"unexpected_{type(exc).__name__}"


def nccl_version_record() -> list[int]:
    import torch

    value = torch.cuda.nccl.version()
    if (
        not isinstance(value, tuple)
        or not value
        or not all(isinstance(item, int) for item in value)
    ):
        raise RuntimeError(f"unexpected torch.cuda.nccl.version() value: {value!r}")
    return list(value)


def run_trial(
    *, wall_timeout: int, stack_capture_after: int, py_spy: str, preflight: dict
) -> dict:
    observed_flags: dict[str, dict[str, bool]] | None = None
    scan_error: str | None = None

    def parse_with_diagnostics(output: str, kind: str) -> list[dict]:
        nonlocal observed_flags, scan_error
        if observed_flags is None:
            observed_flags, scan_error = scan_library_log_flags_safe(output)
        return gate1e.parse_records(output, kind)

    previous_parser = gate1d.parse_records
    previous_log_level = os.environ.get("TORCH_CPP_LOG_LEVEL")
    gate1d.parse_records = parse_with_diagnostics
    os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    try:
        result = gate1d.run_trial(
            "affected",
            1,
            wall_timeout=wall_timeout,
            stack_capture_after=stack_capture_after,
            py_spy=py_spy,
            preflight_record=preflight,
        )
    finally:
        gate1d.parse_records = previous_parser
        if previous_log_level is None:
            os.environ.pop("TORCH_CPP_LOG_LEVEL", None)
        else:
            os.environ["TORCH_CPP_LOG_LEVEL"] = previous_log_level
    if observed_flags is None:
        raise RuntimeError("library log scanner did not run")
    result["schema_version"] = 4
    result["diagnostic_change"] = "TORCH_CPP_LOG_LEVEL=INFO"
    result["library_log_flags"] = observed_flags
    result["library_log_scan_error"] = scan_error
    result["expected_rank0_library_log_flags"] = EXPECTED_RANK0_FLAGS
    result["expected_rank1_library_log_flags"] = EXPECTED_RANK1_FLAGS
    result["rank0_library_log_prediction_matched"] = (
        scan_error is None and observed_flags["0"] == EXPECTED_RANK0_FLAGS
    )
    result["rank1_library_log_prediction_matched"] = (
        scan_error is None and observed_flags["1"] == EXPECTED_RANK1_FLAGS
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-timeout", type=int, default=60)
    parser.add_argument("--stack-capture-after", type=int, default=20)
    parser.add_argument("--py-spy", default="py-spy")
    args = parser.parse_args()
    if not 0 < args.stack_capture_after < 30:
        parser.error("stack capture must occur before the process-group timeout")
    if args.wall_timeout != 60:
        parser.error("Gate 1f freezes the wall bound at 60 seconds")

    preflight = gate1d.preflight(args.py_spy)
    preflight["nccl_version"] = nccl_version_record()
    if preflight["nccl_version"] != [2, 29, 7]:
        raise RuntimeError(
            f"Gate 1f requires NCCL 2.29.7, got {preflight['nccl_version']}"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    result = run_trial(
        wall_timeout=args.wall_timeout,
        stack_capture_after=args.stack_capture_after,
        py_spy=args.py_spy,
        preflight=preflight,
    )
    path = args.output / "affected-trial-1.json"
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "affected trial 1: mechanism="
        f"{result['mechanism_classification']}; termination="
        f"{result['termination_classification']}; rank0_dump_prediction="
        f"{result['rank0_library_log_prediction_matched']}; "
        "rank1_shutdown_prediction="
        f"{result['rank1_library_log_prediction_matched']}"
    )
    if result["mechanism_classification"] != gate1d.EXPECTED_MECHANISM["affected"]:
        print("STOP: mechanism mismatch retained", file=sys.stderr)
        return 2
    if result["library_log_scan_error"] is not None:
        print("STOP: library log scan error retained", file=sys.stderr)
        return 3
    if not (
        result["rank0_library_log_prediction_matched"]
        and result["rank1_library_log_prediction_matched"]
    ):
        print(
            "STOP: per-rank library log prediction mismatch retained", file=sys.stderr
        )
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
