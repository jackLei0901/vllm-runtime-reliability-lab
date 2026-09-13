"""Run Gate 1e with robust adjacent-marker parsing and control fail-fast."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gate1d_campaign as gate1d

MARKER_PREFIXES = {
    "reduce": "DFX_RANK_REDUCE=",
    "reduce_return": "DFX_RANK_REDUCE_RETURN=",
    "barrier_enter": "DFX_RANK_BARRIER_ENTER=",
    "barrier_return": "DFX_RANK_BARRIER_RETURN=",
    "teardown_enter": "DFX_RANK_TEARDOWN_ENTER=",
    "teardown_return": "DFX_RANK_TEARDOWN_RETURN=",
    "outcome": "DFX_RANK_OUTCOME=",
    "ptrace": "DFX_RANK_PTRACE=",
}
EXPECTED_MECHANISM = gate1d.EXPECTED_MECHANISM
EXPECTED_TERMINATION = gate1d.EXPECTED_TERMINATION


def parse_records(output: str, kind: str) -> list[dict]:
    """Decode every complete marker independently, including adjacent writes."""
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
    records.sort(key=lambda item: (item["rank"], item.get("call_index", 0)))
    return records


def stop_reason(arm: str, result: dict) -> str | None:
    if result["mechanism_classification"] != EXPECTED_MECHANISM[arm]:
        return "mechanism_mismatch"
    if arm == "control" and not result["termination_prediction_matched"]:
        return "control_termination_mismatch"
    return None


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
        parser.error("Gate 1e requires at least a 60-second wall bound")

    gate1d.parse_records = parse_records
    preflight_record = gate1d.preflight(args.py_spy)
    args.output.mkdir(parents=True, exist_ok=False)
    termination_mismatches = 0
    for arm in ("control", "affected"):
        for trial in range(1, args.trials + 1):
            result = gate1d.run_trial(
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
            reason = stop_reason(arm, result)
            if reason is not None:
                print(f"STOP: {reason}; result retained", file=sys.stderr)
                return 2
            if arm == "affected" and not result["termination_prediction_matched"]:
                termination_mismatches += 1
                print("NOTE: affected termination mismatch retained", file=sys.stderr)
    print(f"affected termination prediction mismatches: {termination_mismatches}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
