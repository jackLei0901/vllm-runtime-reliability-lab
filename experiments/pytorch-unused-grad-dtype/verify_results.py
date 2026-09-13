"""Verify a frozen campaign result set against an explicit expectation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def verify(
    result_dir: Path,
    trials: int,
    expected_affected: str,
    expected_mode: str = "single",
) -> None:
    expected_files = {
        f"{arm}-trial-{trial}.json"
        for arm in ("control", "affected")
        for trial in range(1, trials + 1)
    }
    actual_files = {path.name for path in result_dir.glob("*.json")}
    if actual_files != expected_files:
        raise AssertionError(
            f"result file set mismatch: expected {sorted(expected_files)}, "
            f"got {sorted(actual_files)}"
        )

    records_by_name = {
        name: json.loads((result_dir / name).read_text(encoding="utf-8"))
        for name in sorted(expected_files)
    }
    records = list(records_by_name.values())
    versions = {item["torch_version"] for item in records}
    if len(versions) != 1:
        raise AssertionError(f"more than one torch version present: {versions}")
    source_hashes = {item["reproducer_sha256"] for item in records}
    if len(source_hashes) != 1 or len(next(iter(source_hashes))) != 64:
        raise AssertionError("results do not share one valid reproducer hash")
    for name, item in records_by_name.items():
        expected_name = f"{item['arm']}-trial-{item['trial']}.json"
        if name != expected_name:
            raise AssertionError(
                f"result identity does not match filename {name}: {item}"
            )
        if item["cuda_visible_devices"] != 2:
            raise AssertionError("every trial must expose exactly two GPUs")
        observed_mode = item.get("execution_mode", "single")
        if observed_mode != expected_mode:
            raise AssertionError(
                f"expected execution mode {expected_mode}, got {observed_mode}"
            )
        if item["raw_output_persisted"]:
            raise AssertionError("raw output must not be retained")
        if not item["rank_identities_recorded"] or not item["no_tracked_orphans"]:
            raise AssertionError(f"lifecycle contract failed: {item}")

    controls = [item for item in records if item["arm"] == "control"]
    if any(item["classification"] != "completed" for item in controls):
        raise AssertionError("every control trial must complete")

    expected_class = {
        "reproduced": "mixed_gradient_dtype_assertion",
        "not-reproduced": "completed",
    }[expected_affected]
    affected = [item for item in records if item["arm"] == "affected"]
    if any(item["classification"] != expected_class for item in affected):
        raise AssertionError(
            f"affected trials do not match predeclared expectation {expected_affected}"
        )
    print(
        f"PASS ({trials} control, {trials} affected; "
        f"affected={expected_affected}; torch={versions.pop()}; "
        f"source={source_hashes.pop()[:12]})"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--expected-affected",
        choices=("reproduced", "not-reproduced"),
        required=True,
    )
    parser.add_argument(
        "--expected-mode", choices=("single", "accumulated"), default="single"
    )
    args = parser.parse_args()
    verify(
        args.result_dir,
        args.trials,
        args.expected_affected,
        args.expected_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
