"""Verify the dtype mechanism probe against its frozen expectations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dtype_probe_campaign import CASES, MIXED_BF16_FP32, UNIFORM_BF16


def verify(result_dir: Path, trials: int) -> None:
    expected_names = {
        f"{case}-trial-{trial}.json" for case in CASES for trial in range(1, trials + 1)
    }
    actual_names = {path.name for path in result_dir.glob("*.json")}
    if actual_names != expected_names:
        raise AssertionError("dtype-probe result file set mismatch")

    source_hashes: set[str] = set()
    versions: set[str] = set()
    gpu_names: set[tuple[str, ...]] = set()
    for name in sorted(expected_names):
        record = json.loads((result_dir / name).read_text(encoding="utf-8"))
        if name != f"{record['case']}-trial-{record['trial']}.json":
            raise AssertionError(f"identity mismatch in {name}")
        expected = CASES[record["case"]]
        if record["expected_classification"] != expected:
            raise AssertionError(f"expectation drift in {name}")
        if record["classification"] != expected:
            raise AssertionError(f"classification mismatch in {name}")
        records = record["dtype_records"]
        if len(records) != 2 or [item["rank"] for item in records] != [0, 1]:
            raise AssertionError(f"incomplete per-rank dtype evidence in {name}")
        dtype_sets = [tuple(sorted(set(item["grad_dtypes"]))) for item in records]
        required_dtype_set = (
            UNIFORM_BF16 if record["case"] == "unused-parameter" else MIXED_BF16_FP32
        )
        if dtype_sets != [required_dtype_set, required_dtype_set]:
            raise AssertionError(f"unexpected per-rank dtype evidence in {name}")
        expected_assertion = record["case"] == "forced-mixed-gradient"
        if record["assertion_marker_seen"] != expected_assertion:
            raise AssertionError(f"assertion-marker mismatch in {name}")
        expected_nonzero = record["case"] == "forced-mixed-gradient"
        if (record["return_code"] != 0) != expected_nonzero:
            raise AssertionError(f"return-code mismatch in {name}")
        if record["cuda_visible_devices"] != 2:
            raise AssertionError(f"wrong GPU count in {name}")
        names = tuple(record["gpu_names"])
        if len(names) != 2 or len(set(names)) != 1:
            raise AssertionError(f"non-identical two-GPU environment in {name}")
        if record["raw_output_persisted"]:
            raise AssertionError(f"raw output retained in {name}")
        if not record["rank_identities_recorded"]:
            raise AssertionError(f"rank identities missing in {name}")
        if not record["no_tracked_orphans"]:
            raise AssertionError(f"tracked orphan in {name}")
        source_hashes.add(record["reproducer_sha256"])
        versions.add(record["torch_version"])
        gpu_names.add(names)
    if len(source_hashes) != 1 or len(versions) != 1 or len(gpu_names) != 1:
        raise AssertionError("source or PyTorch version changed within the probe")
    print(f"PASS (2 cases x {trials} trials; torch={versions.pop()})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    verify(args.result_dir, args.trials)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
