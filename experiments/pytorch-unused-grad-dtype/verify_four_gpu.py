"""Fail-closed verifier for the frozen four-GPU matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_four_gpu_matrix import MATRIX


def verify(result_dir: Path, trials: int) -> None:
    expected_names = {
        f"{case}-trial-{trial}.json"
        for case in MATRIX
        for trial in range(1, trials + 1)
    }
    actual_names = {path.name for path in result_dir.glob("*.json")}
    if actual_names != expected_names:
        raise AssertionError("four-GPU result file set mismatch")

    versions: set[str] = set()
    gpu_names: set[tuple[str, ...]] = set()
    ports: set[int] = set()
    for name in sorted(expected_names):
        record = json.loads((result_dir / name).read_text(encoding="utf-8"))
        expected_name = f"{record['case']}-trial-{record['trial']}.json"
        if name != expected_name:
            raise AssertionError(f"identity mismatch in {name}")
        contract = MATRIX[record["case"]]
        if record["prepared_sha256"] != contract["sha256"]:
            raise AssertionError(f"prepared source mismatch in {name}")
        if record["expected_classification"] != contract["expected"]:
            raise AssertionError(f"expectation drift in {name}")
        if record["classification"] != contract["expected"]:
            raise AssertionError(f"observed result misses expectation in {name}")
        expected_assertion = contract["expected"] == "mixed_gradient_dtype_assertion"
        if record["assertion_marker_seen"] != expected_assertion:
            raise AssertionError(f"assertion-marker mismatch in {name}")
        if (record["return_code"] != 0) != expected_assertion:
            raise AssertionError(f"return-code mismatch in {name}")
        if record["cuda_visible_devices"] != 4:
            raise AssertionError(f"wrong GPU count in {name}")
        if record["raw_output_persisted"]:
            raise AssertionError(f"raw output retained in {name}")
        if not record["rank_identities_recorded"]:
            raise AssertionError(f"rank identities missing in {name}")
        if not record["no_tracked_orphans"]:
            raise AssertionError(f"tracked orphan in {name}")
        if record["detail_enabled"] or record["training_steps"] != 200:
            raise AssertionError(f"workload contract drift in {name}")
        versions.add(record["torch_version"])
        names = tuple(record["gpu_names"])
        if len(names) != 4 or len(set(names)) != 1:
            raise AssertionError(f"non-identical four-GPU environment in {name}")
        gpu_names.add(names)
        port = record["rendezvous_port"]
        if port in ports:
            raise AssertionError(f"reused rendezvous port in {name}")
        ports.add(port)
    if len(versions) != 1 or len(gpu_names) != 1:
        raise AssertionError("environment changed within the matrix")
    print(f"PASS ({len(MATRIX)} cases x {trials} trials; torch={versions.pop()})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    verify(args.result_dir, args.trials)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
