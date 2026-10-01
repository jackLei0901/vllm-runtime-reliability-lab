"""Fail-closed parser for the Nsight SQLite dispatch probe, not serving attribution."""

from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import runtime


def dispatch_rows(path):
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        db.row_factory = sqlite3.Row
        tables = {
            r[0]
            for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        required = {"StringIds", "NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_KERNEL"}
        if not required <= tables:
            raise ValueError("missing Nsight tables")
        strings = dict(db.execute("SELECT id,value FROM StringIds"))
        kernels = list(db.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL"))
        # Probe is deliberately one GPU process; avoids cross-process reuse of
        # correlation IDs. Multiple CUDA PIDs are not silently joined.
        if not kernels or len({r["globalPid"] for r in kernels}) != 1:
            raise ValueError("dispatch probe must have exactly one CUDA PID")
        ranges = []
        for row in db.execute("SELECT * FROM NVTX_EVENTS"):
            row = dict(row)
            text = row.get("text") or strings.get(row.get("textId"), "")
            if text in ("labperf:base:m64", "labperf:variant:m64"):
                if row.get("end") is None or row["end"] <= row["start"]:
                    raise ValueError("invalid probe NVTX range")
                ranges.append((row, text.split(":")[1]))
        apis = []
        for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
            if table in tables:
                apis.extend(dict(r) for r in db.execute(f"SELECT * FROM {table}"))
        result = []
        for kernel in kernels:
            name = strings.get(kernel["demangledName"], "")
            if "cutlass_3x_gemm_fp8_blockwise" not in name:
                continue
            labels = {
                arm
                for row, arm in ranges
                for api in apis
                if api["correlationId"] == kernel["correlationId"]
                and api["globalTid"] == row["globalTid"]
                and row["start"] <= api["start"] < api["end"] <= row["end"]
            }
            if not labels:
                continue  # warm-up outside marked ranges
            if len(labels) != 1:
                raise ValueError("ambiguous namespace attribution")
            result.append(
                {
                    "arm": labels.pop(),
                    "name": name,
                    "grid": [kernel[f"grid{x}"] for x in "XYZ"],
                    "block": [kernel[f"block{x}"] for x in "XYZ"],
                }
            )
        return result


def validate_dispatch(rows, arms):
    expected = {"base": "Cooperative", "variant": "Pingpong"}
    groups = {arm: [r for r in rows if r["arm"] == arm] for arm in arms}
    for arm, group in groups.items():
        if len(group) != 3 or any(expected[arm] not in r["name"] for r in group):
            raise ValueError("three correctly attributed M64 launches per arm required")
        if len({r["name"] for r in group}) != 1:
            raise ValueError("ambiguous kernel identity")
    if (
        "variant" in groups
        and groups["base"][0]["name"] == groups["variant"][0]["name"]
    ):
        raise ValueError("both arms resolved to the same kernel")
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = runtime.fresh(args.out)
    result = {"status": "unscored", "packet": runtime.packet_hashes()}
    try:
        probe = json.loads(Path(args.probe).read_text())
        if probe["status"] != "launches_completed_trace_pending":
            raise ValueError("probe did not complete")
        if probe["packet"] != runtime.packet_hashes():
            raise ValueError("probe packet differs")
        arms = ["base", "variant"] if len(probe["build_receipts"]) == 2 else ["base"]
        rows = dispatch_rows(args.sqlite)
        result["launches"] = validate_dispatch(rows, arms)
        result["build_receipts"] = probe["build_receipts"]
        result["sqlite_sha256"] = runtime.sha(args.sqlite)
        result["probe_sha256"] = runtime.sha(args.probe)
        result["gpu"] = probe["gpu"]
        result["public_freeze"] = probe["public_freeze"]
        result["status"] = "witnessed"
    except Exception as exc:  # noqa: BLE001
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        runtime.write(out / "dispatch.json", result)
    print(json.dumps({"status": result["status"]}))
    if result["status"] != "witnessed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
