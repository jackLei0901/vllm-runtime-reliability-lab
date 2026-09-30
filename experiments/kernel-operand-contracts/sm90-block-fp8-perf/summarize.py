"""Explicit public allowlist; raw traces, clock samples and paths stay private."""

import argparse
import json
from pathlib import Path
from statistics import median

import runtime


def summary(data):
    result = {
        key: data[key]
        for key in (
            "status",
            "session",
            "build_receipts",
            "public_freeze",
            "cross_session_uuid_changed",
        )
        if key in data
    }
    result["gpu"] = {
        k: data.get("gpu", {})[k]
        for k in ("name", "sm_count", "l2_bytes", "torch", "torch_cuda")
        if k in data.get("gpu", {})
    }
    result["cells"] = []
    for row in data.get("records", []):
        cell = {
            k: row[k]
            for k in ("kind", "shape", "left_m", "right_m", "left_cache", "right_cache")
        }
        cell["status"] = row.get("status", "unscored")
        for key in ("bound_us", "median_difference_us"):
            if key in row:
                cell[key] = row[key]
        valid = [r for r in row["blocks"] if r.get("valid")]
        cell["valid_blocks"] = len(valid)
        cell["planned_blocks"] = 7
        if valid:
            cell["left_us"] = median(median(r["values"][0]) for r in valid)
            cell["right_us"] = median(median(r["values"][1]) for r in valid)
        result["cells"].append(cell)
    if "cases" in data:
        result["cases"] = {
            status: sum(r["status"] == status for r in data["cases"])
            for status in ("passed", "failed", "unscored")
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = runtime.fresh(args.out)
    data = json.loads(Path(args.input).read_text())
    result = summary(data)
    result["private_receipt_sha256"] = runtime.sha(args.input)
    runtime.write(out / "summary.json", result)
    print(json.dumps({"status": result["status"]}))


if __name__ == "__main__":
    main()
