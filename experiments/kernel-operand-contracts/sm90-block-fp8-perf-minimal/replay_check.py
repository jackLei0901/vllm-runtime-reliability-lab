"""Qualitative V2 graph replay witness. Never infer exact M or GPU time share."""

import argparse
import json
import sqlite3
import subprocess
from collections import defaultdict
from contextlib import closing
from pathlib import Path

import runtime

LEVELS = (1, 16, 63, 64)


def batch_windows(loads):
    windows = {}
    for row in loads:
        level = row["concurrency"]
        if level not in LEVELS or level in windows:
            raise ValueError("duplicate/unexpected batch level")
        start, end = row["start_utc_ns"], row["end_utc_ns"]
        elapsed = row["end_monotonic_ns"] - row["start_monotonic_ns"]
        if (
            not all(isinstance(v, int) for v in (start, end, elapsed))
            or not 0 < elapsed
        ):
            raise ValueError("invalid batch clock")
        if start < 10**18 or end <= start or abs((end - start) - elapsed) > 50_000_000:
            raise ValueError("wall-clock discontinuity or invalid UTC basis")
        usage = row["usage"]
        if len(usage) != level or any(u["completion_tokens"] != 64 for u in usage):
            raise ValueError("fixed workload incomplete")
        windows[level] = (start, end)
    if set(windows) != set(LEVELS):
        raise ValueError("missing batch level")
    ordered = sorted(windows.values())
    if any(a[1] >= b[0] for a, b in zip(ordered, ordered[1:], strict=False)):
        raise ValueError("overlapping batch windows")
    return windows


def witness(path, loads, config):
    windows = batch_windows(loads)
    compilation = config["compilation_config"]
    capture = compilation["cudagraph_capture_sizes"]
    if 63 in capture or min((n for n in capture if n >= 63), default=None) != 64:
        raise ValueError("padding configuration not corroborated")
    graph_mode = compilation["cudagraph_mode"]
    # The pinned Enum can serialize as its name, value 2, or [2, 0/1].
    full_mode = graph_mode in (2, "FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE")
    full_mode |= isinstance(graph_mode, (list, tuple)) and graph_mode in (
        [2, 0],
        [2, 1],
        (2, 0),
        (2, 1),
    )
    if not full_mode:
        raise ValueError("single full-decode-graph witness requires FULL-capable mode")
    with closing(
        sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    ) as db:
        db.row_factory = sqlite3.Row
        tables = {
            r[0]
            for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        required = {
            "StringIds",
            "NVTX_EVENTS",
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "CUPTI_ACTIVITY_KIND_RUNTIME",
        }
        if not required <= tables:
            raise ValueError("missing replay tables")
        strings = dict(db.execute("SELECT id,value FROM StringIds"))
        apis = [
            dict(r)
            for r in db.execute(
                "SELECT rowid AS row_id,* FROM CUPTI_ACTIVITY_KIND_RUNTIME"
            )
        ]
        ranges = [dict(r) for r in db.execute("SELECT * FROM NVTX_EVENTS")]
        kernels = [
            dict(r)
            for r in db.execute(
                "SELECT rowid AS row_id,* FROM CUPTI_ACTIVITY_KIND_KERNEL"
            )
        ]
    launches = [r for r in apis if "cudaGraphLaunch" in strings.get(r["nameId"], "")]
    by_correlation = defaultdict(list)
    for row in kernels:
        by_correlation[(row["globalPid"], row["correlationId"])].append(row)
    selected = {level: [] for level in LEVELS}
    errors = {level: [] for level in LEVELS}
    identities = set()
    for api in sorted(launches, key=lambda r: (r["start"], r["row_id"])):
        levels = [
            level
            for level, (start, end) in windows.items()
            if start <= api["start"] < api["end"] <= end
        ]
        if not levels:
            continue
        level = levels[0]
        # CPU decode range contains the launch API; GPU execution need not
        # occur inside that CPU range. No capture-to-replay mapping is used.
        decode = [
            r
            for r in ranges
            if r.get("end") is not None
            and r["globalTid"] == api["globalTid"]
            and r["start"] <= api["start"] < api["end"] <= r["end"]
            and "_context_0(" in (r.get("text") or strings.get(r.get("textId"), ""))
            and "_generation_" in (r.get("text") or strings.get(r.get("textId"), ""))
        ]
        if not decode:
            continue  # prefill/capture/nondecode graph, not a selected step
        if len(decode) != 1 or len(levels) != 1 or api.get("returnValue", 0) != 0:
            errors[level].append("ambiguous or failed graph launch")
            continue
        pid = api["globalTid"] & ~((1 << 24) - 1)
        key = pid, api["correlationId"]
        if key in identities:
            errors[level].append("reused graph correlation")
            continue
        identities.add(key)
        target = [
            r
            for r in by_correlation[key]
            if "cutlass_3x_gemm_fp8_blockwise" in strings.get(r["demangledName"], "")
        ]
        expected = "Pingpong" if level == 1 else "Cooperative"
        valid = (
            len(target) == 144
            and all(expected in strings.get(r["demangledName"], "") for r in target)
            and all(r.get("graphNodeId") is not None for r in target)
            and len({r["graphNodeId"] for r in target}) == 144
        )
        selected[level].append(
            {
                "api_row_id": api["row_id"],
                "launch_count": len(target),
                "valid": valid,
                "kernel_row_ids": [r["row_id"] for r in target],
            }
        )
    result = {}
    for level in LEVELS:
        # Fixed first ten qualifying decode graph launches, never pick ten
        # successful steps after inspecting the kernel counts/classes.
        first = selected[level][:10]
        ok = not errors[level] and len(first) == 10 and all(r["valid"] for r in first)
        result[level] = {
            "status": "witnessed" if ok else "unwitnessed",
            "steps": first,
            "errors": errors[level],
            "candidate_step_count": len(selected[level]),
        }
    return {
        "status": "witnessed"
        if all(r["status"] == "witnessed" for r in result.values())
        else "unwitnessed",
        "levels": result,
        "exact_m": "unmeasured",
        "padding": "if a step has 63 decode tokens, configured capture predicts 64",
        "time_share": "not measured",
    }


def main():
    parser = argparse.ArgumentParser()
    for name in ("collection", "nsys-report", "out", "freeze-commit"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    out = runtime.fresh(args.out)
    result = {"status": "unwitnessed", "packet": runtime.packet_hashes()}
    try:
        result["public_freeze"] = runtime.public_freeze(args.freeze_commit)
        collection = json.loads(Path(args.collection).read_text())
        if (
            collection["status"] != "collected_review_pending"
            or collection["packet"] != runtime.packet_hashes()
            or collection["public_freeze"] != result["public_freeze"]
            or collection["mode"] != "nsys"
        ):
            raise ValueError("new frozen collection required")
        if len(collection["runs"]) != 1:
            raise ValueError("one forced run required")
        run = collection["runs"][0]
        if (
            run["configuration"] != "forced"
            or run["status"] != "collected"
            or run["timestamp_basis"] != "utc_epoch_ns"
            or run.get("runner_witness") is not True
        ):
            raise ValueError("forced UTC-window collection required")
        if runtime.sha(args.nsys_report) not in run["artifact_sha256"].values():
            raise ValueError("report digest differs")
        database = out / "serve.sqlite"
        command = [
            "nsys",
            "export",
            "--type=sqlite",
            "--ts-normalize=true",
            "--output",
            str(database),
            args.nsys_report,
        ]
        result["export_command"] = command
        subprocess.run(command, check=True, timeout=120)
        result.update(witness(database, run["loads"], run["resolved_config"]))
        result.update(
            sqlite_sha256=runtime.sha(database),
            collection_sha256=runtime.sha(args.collection),
            nsys_report_sha256=runtime.sha(args.nsys_report),
            gpu=collection["gpu"],
        )
    except Exception as exc:  # noqa: BLE001 - E4 is supporting only
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        runtime.write(out / "replay.json", result)
    print(json.dumps({"status": result["status"]}))
    if result["status"] != "witnessed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
