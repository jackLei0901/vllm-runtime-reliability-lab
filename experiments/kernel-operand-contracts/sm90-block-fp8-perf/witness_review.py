"""Validate a bounded human trace review; never infer replay M from concurrency."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path

import protocol
import runtime


def union_duration(intervals):
    end, total = None, 0
    for a, b in sorted(intervals):
        if b <= a:
            raise ValueError("invalid GPU interval")
        total += (
            b - max(a, end)
            if end is not None and b > end
            else (b - a if end is None else 0)
        )
        end = max(end, b) if end is not None else b
    return total


def validate_capture(events, index, m, n, k):
    event = events[index]
    if "cutlass_scaled_mm" not in event.get("name", ""):
        raise ValueError("forced CUTLASS op shape evidence missing")
    dims = event.get("args", {}).get("Input Dims")
    if not dims or dims[:3] != [[m, n], [m, k], [k, n]]:
        raise ValueError("capture op dimensions mismatch")
    ranges = [r for r in events if r.get("name", "").startswith(f"capture_{m}_")]
    if not any(
        r.get("pid") == event.get("pid")
        and r.get("tid") == event.get("tid")
        and r["ts"] <= event["ts"]
        and event["ts"] + event.get("dur", 0) <= r["ts"] + r.get("dur", 0)
        for r in ranges
    ):
        raise ValueError("shape evidence is not inside the relevant capture")


def inspect_review(collection, mapping):
    if collection.get("status") != "collected_review_pending":
        raise ValueError("Nsight collection incomplete")
    if mapping.get("reviewer") not in ("human", "AI_with_human_review"):
        raise ValueError("human review of capture/replay topology required")
    if (
        mapping.get("mapping_basis")
        != "capture_op_metadata_and_reviewed_replay_topology"
    ):
        raise ValueError("kernel names alone do not establish replay shape")
    if not mapping.get("review_note"):
        raise ValueError("explain how graph nodes were linked to capture metadata")
    trace = Path(mapping["sqlite"])
    if runtime.sha(trace) != mapping["sqlite_sha256"]:
        raise ValueError("trace identity differs")
    profiles = {}
    for item in mapping["capture_profiles"]:
        path = Path(item["path"])
        if runtime.sha(path) != item["sha256"]:
            raise ValueError("capture profile identity differs")
        raw = gzip.open(path, "rt") if path.suffix == ".gz" else path.open()
        with raw:
            profiles[item["sha256"]] = json.load(raw)["traceEvents"]
    with closing(
        sqlite3.connect(trace.resolve().as_uri() + "?mode=ro", uri=True)
    ) as db:
        db.row_factory = sqlite3.Row
        strings = dict(db.execute("SELECT id,value FROM StringIds"))
        kernels = {
            r["row_id"]: dict(r)
            for r in db.execute(
                "SELECT rowid AS row_id,* FROM CUPTI_ACTIVITY_KIND_KERNEL"
            )
        }
        ranges = {
            r["row_id"]: dict(r)
            for r in db.execute("SELECT rowid AS row_id,* FROM NVTX_EVENTS")
        }
    counts = {c: 0 for c in (1, 16, 63, 64)}
    used, used_ranges, summary = set(), set(), []
    for step in mapping["steps"]:
        concurrency = step["concurrency"]
        if concurrency not in counts:
            raise ValueError("unexpected workload load")
        row = ranges[step["nvtx_rowid"]]
        if step["nvtx_rowid"] in used_ranges:
            raise ValueError("decode range counted more than once")
        used_ranges.add(step["nvtx_rowid"])
        text = row.get("text") or strings.get(row.get("textId"), "")
        match = re.match(
            r"execute_(\d+)_context_0\(sq0sk\d+sqsq\d+sqsk\d+\)_generation_(\d+)\(sq(\d+)sk",
            text,
        )
        if not match or [int(x) for x in match.groups()] != [concurrency] * 3:
            raise ValueError(
                "not a uniform steady decode step at the claimed logical M"
            )
        expected_m = 64 if concurrency == 63 else concurrency
        if step["execution_m"] != expected_m:
            raise ValueError("graph M differs from this frozen configuration")
        ids = step["all_kernel_rowids"]
        if not ids or len(set(ids)) != len(ids) or used & set(ids):
            raise ValueError("empty/duplicate GPU step membership")
        used.update(ids)
        selected = [kernels[i] for i in ids]
        if len({r["globalPid"] for r in selected}) != 1:
            raise ValueError("ambiguous CUDA process")
        if not step.get("complete_projected_gpu_step_reviewed"):
            raise ValueError("review completeness of projected GPU step")
        target, shapes, layers, capture_ops = [], set(), set(), set()
        for linear in step["linears"]:
            n, k = linear["shape"]
            if (n, k) not in protocol.SHAPES:
                raise ValueError("unexpected projection shape")
            shapes.add((n, k))
            layer = linear["layer_index"]
            if layer not in range(36) or (layer, n, k) in layers:
                raise ValueError("invalid/duplicate layer projection")
            layers.add((layer, n, k))
            events = profiles[linear["capture_profile_sha256"]]
            capture_key = (
                linear["capture_profile_sha256"],
                linear["capture_event_index"],
            )
            if capture_key in capture_ops:
                raise ValueError("capture op reused for another layer within the step")
            capture_ops.add(capture_key)
            validate_capture(events, linear["capture_event_index"], expected_m, n, k)
            if linear["kernel_rowid"] not in ids:
                raise ValueError("linear kernel not in reviewed step")
            kernel = kernels[linear["kernel_rowid"]]
            if kernel.get("graphNodeId") is None:
                raise ValueError("graph node witness missing")
            if kernel["graphNodeId"] != linear["graph_node_id"]:
                raise ValueError("reviewed graph node differs")
            name = strings[kernel["demangledName"]]
            if "cutlass_3x_gemm_fp8_blockwise" not in name:
                raise ValueError("target CUTLASS kernel not executed")
            target.append((kernel["start"], kernel["end"]))
        if (
            shapes != set(protocol.SHAPES)
            or len(target) != len(set(target))
            or len(layers) != 144
        ):
            raise ValueError(
                "all 36 layers x four projections need distinct execution witnesses"
            )
        fraction = union_duration(target) / union_duration(
            [(r["start"], r["end"]) for r in selected]
        )
        if not 0 < fraction <= 1:
            raise ValueError("invalid GPU time fraction")
        counts[concurrency] += 1
        summary.append(
            {
                "logical_m": concurrency,
                "execution_m": expected_m,
                "target_gpu_fraction": fraction,
            }
        )
    if any(n < 10 for n in counts.values()):
        raise ValueError("at least ten steady decode steps per load required")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = runtime.fresh(args.out)
    result = {"status": "unscored", "packet": runtime.packet_hashes()}
    try:
        collection = json.loads(Path(args.collection).read_text())
        if (
            collection["mode"] != "nsys"
            or collection["packet"] != runtime.packet_hashes()
        ):
            raise ValueError("matching Nsight collector required")
        mapping = json.loads(Path(args.mapping).read_text())
        if mapping["collection_sha256"] != runtime.sha(args.collection):
            raise ValueError("collection binding differs")
        shapes = json.loads(Path(mapping["shape_collection"]).read_text())
        if (
            runtime.sha(mapping["shape_collection"])
            != mapping["shape_collection_sha256"]
        ):
            raise ValueError("shape collector identity differs")
        if (
            shapes.get("status") != "collected_review_pending"
            or shapes.get("mode") != "shapes"
        ):
            raise ValueError("separate native capture profile run required")
        for field in ("packet", "gpu", "model", "installed", "serving_build_sha256"):
            if shapes[field] != collection[field]:
                raise ValueError(f"capture/serving identities differ: {field}")
        forced = next(r for r in collection["runs"] if r["configuration"] == "forced")
        shape_run = next(r for r in shapes["runs"] if r["configuration"] == "forced")
        if runtime.sha(mapping["nsys_report"]) != mapping["nsys_report_sha256"]:
            raise ValueError("original Nsight report identity differs")
        if mapping["nsys_report_sha256"] not in forced["artifact_sha256"].values():
            raise ValueError("report not from forced serving collection")
        allowed_profiles = set(shape_run["artifact_sha256"].values())
        if any(
            p["sha256"] not in allowed_profiles for p in mapping["capture_profiles"]
        ):
            raise ValueError("shape profile not from forced capture collection")
        if runtime.sha(mapping["forced_server_log"]) != forced["log_sha256"]:
            raise ValueError("forced selection log identity differs")
        if (
            "Selected CutlassFp8BlockScaledMMKernel"
            not in Path(mapping["forced_server_log"]).read_text()
        ):
            raise ValueError("CUTLASS selection witness missing")
        capture = forced["resolved_config"]["compilation_config"][
            "cudagraph_capture_sizes"
        ]
        if not {24, 40, 56, 64} <= set(capture) or 63 in capture:
            raise ValueError("capture configuration differs")
        result["steps"] = inspect_review(collection, mapping)
        result["review_grade"] = "human_trace_review_not_automatic_graph_attribution"
        result["mapping_sha256"] = runtime.sha(args.mapping)
        result["collection_sha256"] = runtime.sha(args.collection)
        result["gpu"] = collection["gpu"]
        result["model"] = collection["model"]
        result["public_freeze"] = collection["public_freeze"]
        result["status"] = "witnessed"
    except Exception as exc:  # noqa: BLE001
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        runtime.write(out / "workload.json", result)
    print(json.dumps({"status": result["status"]}))
    if result["status"] != "witnessed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
