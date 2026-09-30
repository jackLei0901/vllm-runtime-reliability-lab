"""Paired CUDA graph timings; raw values stay private, no automatic retries."""

from __future__ import annotations

import argparse
import bisect
import json
import time
from pathlib import Path
from statistics import median

import protocol
import runtime


def plan(session):
    cells = [
        (shape, m, "rotating") for shape in protocol.SHAPES for m in protocol.M_VALUES
    ]
    if session == "A":
        cells += [(shape, m, "hot") for shape in protocol.SHAPES for m in (1, 16, 64)]
    pairs = []
    if session == "A":
        pairs += [("boundary", shape, 64, 63) for shape in protocol.SHAPES]
        pairs += [
            ("cache", shape, m, m) for shape in protocol.SHAPES for m in (1, 16, 64)
        ]
    else:
        pairs += [
            ("variant", shape, m, m)
            for shape in protocol.SHAPES
            for m in protocol.M_VALUES
        ]
    return cells, pairs


def graphs(torch, ops, shape, ms, cache_modes, l2):
    n, k = shape
    count, calls = protocol.rotation_plan(l2, n * k + 4 * (n // 128) * (k // 128))
    values = {}
    # Paired arms share input/storage identity, including all rotating weights.
    for m in set(ms):
        values[m] = [
            runtime.tensors(torch, m, n, k, torch.bfloat16, 730 + i)
            for i in range(count)
        ]
    result = []
    for op, m, mode in zip(ops, ms, cache_modes, strict=True):
        selected = values[m] if mode == "rotating" else [values[m][0]]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for inputs in selected:
                op(*inputs, None)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for i in range(calls):
                op(*selected[i % len(selected)], None)
        result.append(graph)
    # Keep every capture's inputs alive until both graphs are discarded.
    return result, values, count, calls


def measure(torch, pair, calls, monitor, deadline, rows, checkpoint):
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(protocol.REPLAYS)
    ]
    for block in range(protocol.BLOCKS):
        if time.monotonic() >= deadline:
            raise TimeoutError("fixed session work window exhausted")
        row = {"block": block, "values": [[], []], "intervals": [], "endpoints": []}
        row["endpoints"].append(monitor.sample())
        for side in protocol.pair_order(block, "left", "right"):
            index = 0 if side == "left" else 1
            graph = pair[index]
            for _ in range(protocol.WARMUP):
                graph.replay()
            torch.cuda.synchronize()
            start = time.monotonic()
            for begin, end in events:
                begin.record()
                graph.replay()
                end.record()
            torch.cuda.synchronize()
            finish = time.monotonic()
            row["values"][index] += [
                begin.elapsed_time(end) * 1000 / calls for begin, end in events
            ]
            row["intervals"].append([start, finish])
        row["endpoints"].append(monitor.sample())
        if monitor.failure:
            raise ValueError("NVML sampling failed")
        rows.append(row)
        checkpoint()
    return rows


def interval_samples(samples, timestamps, intervals):
    return [
        s
        for a, b in intervals
        for s in samples[
            bisect.bisect_left(timestamps, a) : bisect.bisect_right(timestamps, b)
        ]
    ]


def score_blocks(rows, samples, timestamps, clock, limit):
    differences, references = [], []
    for row in rows:
        hot = interval_samples(samples, timestamps, row["intervals"])
        row["hot_samples"] = hot
        row["valid"] = runtime.clock_valid(hot, clock, limit)
        if row["valid"]:
            difference = protocol.paired_difference(*row["values"])
            references.append(median(row["values"][1]))
            differences.append(difference)
        else:
            differences.append(None)
    return differences, references


def score(records, samples):
    # Reference from all actual timed positions, not idle/warm-up snapshots.
    intervals = [v for r in records for row in r["blocks"] for v in row["intervals"]]
    timestamps = [s["time"] for s in samples]
    hot = interval_samples(samples, timestamps, intervals)
    if not hot or len({s["limit_mw"] for s in samples}) != 1:
        raise ValueError("missing hot-state clock witness or changed power setting")
    clock, limit = median(s["sm_mhz"] for s in hot), samples[0]["limit_mw"]
    calibration = {}
    for record in records:
        diff, refs = score_blocks(record["blocks"], samples, timestamps, clock, limit)
        record["differences"] = diff
        record["status"] = "unscored"
        key = (tuple(record["shape"]), record["left_m"], record["left_cache"])
        if record["kind"] == "AA":
            try:
                bound = protocol.decision_bound(
                    median(refs), [x for x in diff if x is not None]
                )
            except (ValueError, IndexError):
                continue
            calibration[key] = bound
            record.update(status="calibrated", bound_us=bound)
        else:
            other = (tuple(record["shape"]), record["right_m"], record["right_cache"])
            if key not in calibration or other not in calibration:
                continue
            valid = [x for x in diff if x is not None]
            if len(valid) < protocol.REQUIRED_BLOCKS:
                continue
            bound = max(calibration[key], calibration[other])
            record.update(bound_us=bound, median_difference_us=median(valid))
            if protocol.exceeds_bound(diff, bound):
                record["status"] = "positive"
            elif median(valid) < -bound:
                record["status"] = "negative"
            else:
                record["status"] = "below_positive_gate"
    return {"reference_sm_mhz": clock, "fixed_limit_mw": limit}


def verify_artifact(path, status, build_hashes):
    data = json.loads(Path(path).read_text())
    if data.get("status") != status or data.get("packet") != runtime.packet_hashes():
        raise ValueError("artifact status/packet mismatch")
    if data.get("build_receipts") != build_hashes:
        raise ValueError("artifact build identities differ")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", choices=["A", "B"], required=True)
    parser.add_argument("--base")
    parser.add_argument("--variant")
    parser.add_argument("--dispatch")
    parser.add_argument("--workload")
    parser.add_argument("--correctness")
    parser.add_argument("--session-a")
    parser.add_argument("--out")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--freeze-commit")
    args = parser.parse_args()
    cells, pairs = plan(args.session)
    if args.plan:
        print(
            json.dumps(
                {
                    "calibration_cells": len(cells),
                    "pairs": len(pairs),
                    "replays": (len(cells) + len(pairs)) * 392,
                }
            )
        )
        return
    if not all((args.base, args.dispatch, args.workload, args.out)):
        parser.error("base, dispatch/workload witnesses and a new output required")
    if args.session == "B" and not all(
        (args.variant, args.correctness, args.session_a)
    ):
        parser.error("B requires variant, full correctness and Session A gate")
    out = runtime.fresh(args.out)
    result = {
        "status": "unscored",
        "session": args.session,
        "packet": runtime.packet_hashes(),
        "records": [],
    }
    monitor = None
    try:
        result["public_freeze"] = runtime.public_freeze(args.freeze_commit)
        base, binary = runtime.verify_build(args.base, "base")
        if base["public_freeze"] != result["public_freeze"]:
            raise ValueError("build freeze differs from run")
        builds = [runtime.sha(args.base)]
        variant = vb = None
        if args.session == "B":
            variant, vb = runtime.verify_build(args.variant, "variant")
            runtime.verify_pair(base, variant)
            builds.append(runtime.sha(args.variant))
            correct = verify_artifact(args.correctness, "passed", builds)
            if len(correct["cases"]) != 176 or any(
                r["status"] != "passed" for r in correct["cases"]
            ):
                raise ValueError("full correctness matrix not passed")
            previous = json.loads(Path(args.session_a).read_text())
            if (
                previous.get("status") != "gap_supported"
                or previous.get("packet") != runtime.packet_hashes()
            ):
                raise ValueError("Session A gap gate not passed")
            if previous.get("build_receipts") != builds[:1]:
                raise ValueError("Session A base differs")
        result["build_receipts"] = builds
        dispatch = verify_artifact(args.dispatch, "witnessed", builds)
        if dispatch["public_freeze"] != result["public_freeze"]:
            raise ValueError("dispatch witness freeze differs")
        workload = json.loads(Path(args.workload).read_text())
        if (
            workload.get("status") != "witnessed"
            or workload.get("packet") != runtime.packet_hashes()
        ):
            raise ValueError("workload/shape witness missing")
        result["input_sha256"] = {
            "dispatch": runtime.sha(args.dispatch),
            "workload": runtime.sha(args.workload),
        }
        torch, result["gpu"] = runtime.gpu()
        if workload["public_freeze"] != result["public_freeze"]:
            raise ValueError("workload witness freeze differs")
        if args.session == "B" and correct["gpu"] != result["gpu"]:
            raise ValueError("correctness GPU identity differs")
        if args.session == "B":
            result["cross_session_uuid_changed"] = runtime.compatible_gpu(
                previous["gpu"], result["gpu"]
            )
            runtime.compatible_gpu(workload["gpu"], result["gpu"])
            result["session_a_gpu"] = previous["gpu"]
        elif workload["gpu"] != result["gpu"]:
            raise ValueError("workload GPU identity differs")
        if dispatch["gpu"] != result["gpu"]:
            raise ValueError("dispatch probe GPU identity differs")
        ops = {"base": runtime.load(torch, base, binary)}
        if variant:
            ops["variant"] = runtime.load(torch, variant, vb)
        monitor = runtime.Monitor(torch)
        result["gpu_uuid"] = monitor.uuid
        deadline = time.monotonic() + (1800 if args.session == "A" else 1200)
        specifications = [
            ("AA", shape, m, m, cache, cache) for shape, m, cache in cells
        ]
        for kind, shape, lm, rm in pairs:
            lc, rc = (
                ("rotating", "hot") if kind == "cache" else ("rotating", "rotating")
            )
            specifications.append((kind, shape, lm, rm, lc, rc))
        start = time.monotonic()
        for index, (kind, shape, lm, rm, lc, rc) in enumerate(specifications):
            selected_ops = [
                ops["base"],
                ops["variant"] if kind == "variant" else ops["base"],
            ]
            pair, keepalive, count, calls = graphs(
                torch,
                selected_ops,
                shape,
                [lm, rm],
                [lc, rc],
                result["gpu"]["l2_bytes"],
            )
            record = {
                "kind": kind,
                "shape": shape,
                "left_m": lm,
                "right_m": rm,
                "left_cache": lc,
                "right_cache": rc,
                "weight_sets": count,
                "calls_per_replay": calls,
            }
            record["blocks"] = []
            result["records"].append(record)
            measure(
                torch,
                pair,
                calls,
                monitor,
                deadline,
                record["blocks"],
                lambda: runtime.write(out / "raw.json", result),
            )
            runtime.write(out / "raw.json", result)
            if index == 2:
                result["first_three_seconds"] = time.monotonic() - start
                result["projected_work_seconds"] = (
                    result["first_three_seconds"] * len(specifications) / 3
                )
            del pair, keepalive
        monitor.close()
        result["clock_samples"] = monitor.samples
        if monitor.failure:
            raise ValueError("clock monitor failed")
        result["clock_reference"] = score(result["records"], monitor.samples)
        if any(r["status"] == "unscored" for r in result["records"]):
            result["status"] = "insufficient_evidence"
        elif args.session == "A":
            boundaries = [r for r in result["records"] if r["kind"] == "boundary"]
            result["status"] = (
                "gap_supported"
                if any(r["status"] == "positive" for r in boundaries)
                else "no_gap"
            )
        else:
            changed = [
                r
                for r in result["records"]
                if r["kind"] == "variant" and r["left_m"] in protocol.CHANGED_M
            ]
            unchanged = [
                r
                for r in result["records"]
                if r["kind"] == "variant" and r["left_m"] not in protocol.CHANGED_M
            ]
            rejected = any(r["median_difference_us"] < -r["bound_us"] for r in changed)
            rejected |= any(
                abs(r["median_difference_us"]) > r["bound_us"] for r in unchanged
            )
            relevant = {
                tuple(r["shape"])
                for r in previous["records"]
                if r["kind"] == "boundary" and r["status"] == "positive"
            }
            improvements = [
                r
                for r in changed
                if tuple(r["shape"]) in relevant and r["status"] == "positive"
            ]
            result["status"] = (
                "regression"
                if rejected
                else ("candidate_supported" if improvements else "no_improvement")
            )
    except Exception as exc:  # noqa: BLE001 - retain failed first attempt
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if monitor:
            if monitor.thread.is_alive():
                monitor.close()
            result["clock_samples"] = monitor.samples
        runtime.write(out / "raw.json", result)
    print(json.dumps({"status": result["status"], "cells": len(result["records"])}))
    if result["status"] in ("unscored", "insufficient_evidence"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
