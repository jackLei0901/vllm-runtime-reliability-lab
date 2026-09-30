"""Fixed 88 cases per arm; no reruns, copied kernel logic or GPU skips."""

from __future__ import annotations

import argparse
import json

import protocol
import runtime


def cases():
    result = []
    for shape in protocol.SHAPES:
        result += [(shape, m, "bf16", "eager") for m in protocol.CORRECTNESS_M]
        result += [(shape, m, "bf16", "graph") for m in protocol.GRAPH_M]
    result += [
        (protocol.FP16_SHAPE, m, "fp16", "eager") for m in protocol.CORRECTNESS_M
    ]
    if len(result) != 88:
        raise ValueError("incorrect fixed matrix")
    return result


def run_case(torch, op, case):
    (n, k), m, dtype, mode = case
    values = runtime.tensors(
        torch, m, n, k, torch.bfloat16 if dtype == "bf16" else torch.float16
    )
    ref = runtime.reference(torch, values)
    op(*values, None)
    torch.cuda.synchronize()
    if mode == "graph":
        # CUDA initialization/warm-up precedes capture on the capture stream.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                op(*values, None)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            op(*values, None)
        # Change input, keeping its storage fixed: exercise replay, not just
        # a result left in the output buffer from capture or warm-up.
        values[1].zero_()
        graph.replay()
        torch.cuda.synchronize()
        if not torch.isfinite(values[0]).all() or values[0].abs().max().item() != 0:
            raise ValueError("graph replay did not consume changed input")
        original = runtime.tensors(torch, m, n, k, values[0].dtype)
        values[1].copy_(original[1])
        graph.replay()
        torch.cuda.synchronize()
    error = runtime.relative_error(torch, values[0], ref)
    if not error < 0.005:
        raise ValueError(f"relative error {error} exceeds 0.005")
    if torch.ones(1, device="cuda").item() != 1:
        raise ValueError("CUDA context unusable after case")
    return error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base")
    parser.add_argument("--variant")
    parser.add_argument("--out")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--freeze-commit")
    args = parser.parse_args()
    if args.plan:
        print(json.dumps({"cases_per_arm": len(cases()), "arm_cases": 176}))
        return
    if not all((args.base, args.variant, args.out)):
        parser.error("--base, --variant and a new --out are required")
    out = runtime.fresh(args.out)
    result = {
        "status": "unscored",
        "packet": runtime.packet_hashes(),
        "cases": [
            {"arm": arm, "case": case, "status": "unscored"}
            for arm in ("base", "variant")
            for case in cases()
        ],
    }
    try:
        result["public_freeze"] = runtime.public_freeze(args.freeze_commit)
        left, lb = runtime.verify_build(args.base, "base")
        right, rb = runtime.verify_build(args.variant, "variant")
        runtime.verify_pair(left, right)
        if left["public_freeze"] != result["public_freeze"]:
            raise ValueError("build freeze differs from run")
        result["build_receipts"] = [runtime.sha(args.base), runtime.sha(args.variant)]
        torch, result["gpu"] = runtime.gpu()
        ops = {
            "base": runtime.load(torch, left, lb),
            "variant": runtime.load(torch, right, rb),
        }
        for row in result["cases"]:
            runtime.write(out / "correctness.json", result)
            try:
                row["relative_error"] = run_case(torch, ops[row["arm"]], row["case"])
                row["status"] = "passed"
            except ValueError as exc:
                row.update(status="failed", failure=str(exc))
                # A numerical miss does not hide later fixed cases. Runtime
                # CUDA failures abort instead; remaining rows stay unscored.
            runtime.write(out / "correctness.json", result)
        result["status"] = (
            "passed"
            if all(r["status"] == "passed" for r in result["cases"])
            else "failed"
        )
    except Exception as exc:  # noqa: BLE001 - abort, preserve first failure
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        runtime.write(out / "correctness.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": sum(r["status"] == "passed" for r in result["cases"]),
            }
        )
    )
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
