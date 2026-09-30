"""Run under Nsight Systems; namespaced M64 launches are a dispatch witness."""

import argparse
import json

import runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--variant")
    parser.add_argument("--out", required=True)
    parser.add_argument("--freeze-commit", required=True)
    args = parser.parse_args()
    out = runtime.fresh(args.out)
    result = {"status": "unscored", "packet": runtime.packet_hashes()}
    try:
        result["public_freeze"] = runtime.public_freeze(args.freeze_commit)
        base, binary = runtime.verify_build(args.base, "base")
        if base["public_freeze"] != result["public_freeze"]:
            raise ValueError("build freeze differs from probe")
        builds = [("base", base, binary)]
        if args.variant:
            variant, vb = runtime.verify_build(args.variant, "variant")
            runtime.verify_pair(base, variant)
            builds.append(("variant", variant, vb))
        torch, result["gpu"] = runtime.gpu()
        values = runtime.tensors(torch, 64, 4096, 4096, torch.bfloat16)
        ref = runtime.reference(torch, values)
        result["build_receipts"] = [runtime.sha(args.base)]
        if args.variant:
            result["build_receipts"].append(runtime.sha(args.variant))
        for arm, receipt, library in builds:
            op = runtime.load(torch, receipt, library)
            op(*values, None)
            torch.cuda.synchronize()
            for _ in range(3):
                with torch.cuda.nvtx.range(f"labperf:{arm}:m64"):
                    op(*values, None)
                    torch.cuda.synchronize()
            if runtime.relative_error(torch, values[0], ref) >= 0.005:
                raise ValueError("identity launch numerical failure")
        result["status"] = "launches_completed_trace_pending"
    except Exception as exc:  # noqa: BLE001
        result["failure"] = type(exc).__name__
    finally:
        runtime.write(out / "probe.json", result)
    print(json.dumps({"status": result["status"]}))
    if result["status"] == "unscored":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
