"""Sequential builds in one non-renewable 35-minute window; no retries."""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import build_perf
import runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-src", type=Path, required=True)
    parser.add_argument("--variant-src", type=Path, required=True)
    parser.add_argument("--cutlass-src", type=Path, required=True)
    parser.add_argument("--cmake-reference", type=Path, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--freeze-commit", required=True)
    args = parser.parse_args()
    out = runtime.fresh(args.out)
    result = {"status": "unscored", "order": "base then variant", "arms": []}
    try:
        result["public_freeze"] = runtime.public_freeze(args.freeze_commit)
        deadline = datetime.now(timezone.utc) + timedelta(seconds=2100)
        result["shared_deadline_utc"] = deadline.isoformat()
        runtime.write(out / "build_pair.json", result)
        print(json.dumps({"shared_deadline_utc": deadline.isoformat()}), flush=True)
        for arm, src in (("base", args.base_src), ("variant", args.variant_src)):
            timeout = build_perf.remaining_timeout(1200, deadline.isoformat())
            command = [
                sys.executable,
                str(Path(__file__).with_name("build_perf.py")),
                "--vllm-src",
                str(src),
                "--cutlass-src",
                str(args.cutlass_src),
                "--cmake-reference",
                str(args.cmake_reference),
                "--arm",
                arm,
                "--namespace",
                f"lab_sm90_perf_{arm}",
                "--out",
                str(out / arm),
                "--freeze-commit",
                args.freeze_commit,
                "--timeout-seconds",
                str(timeout),
                "--deadline-utc",
                deadline.isoformat(),
            ]
            record = {"arm": arm, "timeout_seconds": timeout, "command": command}
            result["arms"].append(record)
            runtime.write(out / "build_pair.json", result)
            record["returncode"] = subprocess.call(command)
            runtime.write(out / "build_pair.json", result)
            if record["returncode"]:
                raise ValueError(f"{arm} build failed; no retry")
            build_perf.remaining_timeout(1200, deadline.isoformat())
        result["status"] = "built_not_validated"
    except Exception as exc:  # noqa: BLE001 - preserve failed shared window
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        runtime.write(out / "build_pair.json", result)
    if result["status"] != "built_not_validated":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
