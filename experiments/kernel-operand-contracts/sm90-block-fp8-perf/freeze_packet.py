"""Generate/check byte bindings; a generated manifest is not a public freeze."""

import argparse
import json
from pathlib import Path

import build_perf
import protocol
import runtime

PACKET = Path(__file__).resolve().parent
ROOT = PACKET.parents[2]


def entries():
    paths = [
        p for p in PACKET.iterdir() if p.is_file() and p.name != "FREEZE_MANIFEST.json"
    ]
    paths += [
        ROOT / "docs/kernel/SM90_BLOCK_FP8_PERF_CHARTER.md",
        ROOT / "docs/kernel/SM90_BLOCK_FP8_PERF_CHARTER.zh-CN.md",
        ROOT / "tests/test_sm90_perf_preflight.py",
        ROOT / "tests/test_sm90_perf_tools.py",
    ]
    return {p.relative_to(ROOT).as_posix(): runtime.sha(p) for p in sorted(paths)}


def manifest_bytes(record):
    return (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_manifest(path, record):
    # Binary writes avoid Windows translating LF to CRLF before Git normalizes it.
    with path.open("xb") as file:
        file.write(manifest_bytes(record))


def check_manifest(path, record):
    if path.read_bytes() != manifest_bytes(record):
        raise ValueError("freeze bytes differ; canonical UTF-8/LF required")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    path = PACKET / "FREEZE_MANIFEST.json"
    record = {
        "status": "byte_manifest_not_public_freeze_confirmation",
        "source_pin": build_perf.PIN,
        "cutlass_pin": build_perf.CUTLASS_PIN,
        "model_revision": protocol.MODEL_REVISION,
        "files_sha256": entries(),
    }
    if args.check:
        check_manifest(path, record)
        print("MANIFEST_MATCHES")
    elif args.write:
        write_manifest(path, record)
        print("MANIFEST_WRITTEN_NOT_YET_PUBLIC")
    else:
        print(
            json.dumps(
                {"files": len(record["files_sha256"]), "status": record["status"]}
            )
        )


if __name__ == "__main__":
    main()
