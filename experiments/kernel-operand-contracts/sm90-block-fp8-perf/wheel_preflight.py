"""Retain/inspect the advertised parent wheel, without installing or importing Torch."""

import argparse
import email
import json
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import runtime
import serve_trace

MAX_BYTES = 8 * 1024**3
DOWNLOAD_SECONDS = 1800


def inspect_archive(index, wheel):
    row, url = serve_trace.select_wheel_index(json.loads(Path(index).read_text()))
    with zipfile.ZipFile(wheel) as archive:
        metadata = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        tags = [n for n in archive.namelist() if n.endswith(".dist-info/WHEEL")]
        if len(metadata) != 1 or len(tags) != 1:
            raise ValueError("one METADATA and WHEEL record required")
        info = email.message_from_bytes(archive.read(metadata[0]))
        serve_trace.validate_wheel_metadata(info, row)
        wheel_info = email.message_from_bytes(archive.read(tags[0]))
        allowed_tags = {"cp38-abi3-manylinux_2_28_x86_64", "cp38-abi3-linux_x86_64"}
        archive_tags = wheel_info.get_all("Tag", [])
        if not archive_tags or not set(archive_tags) <= allowed_tags:
            raise ValueError("archive ABI/platform tag differs")
        # At the pin, legacy _C is HIP-only; CUDA loads the stable extension.
        required = ("vllm/_C_stable_libtorch.abi3.so",)
        for name in required:
            if name not in archive.namelist():
                raise ValueError("required serving extension missing")
        return {
            "wheel_commit": serve_trace.PARENT,
            "wheel_index_url": serve_trace.WHEEL_INDEX,
            "wheel_index_path": str(Path(index).resolve()),
            "wheel_index_sha256": runtime.sha(index),
            "wheel_url": url,
            "wheel_path": str(Path(wheel).resolve()),
            "wheel_sha256": runtime.sha(wheel),
            "wheel_bytes": Path(wheel).stat().st_size,
            "version": info["Version"],
            "requires_python": info.get("Requires-Python"),
            "requires_dist": info.get_all("Requires-Dist", []),
            "archive_tags": archive_tags,
            "index_platform_tag": row["platform_tag"],
            "generic_linux_internal_tag": "cp38-abi3-linux_x86_64" in archive_tags,
            "extensions_present": [n for n in archive.namelist() if n.endswith(".so")],
            "limits": "archive only; not installed, CUDA/ABI/runtime unverified",
        }


def fetch(url, path, maximum):
    start = time.monotonic()
    print(
        f"Downloading {Path(path).name}; socket timeout=30s, "
        f"total cap={DOWNLOAD_SECONDS}s",
        file=sys.stderr,
        flush=True,
    )
    with (
        urllib.request.urlopen(url, timeout=30) as response,
        Path(path).open("xb") as file,
    ):
        length = response.headers.get("Content-Length")
        print(
            f"Response received; bytes={length or 'unknown'}",
            file=sys.stderr,
            flush=True,
        )
        if length and int(length) > maximum:
            raise ValueError("download exceeds fixed size cap")
        if length and shutil.disk_usage(Path(path).parent).free < int(length) + 1024**3:
            raise ValueError("insufficient disk space for retained archive")
        count = 0
        last_report = start
        reader = getattr(response, "read1", response.read)
        while chunk := reader(256 * 1024):
            if time.monotonic() - start > DOWNLOAD_SECONDS:
                raise TimeoutError("fixed total download window exhausted")
            count += len(chunk)
            if count > maximum:
                raise ValueError("download exceeds fixed size cap")
            file.write(chunk)
            now = time.monotonic()
            if now - last_report >= 2:
                file.flush()
                print(
                    f"Received {count / 1024**2:.2f} MiB; elapsed {now - start:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
                last_report = now
        if time.monotonic() - start > DOWNLOAD_SECONDS:
            raise TimeoutError("fixed total download window exhausted")
        if length and count != int(length):
            raise ValueError("download count differs from Content-Length")
        print(
            f"Downloaded {count} bytes in {time.monotonic() - start:.1f}s",
            file=sys.stderr,
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="new private output directory")
    parser.add_argument("--index", help="existing original index, offline mode")
    parser.add_argument("--wheel", help="existing archive, offline mode")
    args = parser.parse_args()
    if bool(args.index) != bool(args.wheel):
        parser.error("offline mode requires both --index and --wheel")
    out = runtime.fresh(args.out)
    result = {"status": "unscored"}
    try:
        if args.index:
            index, wheel = Path(args.index), Path(args.wheel)
        else:
            index = out / "metadata.json"
            fetch(serve_trace.WHEEL_INDEX, index, 1024**2)
            row, url = serve_trace.select_wheel_index(json.loads(index.read_text()))
            wheel = out / row["filename"]
            fetch(url, wheel, MAX_BYTES)
        result.update(inspect_archive(index, wheel))
        result["status"] = "archive_prepared_runtime_unverified"
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - retain first failed download/inspection
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        runtime.write(out / "wheel_receipt.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "version", "wheel_bytes", "wheel_sha256", "failure")
                if k in result
            }
        )
    )
    if result["status"] == "unscored":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
