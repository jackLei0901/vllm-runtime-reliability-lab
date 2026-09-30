"""Fixed workload collector. Collection is not proof of replay shape attribution."""

from __future__ import annotations

import argparse
import concurrent.futures
import email
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import build_perf
import model_preflight
import runtime

DISABLED = (
    "FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel"
)
CONCURRENCIES = (1, 16, 63, 64)
PARENT = "4f1451679088e5832bce0965a254295c854aaa09"
WHEEL_INDEX = f"https://wheels.vllm.ai/{PARENT}/cu130/vllm/metadata.json"


def select_wheel_index(data):
    rows = [
        row
        for row in data
        if row.get("package_name") == "vllm"
        and row.get("platform_tag") == "manylinux_2_28_x86_64"
        and row.get("python_tag") == "cp38"
        and row.get("abi_tag") == "abi3"
        and PARENT[:7] in row.get("version", "")
    ]
    if len(rows) != 1:
        raise ValueError("one compatible parent wheel index entry required")
    row = rows[0]
    url = urllib.parse.urljoin(WHEEL_INDEX, row["path"])
    parsed = urllib.parse.urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "wheels.vllm.ai"
        or not parsed.path.startswith(f"/{PARENT}/")
    ):
        raise ValueError("wheel index URL escaped the exact parent")
    if urllib.parse.unquote(Path(parsed.path).name) != row["filename"]:
        raise ValueError("index URL and filename differ")
    return row, url


def validate_wheel_metadata(info, row):
    if info.get("Name", "").lower() != "vllm" or info.get("Version") != row["version"]:
        raise ValueError("wheel package/version differs from index")
    requirements = [
        r.replace(" ", "")
        for r in info.get_all("Requires-Dist", [])
        if r.lower().startswith("torch")
        and not r.lower().startswith(("torchaudio", "torchvision", "torchcodec"))
    ]
    if requirements not in (["torch==2.13.0"], ["torch(==2.13.0)"]):
        raise ValueError("wheel must require torch 2.13.0 without ambiguity")


def verify_provenance(provenance, installed, src):
    if (
        provenance.get("source_pin") != build_perf.PIN
        or provenance.get("extensions_sha256") != installed["extensions_sha256"]
    ):
        raise ValueError("serving source/binary provenance differs")
    if not provenance.get("build_command") or not provenance.get("build_log_sha256"):
        raise ValueError("retained install/build command and log digest required")
    kind = provenance.get("kind", "full_source_build")
    if kind == "full_source_build":
        return kind
    if kind != "precompiled_parent":
        raise ValueError("unknown serving provenance route")
    if runtime.sha(provenance["build_log"]) != provenance["build_log_sha256"]:
        raise ValueError("retained precompiled install log differs")
    if provenance.get("wheel_commit") != PARENT:
        raise ValueError("only the explicit parent wheel is admitted")
    index_url = WHEEL_INDEX
    if provenance.get("wheel_index_url") != index_url:
        raise ValueError("explicit parent cu130 index required")
    index = Path(provenance["wheel_index_path"])
    if runtime.sha(index) != provenance["wheel_index_sha256"]:
        raise ValueError("retained wheel index differs")
    row, advertised_url = select_wheel_index(json.loads(index.read_text()))
    if provenance.get("wheel_url") != advertised_url:
        raise ValueError("wheel URL must resolve from the retained cu130 index")
    wheel = Path(provenance["wheel_path"])
    if runtime.sha(wheel) != provenance["wheel_sha256"]:
        raise ValueError("retained wheel digest differs")
    if build_perf.git(src, "rev-parse", "HEAD^") != PARENT:
        raise ValueError("source parent differs")
    changes = build_perf.git(
        src, "diff", "--name-only", PARENT, build_perf.PIN, "--", "csrc"
    ).splitlines()
    if changes != [build_perf.ROOT + "scaled_mm_entry.cu"]:
        raise ValueError("parent C++ differences exceed the reviewed entry checks")
    with zipfile.ZipFile(wheel) as archive:
        metadata = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise ValueError("one wheel dependency metadata record required")
        info = email.message_from_bytes(archive.read(metadata[0]))
        validate_wheel_metadata(info, row)
        for relative, expected in installed["extensions_sha256"].items():
            member = "vllm/" + relative
            with archive.open(member) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                    raise ValueError("installed extension differs from retained wheel")
    return kind


def command(python, model, directory, mode):
    profiler = {"profiler": "cuda", "detailed_trace_annotation": True}
    if mode == "shapes":
        profiler.update(
            profiler="torch",
            torch_profiler_dir=str(Path(directory).resolve()),
            torch_profiler_record_shapes=True,
            capture_torch_profiler=True,
            torch_profiler_with_stack=False,
            torch_profiler_use_gzip=True,
        )
    serve = [
        python,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        "127.0.0.1",
        "--port",
        "8517",
        "--model",
        str(model),
        "--served-model-name",
        "lab-perf",
        "--tensor-parallel-size",
        "1",
        "--distributed-executor-backend",
        "uni",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "64",
        "--max-num-batched-tokens",
        "8192",
        "--no-enable-prefix-caching",
        "--performance-mode",
        "balanced",
        "--profiler-config",
        json.dumps(profiler),
    ]
    if mode == "nsys":
        return [
            "nsys",
            "profile",
            "--trace=cuda,nvtx,osrt",
            "--cuda-graph-trace=node",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--force-overwrite=false",
            "--output",
            str(Path(directory) / "serve"),
            *serve,
        ]
    return serve


def request(path, data=None, timeout=120):
    url = "http://127.0.0.1:8517" + path
    encoded = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(
        url, data=encoded, headers={"Content-Type": "application/json"}
    )
    # Local requests must not inherit external HTTP proxies.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as response:
        return response.read()


def batch(concurrency):
    def one(index):
        body = json.loads(
            request(
                "/v1/completions",
                {
                    "model": "lab-perf",
                    "prompt": [1000 + index] * 128,
                    "max_tokens": 64,
                    "temperature": 0,
                    "ignore_eos": True,
                },
            )
        )
        if body["usage"]["completion_tokens"] != 64:
            raise ValueError("fixed output length not completed")
        # Discard generated text, preserve only workload counts.
        return body["usage"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(one, range(concurrency)))


def installed_identity(src):
    build_perf.validate_source(src, "base")
    spec = importlib.util.find_spec("vllm")
    if not spec or not spec.origin:
        raise ValueError("vLLM not installed in this interpreter")
    installed = Path(spec.origin).parent
    hashes = {}
    for name in build_perf.git(src, "ls-files", "vllm").splitlines():
        if name.endswith(".py"):
            target = installed / Path(name).relative_to("vllm")
            if runtime.sha(src / name) != runtime.sha(target):
                raise ValueError(f"installed Python differs: {name}")
            hashes[name] = runtime.sha(target)
    extensions = {
        str(p.relative_to(installed)): runtime.sha(p) for p in installed.rglob("*.so")
    }
    if not extensions:
        raise ValueError("no serving extension identities")
    return {
        "source_pin": build_perf.PIN,
        "python_sha256": hashes,
        "extensions_sha256": extensions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--vllm-src", required=True)
    parser.add_argument(
        "--serving-build", help="retained full build provenance, not minimal build"
    )
    parser.add_argument("--out")
    parser.add_argument("--mode", choices=["nsys", "shapes"], default="nsys")
    parser.add_argument(
        "--configuration", choices=["forced", "default"], default="forced"
    )
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--freeze-commit")
    parser.add_argument(
        "--deadline-utc",
        help="same absolute Stage 0 deadline for both collector passes",
    )
    args = parser.parse_args()
    if args.configuration == "default" and args.mode == "shapes":
        parser.error("default shape-profile pass is out of scope")
    if args.plan:
        print(
            json.dumps(
                {
                    "configurations": [args.configuration],
                    "concurrencies": CONCURRENCIES,
                    "command": command(
                        sys.executable, args.model_dir, "PRIVATE_WORK", args.mode
                    ),
                    "requires_serving_build_provenance": True,
                }
            )
        )
        return
    if not args.out or not args.serving_build or not args.deadline_utc:
        parser.error("new --out, serving provenance and shared --deadline-utc required")
    out = runtime.fresh(args.out)
    result = {
        "status": "unscored",
        "packet": runtime.packet_hashes(),
        "mode": args.mode,
        "runs": [],
    }
    try:
        result["public_freeze"] = runtime.public_freeze(args.freeze_commit)
        result["model"] = model_preflight.inspect_model(args.model_dir)
        result["installed"] = installed_identity(Path(args.vllm_src).resolve())
        provenance = json.loads(Path(args.serving_build).read_text())
        result["serving_route"] = verify_provenance(
            provenance, result["installed"], Path(args.vllm_src)
        )
        result["serving_build_sha256"] = runtime.sha(args.serving_build)
        torch, result["gpu"] = runtime.gpu()
        if result["serving_route"] == "precompiled_parent" and (
            result["gpu"]["torch"] != "2.13.0+cu130"
            or result["gpu"]["torch_cuda"] != "13.0"
        ):
            raise ValueError("parent cu130 wheel requires the reviewed Torch/CUDA pair")
        del torch
        end = datetime.fromisoformat(args.deadline_utc.replace("Z", "+00:00"))
        remaining = (end - datetime.now(timezone.utc)).total_seconds()
        if not 0 < remaining <= 1500:
            raise ValueError("Stage 0 remaining window must be within 25 minutes")
        deadline = time.monotonic() + remaining
        for configuration in (args.configuration,):
            directory = out / configuration
            directory.mkdir()
            env = os.environ.copy()
            for key in ("VLLM_DISABLED_KERNELS", "VLLM_USE_DEEP_GEMM"):
                env.pop(key, None)
            if configuration == "forced":
                env.update(VLLM_DISABLED_KERNELS=DISABLED, VLLM_USE_DEEP_GEMM="0")
            row = {"configuration": configuration, "status": "unscored", "loads": []}
            result["runs"].append(row)
            with (directory / "server.log").open("w") as log:
                process = subprocess.Popen(
                    command(sys.executable, args.model_dir, directory, args.mode),
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    startup = min(deadline, time.monotonic() + 600)
                    while time.monotonic() < startup:
                        if process.poll() is not None:
                            raise ValueError("server exited during startup")
                        try:
                            request("/health", timeout=2)
                            break
                        except OSError:
                            time.sleep(1)
                    else:
                        raise TimeoutError("fixed startup window exhausted")
                    info = json.loads(
                        request("/server_info?config_format=json", timeout=30)
                    )
                    row["resolved_config"] = info["vllm_config"]
                    capture = row["resolved_config"]["compilation_config"][
                        "cudagraph_capture_sizes"
                    ]
                    if configuration == "forced" and not {24, 40, 56, 64} <= set(
                        capture
                    ):
                        raise ValueError("required capture sizes not resolved")
                    if configuration == "forced" and 63 in capture:
                        raise ValueError("M63 is unexpectedly captured without padding")
                    runtime.write(out / "collection.json", result)
                    batch(1)  # warm-up, outside active profiling
                    request("/start_profile", {}, timeout=30)
                    for concurrency in CONCURRENCIES:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("serving stage exhausted")
                        row["loads"].append(
                            {"concurrency": concurrency, "usage": batch(concurrency)}
                        )
                        runtime.write(out / "collection.json", result)
                    request("/stop_profile", {}, timeout=120)
                    row["status"] = "collected"
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
            row["log_sha256"] = runtime.sha(directory / "server.log")
            pattern = "*.nsys-rep" if args.mode == "nsys" else "*.json.gz"
            artifacts = list(directory.rglob(pattern))
            if not artifacts:
                raise ValueError("profiler did not produce trace artifacts")
            row["artifact_sha256"] = {
                str(p.relative_to(directory)): runtime.sha(p) for p in artifacts
            }
        result["status"] = "collected_review_pending"
    except Exception as exc:  # noqa: BLE001
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        runtime.write(out / "collection.json", result)
    print(json.dumps({"status": result["status"]}))
    if result["status"] == "unscored":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
