"""Bounded default-path discovery; no kernel changes or performance scoring."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from preflight import check_public

PIN = "b0e21b308352587a6fd02f72722a2e815bfd62f0"
REVISION = "220b46e3b2180893580a4454f21f22d3ebb187d3"
LOADS = (1, 32)
INPUT_TOKENS = 128
OUTPUT_TOKENS = 64


def sha(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def launch_env(original):
    env = dict(original)
    # Refuse hidden configuration overrides instead of silently changing them.
    overrides = (
        "VLLM_DISABLED_KERNELS",
        "VLLM_USE_DEEP_GEMM",
        "VLLM_USE_V2_MODEL_RUNNER",
        "VLLM_BATCH_INVARIANT",
        "VLLM_ATTENTION_BACKEND",
    )
    if any(env.get(key) for key in overrides):
        raise ValueError("remove backend/runner/batch overrides before discovery")
    env["VLLM_SERVER_DEV_MODE"] = "1"
    return env


def command(python, model, out):
    return [
        "nsys",
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--cuda-graph-trace=node",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--force-overwrite=false",
        "--output",
        str(out / "serve"),
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
        "lab-discovery",
        "--tensor-parallel-size",
        "1",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "32",
        "--no-enable-prefix-caching",
        "--profiler-config",
        json.dumps(
            {
                "profiler": "cuda",
                "detailed_trace_annotation": True,
            }
        ),
    ]


def payload(index):
    return {
        "model": "lab-discovery",
        "prompt": [1000 + index] * INPUT_TOKENS,
        "max_tokens": OUTPUT_TOKENS,
        "temperature": 0,
        "ignore_eos": True,
    }


def validate_usage(usages, concurrency):
    if len(usages) != concurrency or any(
        u.get("prompt_tokens") != INPUT_TOKENS
        or u.get("completion_tokens") != OUTPUT_TOKENS
        for u in usages
    ):
        raise ValueError("fixed workload incomplete")


def http(path, deadline, data=None):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("session deadline exhausted")
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:8517" + path,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=min(120, remaining)) as reply:
        return json.loads(reply.read() or b"null")


def batch(concurrency, deadline):
    def one(index):
        # Generated text is discarded, never written into the evidence.
        return http("/v1/completions", deadline, payload(index))["usage"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        usages = list(pool.map(one, range(concurrency)))
    validate_usage(usages, concurrency)
    return usages


def save(out, result):
    (out / "collection.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--public-receipt", type=Path, required=True)
    parser.add_argument("--packet-root", type=Path, required=True)
    args = parser.parse_args()
    public = check_public(args.packet_root, args.freeze, args.public_receipt)
    if (
        sha(__file__)
        != public["files"]["experiments/native-kernel-discovery/collect.py"]
    ):
        raise ValueError("executed collector differs from public freeze")
    # Receipt is a prerequisite, not generated or self-approved by this collector.
    receipt = json.loads(args.preflight.read_text())
    required = (
        "wheel_verified",
        "source_verified",
        "installed_verified",
        "model_verified",
        "resources_verified",
        "gpu_idle_verified",
    )
    if (
        receipt.get("pin") != PIN
        or receipt.get("revision") != REVISION
        or any(receipt.get(key) is not True for key in required)
    ):
        raise ValueError("complete reviewed preflight receipt required")
    if receipt.get("freeze") != args.freeze or receipt.get(
        "public_receipt_sha256"
    ) != sha(args.public_receipt):
        raise ValueError("admission/public freeze binding differs")
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if receipt.get("boot_id") != boot:
        raise ValueError("admission belongs to another host boot")
    if Path(receipt["python"]).resolve() != Path(args.python).resolve() or (
        Path(receipt["model_dir"]).resolve() != args.model_dir.resolve()
    ):
        raise ValueError("preflight interpreter/model differs")
    env = launch_env(os.environ)
    start = receipt.get("session_started_monotonic")
    if type(start) not in (int, float) or not 0 <= time.monotonic() - start <= 300:
        raise ValueError(
            "fresh same-host session start required; includes GPU preflight"
        )
    args.out.mkdir(parents=True, exist_ok=False)
    # These stages consume the one session budget; reserve five minutes for cleanup.
    deadline = start + 55 * 60
    result = {
        "status": "insufficient_evidence",
        "pin": PIN,
        "revision": REVISION,
        "collector_sha256": sha(__file__),
        "preflight_sha256": sha(args.preflight),
        "freeze": args.freeze,
        "public_receipt_sha256": sha(args.public_receipt),
        "windows": [],
        "started_utc_ns": time.time_ns(),
    }
    process = None
    try:
        with (args.out / "server.log").open("w") as log:
            process = subprocess.Popen(
                command(args.python, args.model_dir, args.out),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            result["private_process_group"] = process.pid
            save(args.out, result)
            startup = min(deadline, start + 20 * 60)
            while time.monotonic() < startup:
                if process.poll() is not None:
                    raise RuntimeError("server exited during startup")
                try:
                    http("/health", min(startup, time.monotonic() + 2))
                    break
                except (OSError, TimeoutError):
                    time.sleep(1)
            else:
                raise TimeoutError("startup cap exhausted")
            result["resolved_config"] = http(
                "/server_info?config_format=json", deadline
            )
            # Two warm-ups per load, then one unprofiled reference batch per load.
            for level in LOADS:
                for _ in range(2):
                    batch(level, deadline)
                begin = time.monotonic_ns()
                batch(level, deadline)
                result.setdefault("unprofiled_batch_ns", {})[str(level)] = (
                    time.monotonic_ns() - begin
                )
            # Do not begin profiling late merely because startup finally succeeded.
            if deadline - time.monotonic() < 10 * 60:
                raise TimeoutError("less than ten minutes remain for collection")
            http("/start_profile", deadline, {})
            for repeat in range(2):
                for level in LOADS:
                    row = {
                        "concurrency": level,
                        "repeat": repeat,
                        "start_utc_ns": time.time_ns(),
                        "start_monotonic_ns": time.monotonic_ns(),
                    }
                    result["windows"].append(row)
                    save(args.out, result)
                    row["usage"] = batch(level, deadline)
                    row["end_monotonic_ns"] = time.monotonic_ns()
                    row["end_utc_ns"] = time.time_ns()
                    save(args.out, result)
            http("/stop_profile", deadline, {})
            result["status"] = "collected_review_pending"
    except Exception as exc:  # noqa: BLE001
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=15)
        result["elapsed_seconds"] = time.monotonic() - start
        result["artifacts_sha256"] = {
            p.name: sha(p) for p in args.out.glob("*.nsys-rep")
        }
        if not result["artifacts_sha256"]:
            result["status"] = "insufficient_evidence"
        save(args.out, result)
    print(json.dumps({"status": result["status"]}))
    return 0 if result["status"] == "collected_review_pending" else 1


if __name__ == "__main__":
    sys.exit(main())
