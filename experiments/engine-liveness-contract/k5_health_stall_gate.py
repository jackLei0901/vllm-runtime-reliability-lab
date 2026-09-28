"""One-cell, bounded TP=1/TP=2 EngineCore health-stall experiment.

Run separately for each arm under the frozen K5 protocol. Raw server logs and
PID-bearing hold markers stay in an owner-only output directory. Stdout carries
only a closed summary; a missing witness is never scored as a healthy engine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import stat
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

TIMEOUT_S = 30
HOLD_S = 45
POLL_S = 2
MODEL_ALIAS = "llr-k5"
EXPECTED_VERSION = "gc8602c79"
TIMEOUT_MARKER = "RPC call to execute_model timed out."
SOURCE_SHA256 = {
    "envs.py": "c36ad82467e4fa05ceea3e613e07283118da14bf2c49346abc1702d2116db1ce",
    "config/parallel.py": "5ee765f980d8e371314caa38a27c5a21ab90f3865e697707d48e608ca4d17f3d",
    "entrypoints/serve/instrumentator/health.py": "bac5cf6a8c2f8ce7fde69bca14ec699cfe1401c610e06f01a0ffece65adffe3e",
    "v1/engine/async_llm.py": "81a0cae6d5da22140f509a59d6c6bb8fc6ee1572da2a2cd793b5830222d18bcc",
    "v1/executor/multiproc_executor.py": "ceb3477473bdb1de36e01687e0b2083ad239d42b1ddc2ec8cec16ba2c477efa6",
    "v1/executor/uniproc_executor.py": "4625c8e59aef1482600b02204701d348d7f716427581d3cf23b06bff9b056abe",
    "v1/worker/gpu_worker.py": "50134ca3a147f5470e556437cbd40a13ea05b098efbac6553d5ea2f3e5f7ff57",
}


class _AbortCell(Exception):
    """A known apparatus failure; the server must still be stopped first."""


class _Interrupted(Exception):
    """A shell timeout or operator stop that permits bounded child cleanup."""


def _interrupt(signum: int, frame: Any) -> None:
    raise _Interrupted


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _http(url: str, *, payload: dict | None = None, timeout: float = 2) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body else {},
        method="POST" if body else "GET",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return {"status": response.status, "error": None,
                    "body": response.read(65536)}
    except urllib.error.HTTPError as exc:
        return {"status": exc.code, "error": None, "body": exc.read(65536)}
    except (OSError, TimeoutError) as exc:
        return {"status": None, "error": type(exc).__name__, "body": b""}


def _completion(base: str, *, timeout: float) -> dict:
    result = _http(
        f"{base}/v1/completions",
        payload={"model": MODEL_ALIAS, "prompt": "Hello", "max_tokens": 16,
                 "temperature": 0, "stream": False},
        timeout=timeout,
    )
    try:
        parsed = json.loads(result["body"])
        valid = (isinstance(parsed.get("choices"), list)
                 and len(parsed["choices"]) == 1)
    except (ValueError, AttributeError, TypeError):
        valid = False
    return {"status": result["status"], "error": result["error"],
            "valid_response": valid}


def _proc_stat(pid: int) -> tuple[int, int]:
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
    return int(fields[1]), int(fields[19])  # PPID (field 4), starttime (22)


def _descends_from(pid: int, root_pid: int, root_ticks: int) -> bool:
    seen: set[int] = set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        try:
            parent, ticks = _proc_stat(pid)
        except (OSError, ValueError, IndexError):
            return False
        if pid == root_pid:
            return ticks == root_ticks
        pid = parent
    return False


def _read_marker(directory: Path, name: str) -> dict | None:
    path = directory / name
    if not path.is_file() or path.stat().st_size > 2048:
        return None
    try:
        return json.loads(path.read_text(encoding="ascii"))
    except (OSError, ValueError):
        return None


def _check_marker(marker: dict | None, *, tp: int, root_pid: int,
                  root_ticks: int) -> bool:
    if not isinstance(marker, dict) or set(marker) != {
        "schema", "pid", "start_ticks", "rank", "tp",
        "scheduled_tokens_positive", "monotonic_ns"
    }:
        return False
    if (marker["schema"] != "llr-k5-worker-hold-v1"
            or type(marker["pid"]) is not int
            or type(marker["start_ticks"]) is not int
            or type(marker["monotonic_ns"]) is not int
            or marker["rank"] != 0 or marker["tp"] != tp
            or marker["scheduled_tokens_positive"] is not True):
        return False
    try:
        _, current_ticks = _proc_stat(marker["pid"])
    except (OSError, ValueError, IndexError):
        return False
    return (current_ticks == marker["start_ticks"]
            and _descends_from(marker["pid"], root_pid, root_ticks))


def score(mode: str, *, baseline_ok: bool, marker_ok: bool,
          samples: list[dict], completion: dict, released: bool,
          timeout_logged: bool, post_health: int | None) -> str:
    """Closed result vocabulary. Samples are seconds since hold entry."""
    if not baseline_ok or not marker_ok:
        return "unscored_apparatus"
    early = [row for row in samples if 2 <= row["t"] <= 26]
    late = [row for row in samples if 32 <= row["t"] <= 44]
    if len(early) < 5 or len(late) < 2:
        return "unscored_window"
    if mode == "tp1":
        if any(row["status"] != 200 for row in samples if row["t"] <= HOLD_S):
            return "contradicted_tp1_unhealthy"
        if (released and completion.get("status") == 200
                and completion.get("valid_response") is True
                and post_health == 200 and not timeout_logged):
            return "tp1_no_timeout_health_200"
        return "unscored_recovery"
    if mode == "tp2":
        if any(row["status"] != 200 for row in early):
            return "contradicted_early_unhealthy"
        if any(row["status"] == 503 for row in late) and timeout_logged:
            return "tp2_timeout_health_503"
        if all(row["status"] == 200 for row in late):
            return "contradicted_no_timeout"
        return "unscored_timeout_attribution"
    raise ValueError("unknown K5 mode")


def _preflight(vllm: Path, model: Path, output: Path, port: int) -> dict:
    if os.name != "posix" or not Path("/proc/self/stat").is_file():
        raise ValueError("K5 requires Linux /proc")
    if not vllm.is_file() or not os.access(vllm, os.X_OK):
        raise ValueError("vLLM executable unavailable")
    if not (model / "config.json").is_file():
        raise ValueError("cached model config unavailable")
    if not output.is_absolute() or output.exists():
        raise ValueError("output must be a fresh absolute path")
    if not output.parent.is_dir():
        raise ValueError("output parent unavailable")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))
    gpu_output = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15, check=True,
    ).stdout.strip().splitlines()
    if len(gpu_output) != 2:
        raise ValueError("K5 requires exactly two visible GPUs")
    for row in gpu_output:
        name, free_mib = row.rsplit(",", 1)
        if "RTX 4090" not in name or int(free_mib.strip()) < 14000:
            raise ValueError("K5 GPU identity or free memory differs")
    plugin_dir = Path(__file__).with_name("k5-worker-hold-plugin")
    plugin = plugin_dir / "llr_k5_worker_hold.py"
    if not plugin.is_file():
        raise ValueError("K5 plugin unavailable")
    version_output = subprocess.run(
        [str(vllm), "--version"], capture_output=True, text=True,
        timeout=45, check=True,
    ).stdout.strip()
    version = version_output.splitlines()[-1] if version_output else ""
    if not re.fullmatch(r"[A-Za-z0-9_.+\-]+", version):
        raise ValueError("vLLM version output is not a single version token")
    if EXPECTED_VERSION not in version:
        raise ValueError("vLLM build is not the preregistered revision")
    python = vllm.with_name("python")
    if not python.is_file():
        raise ValueError("matching vLLM Python executable is unavailable")
    ninja_search_path = str(vllm.parent) + os.pathsep + os.environ.get("PATH", "")
    ninja = shutil.which("ninja", path=ninja_search_path)
    if ninja is None:
        raise ValueError("ninja unavailable in the vLLM launch environment")
    source_output = subprocess.run(
        [str(python), "-c", "import vllm; print(vllm.__file__)"],
        capture_output=True, text=True, timeout=60, check=True,
    ).stdout.strip()
    if not source_output:
        raise ValueError("vLLM package root is unavailable")
    source_root = Path(source_output.splitlines()[-1]).resolve().parent
    source_hashes = {
        name: sha256_file(source_root / name) for name in SOURCE_SHA256
    }
    if source_hashes != SOURCE_SHA256:
        raise ValueError("installed Python sources differ from the pinned commit")
    return {"vllm_version": version, "plugin_sha256": sha256_file(plugin),
            "runner_sha256": sha256_file(Path(__file__)),
            "ninja_sha256": sha256_file(Path(ninja)),
            "model_config_sha256": sha256_file(model / "config.json"),
            "source_sha256": source_hashes,
            "gpu_count": len(gpu_output),
            "plugin_dir": plugin_dir}


def _stop_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.getpgid(process.pid) != process.pid:
            return
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        # The session was created by this runner; its group is the exact target.
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--mode", required=True, choices=("tp1", "tp2"))
    args = parser.parse_args()
    tp = 1 if args.mode == "tp1" else 2
    try:
        identity = _preflight(args.vllm, args.model, args.output_dir, args.port)
        args.output_dir.mkdir(mode=0o700)
        if stat.S_IMODE(args.output_dir.stat().st_mode) != 0o700:
            raise ValueError("output directory is not owner-only")
    except Exception as exc:
        print("K5_RESULT " + json.dumps({
            "schema": "llr-k5-health-stall-v1", "mode": args.mode,
            "result": "unscored_preflight", "error_type": type(exc).__name__,
        }, sort_keys=True), flush=True)
        return 2
    log_path = args.output_dir / "server.log"
    command = [
        str(args.vllm), "serve", str(args.model),
        "--served-model-name", MODEL_ALIAS, "--host", "127.0.0.1",
        "--port", str(args.port), "--tensor-parallel-size", str(tp),
        "--distributed-executor-backend", "uni" if tp == 1 else "mp",
        "--max-model-len", "512", "--max-num-seqs", "1",
        "--gpu-memory-utilization", "0.65", "--enforce-eager",
        "--disable-custom-all-reduce",
    ]
    environment = os.environ.copy()
    environment.update({
        "PATH": str(args.vllm.parent) + os.pathsep + environment.get("PATH", ""),
        "CUDA_VISIBLE_DEVICES": "0" if tp == 1 else "0,1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "VLLM_NO_USAGE_STATS": "1", "OMP_NUM_THREADS": "1",
        "VLLM_KEEP_ALIVE_ON_ENGINE_DEATH": "1",
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": str(TIMEOUT_S),
        "VLLM_PLUGINS": "llr_k5_worker_hold",
        "LLR_K5_PRIVATE_DIR": str(args.output_dir),
        "LLR_K5_TP_SIZE": str(tp),
        "LLR_K5_HOLD_SECONDS": str(HOLD_S),
        "PYTHONPATH": str(identity["plugin_dir"]) + os.pathsep
                      + environment.get("PYTHONPATH", ""),
    })
    base = f"http://127.0.0.1:{args.port}"
    summary: dict[str, Any] = {
        "schema": "llr-k5-health-stall-v1", "mode": args.mode,
        "tp": tp, "executor": "uni" if tp == 1 else "mp",
        "timeout_s": TIMEOUT_S, "hold_s": HOLD_S,
        "vllm_version": identity["vllm_version"],
        "plugin_sha256": identity["plugin_sha256"],
        "runner_sha256": identity["runner_sha256"],
        "ninja_sha256": identity["ninja_sha256"],
        "model_config_sha256": identity["model_config_sha256"],
        "source_sha256": identity["source_sha256"],
        "gpu_count": identity["gpu_count"],
        "result": "unscored_apparatus",
    }
    signal.signal(signal.SIGTERM, _interrupt)
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, env=environment, start_new_session=True,
        )
        try:
            root_ticks = _proc_stat(process.pid)[1]
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline and process.poll() is None:
                if _http(f"{base}/health")["status"] == 200:
                    break
                time.sleep(1)
            else:
                summary["result"] = "unscored_server_not_ready"
                raise _AbortCell

            baseline = _completion(base, timeout=60)
            baseline_health = _http(f"{base}/health")["status"]
            baseline_ok = (baseline["status"] == baseline_health == 200
                           and baseline["valid_response"] is True)
            summary["baseline_ok"] = baseline_ok
            if not baseline_ok:
                summary["result"] = "unscored_baseline"
                raise _AbortCell

            arm_fd = os.open(args.output_dir / "arm",
                             os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(arm_fd)
            completion: dict[str, Any] = {}

            def send_request() -> None:
                completion.update(_completion(base, timeout=75))

            request_thread = threading.Thread(target=send_request, daemon=True)
            request_thread.start()
            entered = None
            marker_deadline = time.monotonic() + 20
            while time.monotonic() < marker_deadline and process.poll() is None:
                entered = _read_marker(args.output_dir, "entered.json")
                if entered is not None or not request_thread.is_alive():
                    break
                time.sleep(0.1)
            marker_ok = _check_marker(
                entered, tp=tp, root_pid=process.pid, root_ticks=root_ticks
            )
            summary["marker_ok"] = marker_ok
            if not marker_ok:
                summary["result"] = "unscored_hold_not_entered_or_unbound"
                raise _AbortCell

            samples: list[dict] = []
            next_sample = time.monotonic()
            while True:
                elapsed = (time.monotonic_ns() - entered["monotonic_ns"]) / 1e9
                if elapsed > HOLD_S + 4:
                    break
                wait = next_sample - time.monotonic()
                if wait > 0:
                    time.sleep(min(wait, 0.2))
                    continue
                health = _http(f"{base}/health", timeout=1.5)
                samples.append({"t": round(elapsed, 2),
                                "status": health["status"],
                                "error": health["error"]})
                next_sample += POLL_S
            request_thread.join(timeout=10)
            released = _read_marker(args.output_dir, "released.json") is not None
            post_health = _http(f"{base}/health")["status"]
            summary.update({
                "released": released, "completion": completion,
                "post_health": post_health,
                "health_samples": samples,
            })
        except _AbortCell:
            pass
        except _Interrupted:
            summary["result"] = "unscored_interrupted"
        except Exception as exc:
            summary["result"] = "unscored_runner_error"
            summary["error_type"] = type(exc).__name__
        finally:
            _stop_group(process)
    if summary["result"] != "unscored_apparatus":
        return _finish(summary, log_path)
    log_bytes = log_path.read_bytes()
    summary["timeout_logged"] = TIMEOUT_MARKER.encode() in log_bytes
    summary["server_log_sha256"] = hashlib.sha256(log_bytes).hexdigest()
    summary["result"] = score(
        args.mode, baseline_ok=baseline_ok, marker_ok=marker_ok,
        samples=samples, completion=completion, released=released,
        timeout_logged=summary["timeout_logged"], post_health=post_health,
    )
    return _finish(summary, log_path)


def _finish(summary: dict, log_path: Path) -> int:
    if log_path.is_file() and "server_log_sha256" not in summary:
        summary["server_log_sha256"] = sha256_file(log_path)
    result_path = log_path.parent / "result.json"
    try:
        encoded = (json.dumps(summary, sort_keys=True) + "\n").encode("utf-8")
        fd = os.open(result_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        summary["result"] = "unscored_result_persistence"
        summary["error_type"] = type(exc).__name__
    print("K5_RESULT " + json.dumps(summary, sort_keys=True), flush=True)
    return 0 if summary["result"] in {
        "tp1_no_timeout_health_200", "tp2_timeout_health_503"
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
