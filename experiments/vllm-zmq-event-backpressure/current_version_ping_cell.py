"""One bounded, private current-wheel IPC-ping/backpressure test cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def start_ticks(pid: int) -> int | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        return int(stat[stat.rfind(")") + 2 :].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def health(url: str, timeout: float) -> dict[str, object]:
    started = time.monotonic()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            result: dict[str, object] = {"kind": "http", "status": response.status}
    except urllib.error.HTTPError as error:
        result = {"kind": "http", "status": error.code}
    except (OSError, urllib.error.URLError) as error:
        result = {"kind": "transport_error", "error_type": type(error).__name__}
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return result


class Stream:
    def __init__(self) -> None:
        self.started = time.monotonic()
        self.progress: list[float] = []
        self.usage: dict[str, int] | None = None
        self.error: str | None = None
        self.complete = False
        self.finished = threading.Event()
        self.headers_received = threading.Event()
        self.lock = threading.Lock()

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            return {
                "progress_offsets_seconds": self.progress.copy(),
                "usage": self.usage,
                "error": self.error,
                "complete": self.complete,
            }

    def stalled(self) -> bool:
        with self.lock:
            return (
                not self.complete
                and self.error is None
                and len(self.progress) >= 2
                and time.monotonic() - self.started - self.progress[-1] >= 10
            )


def consume_stream(url: str, stream: Stream, prompt_words: int, max_tokens: int) -> None:
    prompt = " ".join(f"word{i}" for i in range(prompt_words))
    payload = json.dumps(
        {
            "model": "dfx-poc",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            stream.headers_received.set()
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                body = line.removeprefix("data:").strip()
                if body == "[DONE]":
                    with stream.lock:
                        stream.complete = True
                    return
                event = json.loads(body)
                if any(choice.get("text") for choice in event.get("choices", [])):
                    with stream.lock:
                        stream.progress.append(time.monotonic() - stream.started)
                usage = event.get("usage")
                if isinstance(usage, dict):
                    with stream.lock:
                        stream.usage = {
                            key: value
                            for key, value in usage.items()
                            if key in {"prompt_tokens", "completion_tokens"}
                            and isinstance(value, int)
                        }
        with stream.lock:
            stream.error = "stream_ended_without_done"
    except (OSError, UnicodeError, ValueError, TypeError) as error:
        with stream.lock:
            stream.error = type(error).__name__
    finally:
        stream.finished.set()


def stop_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ping", choices=("on", "off"), required=True)
    parser.add_argument("--trigger", choices=("pause", "control"), required=True)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--prompt-words", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--health-after-headers", action="store_true")
    parser.add_argument("--source-equivalent-root", type=Path)
    args = parser.parse_args()
    if not args.vllm.is_file() or not args.model.is_dir() or not args.packages.is_dir():
        parser.error("vLLM, model, or test package directory is missing")
    if args.source_equivalent_root and not (
        args.source_equivalent_root / "vllm" / "v1" / "engine" / "core_client.py"
    ).is_file():
        parser.error("source-equivalent vLLM overlay is missing")
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    control = args.output_dir / "control"
    control.mkdir(mode=0o700)
    if args.trigger == "control":
        (control / "release").touch()
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "HF_HUB_OFFLINE": "1",
            "VLLM_NO_USAGE_STATS": "1",
            "PATH": f"{args.vllm.parent}:{environment.get('PATH', '')}",
            "PYTHONPATH": ":".join(
                str(path)
                for path in (
                    [args.source_equivalent_root.resolve()]
                    if args.source_equivalent_root else []
                ) + [args.packages.resolve(), environment.get("PYTHONPATH", "")]
            ),
            "VLLM_PLUGINS": (
                "dfx_stage1_backpressure" if args.source_equivalent_root
                else "dfx_stage1_backpressure,dfx_health_ping_poc"
            ),
            "DFX_STAGE1_ENABLE": "1",
            "DFX_STAGE1_CONTROL_DIR": str(control.resolve()),
            "DFX_STAGE1_OBSERVER_PID": str(os.getpid()),
            "DFX_HEALTH_PING_ENABLE": "1" if args.ping == "on" else "0",
            "DFX_HEALTH_PING_TIMEOUT_SECONDS": "60",
            "VLLM_HEALTH_CHECK_TIMEOUT": "60" if args.ping == "on" else "0",
        }
    )
    command = [
        str(args.vllm), "serve", str(args.model),
        "--served-model-name", "dfx-poc",
        "--host", "127.0.0.1", "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", "0.70",
        "--enforce-eager",
        "--kv-events-config",
        json.dumps({
            "enable_kv_cache_events": True,
            "publisher": "zmq",
            "endpoint": "tcp://127.0.0.1:5557",
            "max_queue_size": 1,
        }),
    ]
    summary: dict[str, object] = {
        "ping": args.ping,
        "trigger": args.trigger,
        "source_mode": (
            "pr36451_python_source_equivalent" if args.source_equivalent_root
            else "test_plugin"
        ),
        "result": "undetermined",
    }
    if args.source_equivalent_root:
        summary["source_sha256"] = {
            name: sha256(args.source_equivalent_root / "vllm" / name)
            for name in (
                "envs.py",
                "v1/engine/async_llm.py",
                "v1/engine/core.py",
                "v1/engine/core_client.py",
            )
        }
    log_path = args.output_dir / "server.log"
    descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, env=environment, start_new_session=True,
        )
        try:
            base = f"http://127.0.0.1:{args.port}"
            deadline = time.monotonic() + 240
            ready_path: Path | None = None
            while time.monotonic() < deadline and process.poll() is None:
                paths = list(control.glob("ready-*.json"))
                errors = list(control.glob("hook-error-*.json"))
                if errors or len(paths) > 1:
                    summary["error"] = "stage1_hook_error_or_duplicate"
                    return 1
                if len(paths) == 1 and health(f"{base}/health", 2).get("status") == 200:
                    ready_path = paths[0]
                    break
                time.sleep(1)
            if ready_path is None:
                summary["error"] = "server_or_hook_not_ready"
                summary["server_exit"] = process.poll()
                return 1
            ready = json.loads(ready_path.read_text(encoding="utf-8"))
            pid = ready["pid"]
            ticks = ready["start_time_ticks"]
            summary["engine_pid"] = pid
            summary["engine_start_ticks"] = ticks
            summary["stage1_plugin_sha256"] = ready["plugin_sha256"]
            stage1_source = args.packages / "dfx_stage1_backpressure" / "__init__.py"
            if sha256(stage1_source) != ready["plugin_sha256"]:
                summary["error"] = "stage1_plugin_identity_mismatch"
                return 1
            if start_ticks(pid) != ticks:
                summary["error"] = "engine_identity_unstable_before_request"
                return 1
            if not args.source_equivalent_root:
                ping_source = args.packages / "dfx_health_ping_poc" / "__init__.py"
                ping_markers = list(control.glob("ping-plugin-*.json"))
                summary["ping_plugin_marker_count"] = len(ping_markers)
                markers = [json.loads(path.read_text(encoding="utf-8")) for path in ping_markers]
                if len(markers) < 2 or pid not in {marker["pid"] for marker in markers}:
                    summary["error"] = "ping_plugin_missing"
                    return 1
                if any(
                    marker["enabled"] != (args.ping == "on")
                    or marker["plugin_sha256"] != sha256(ping_source)
                    or start_ticks(marker["pid"]) != marker["start_ticks"]
                    for marker in markers
                ):
                    summary["error"] = "ping_plugin_arm_mismatch"
                    return 1
            stream = Stream()
            worker = threading.Thread(
                target=consume_stream,
                args=(f"{base}/v1/completions", stream,
                      args.prompt_words, args.max_tokens),
                daemon=True,
            )
            worker.start()
            if args.trigger == "pause":
                deadline = time.monotonic() + 90
                while time.monotonic() < deadline and not stream.finished.is_set():
                    if stream.stalled():
                        break
                    time.sleep(0.1)
                summary["stall_confirmed"] = stream.stalled()
                if not summary["stall_confirmed"]:
                    summary["error"] = "stall_not_reached"
                    return 1
                summary["health_during_stall"] = health(f"{base}/health", 75)
                summary["identity_stable_during_stall"] = start_ticks(pid) == ticks
                queue_path = control / f"queue-counts-{pid}.json"
                if queue_path.exists():
                    summary["queue_counts"] = json.loads(queue_path.read_text())
            else:
                if args.health_after_headers:
                    summary["headers_received_before_health"] = (
                        stream.headers_received.wait(timeout=15)
                    )
                    summary["progress_count_at_health_start"] = len(
                        stream.snapshot()["progress_offsets_seconds"]
                    )
                summary["health_during_control"] = health(f"{base}/health", 75)
                worker.join(timeout=90)
            summary["progress_before_release"] = stream.snapshot()
            (control / "release").touch(exist_ok=True)
            worker.join(timeout=30)
            summary["progress_after_release"] = stream.snapshot()
            summary["engine_identity_stable_at_end"] = start_ticks(pid) == ticks
            observed = summary.get("health_during_stall") or summary.get("health_during_control")
            expected_status = 503 if args.ping == "on" and args.trigger == "pause" else 200
            pause_evidence_ok = True
            if args.trigger == "pause":
                before = summary["progress_before_release"]
                after = summary["progress_after_release"]
                counts = summary.get("queue_counts")
                pause_evidence_ok = (
                    summary.get("stall_confirmed") is True
                    and summary.get("identity_stable_during_stall") is True
                    and isinstance(counts, dict)
                    and counts.get("accepted_batch_count", 0) >= 1
                    and isinstance(before, dict)
                    and len(before["progress_offsets_seconds"]) >= 2
                    and isinstance(after, dict)
                    and after["complete"] is True
                    and len(after["progress_offsets_seconds"])
                    > len(before["progress_offsets_seconds"])
                )
            if (
                isinstance(observed, dict)
                and observed.get("kind") == "http"
                and observed.get("status") == expected_status
                and summary["engine_identity_stable_at_end"]
                and pause_evidence_ok
            ):
                summary["result"] = "expected_health_response_observed"
            return 0 if summary["result"] == "expected_health_response_observed" else 1
        finally:
            (control / "release").touch(exist_ok=True)
            stop_group(process)
            summary["server_exit"] = process.returncode
            (args.output_dir / "summary.json").write_text(
                json.dumps(summary, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps({
                "ping": args.ping,
                "trigger": args.trigger,
                "result": summary["result"],
                "health": summary.get("health_during_stall")
                or summary.get("health_during_control"),
                "stall_confirmed": summary.get("stall_confirmed"),
                "error": summary.get("error"),
                "server_exit": process.returncode,
            }, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
