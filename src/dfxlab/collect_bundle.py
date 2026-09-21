from __future__ import annotations

import ctypes
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from ctypes import wintypes
from pathlib import Path
from typing import Any

from dfxlab.bundle import OBSERVATION_SCHEMA, write_bundle
from dfxlab.prometheus import select_metrics
from dfxlab.stacks import capture_stack, disabled_stack

MAX_HTTP_BYTES = 1024 * 1024
MAX_SAMPLES = 10_000


def _process_identity(pid: int) -> tuple[str, str] | None:
    if os.name == "nt":
        query = 0x1000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetProcessTimes.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
        ]
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel32.OpenProcess(query, False, pid)
        if not handle:
            return None
        try:
            creation = wintypes.FILETIME()
            exit_time = wintypes.FILETIME()
            kernel = wintypes.FILETIME()
            user = wintypes.FILETIME()
            if not kernel32.GetProcessTimes(
                handle,
                ctypes.byref(creation),
                ctypes.byref(exit_time),
                ctypes.byref(kernel),
                ctypes.byref(user),
            ):
                return None
            value = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
            return "windows_creation_time", str(value)
        finally:
            kernel32.CloseHandle(handle)
    path = Path(f"/proc/{pid}/stat")
    try:
        line = path.read_text(encoding="ascii")
    except OSError:
        return None
    close = line.rfind(")")
    fields = line[close + 2 :].split()
    if close < 0 or len(fields) < 20:
        return None
    return "linux_start_ticks", fields[19]


def _endpoint_id(base_url: str) -> str:
    nonce = os.urandom(16)
    digest = hashlib.sha256(nonce + base_url.encode()).hexdigest()[:12]
    return f"sha256:{digest}"


def _metrics(base_url: str, timeout: float) -> tuple[dict[str, float], str | None]:
    try:
        with urllib.request.urlopen(
            f"{base_url.rstrip('/')}/metrics", timeout=timeout
        ) as response:
            if response.status != 200:
                return {}, "http_status"
            encoded = response.read(MAX_HTTP_BYTES + 1)
            if len(encoded) > MAX_HTTP_BYTES:
                return {}, "size_limit"
            body = encoded.decode("utf-8")
    except TimeoutError:
        return {}, "timeout"
    except urllib.error.HTTPError as exc:
        exc.close()
        return {}, "http_status"
    except urllib.error.URLError:
        return {}, "connection"
    except (OSError, UnicodeError):
        return {}, "os_error"
    selected = select_metrics(
        body,
        {
            "vllm:generation_tokens_total",
            "vllm:num_requests_running",
            "vllm:num_requests_waiting",
        },
    )
    if body.strip() and not selected:
        return {}, "malformed_metrics"
    return selected, None


def _health(base_url: str, timeout: float) -> tuple[bool, int | None, str | None]:
    try:
        with urllib.request.urlopen(
            f"{base_url.rstrip('/')}/health", timeout=timeout
        ) as response:
            return 200 <= response.status < 300, response.status, None
    except TimeoutError:
        return False, None, "timeout"
    except urllib.error.HTTPError as exc:
        status = exc.code
        exc.close()
        return False, status, None
    except urllib.error.URLError:
        return False, None, "connection"
    except OSError:
        return False, None, "os_error"


class _ClientProbe:
    def __init__(self, base_url: str, request_path: Path, timeout: float) -> None:
        self.base_url = base_url
        self.request_path = request_path
        self.timeout = timeout
        self.request_started_ns = 0
        self.request_completed_ns: int | None = None
        self.chunks: list[dict[str, Any]] = []
        self.available = True
        self._response: Any = None
        self._stopping = False
        self.request_bytes = request_path.read_bytes()
        if len(self.request_bytes) > MAX_HTTP_BYTES:
            raise ValueError("progress request exceeds 1 MiB")
        self.request_sha256 = hashlib.sha256(self.request_bytes).hexdigest()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.request_started_ns = time.perf_counter_ns()
        self._thread.start()

    def stop(self) -> None:
        self._stopping = True
        response = self._response
        if response is not None:
            try:
                response.close()
            except (OSError, ValueError):
                pass
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        try:
            payload = json.loads(self.request_bytes)
            if not isinstance(payload, dict):
                raise ValueError("request root must be an object")
            payload["stream"] = True
            request = urllib.request.Request(
                f"{self.base_url.rstrip('/')}/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                self._response = response
                for raw_line in response:
                    if len(raw_line) > MAX_HTTP_BYTES:
                        raise ValueError("stream event exceeds 1 MiB")
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    kind = _chunk_kind(data)
                    if kind == "content":
                        self.chunks[:] = [
                            {"monotonic_ns": time.perf_counter_ns(), "kind": kind}
                        ]
        except (OSError, ValueError, json.JSONDecodeError):
            if not self._stopping:
                self.available = False
                self.chunks.clear()
        finally:
            self._response = None
            self.request_completed_ns = time.perf_counter_ns()


def _chunk_kind(data: str) -> str:
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return "empty"
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if isinstance(choices, list):
        for choice in choices:
            delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
            if isinstance(delta, dict) and delta.get("content"):
                return "content"
            if isinstance(delta, dict) and delta.get("role"):
                return "role_only"
    if isinstance(payload, dict) and payload.get("usage"):
        return "usage_only"
    return "empty"


def collect_bundle(
    *,
    output_dir: Path,
    base_url: str | None,
    pid: int | None,
    window: float,
    no_progress_window: float,
    sample_interval: float,
    timeout: float,
    unhealthy_samples: int,
    observation_only: bool,
    decision_source: str,
    progress_request: Path | None,
    stack: bool,
) -> dict[str, Any]:
    if pid is None and base_url is None:
        raise ValueError("collect requires --pid or --base-url")
    if base_url is None and not observation_only:
        raise ValueError("PID-only collection requires --observation-only")
    if decision_source not in {"server_counter", "client_request"}:
        raise ValueError("unknown decision source")
    if decision_source == "client_request" and progress_request is None:
        raise ValueError("client_request requires --progress-request")
    if progress_request is not None and base_url is None:
        raise ValueError("--progress-request requires --base-url")
    if stack and pid is None:
        raise ValueError("--stack requires --pid")
    if min(window, no_progress_window, sample_interval, timeout) <= 0:
        raise ValueError("collection durations must be positive")
    if window < no_progress_window or unhealthy_samples <= 0:
        raise ValueError("invalid collection window or health threshold")
    if window / sample_interval + 1 > MAX_SAMPLES:
        raise ValueError("collection exceeds the 10000-sample bound")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("collect output directory must be empty")

    start_identity = _process_identity(pid) if pid is not None else None
    if pid is not None and start_identity is None:
        raise ValueError("process start identity unavailable")
    probe = (
        _ClientProbe(base_url, progress_request, window + timeout)
        if base_url is not None and progress_request is not None
        else None
    )
    if probe is not None:
        probe.start()

    # perf_counter_ns is monotonic and has enough resolution to preserve the
    # verifier's strict sample ordering on Windows implementations where
    # monotonic_ns may be backed by the coarse GetTickCount64 clock.
    collection_start = time.perf_counter_ns()
    deadline = collection_start + int(window * 1_000_000_000)
    health_samples: list[dict[str, Any]] = []
    process_samples: list[dict[str, Any]] = []
    counter_samples: list[dict[str, Any]] = []
    demand_samples: list[dict[str, Any]] = []
    while True:
        now = time.perf_counter_ns()
        if base_url is not None:
            health_ok, health_status, health_error = _health(base_url, timeout)
            health_samples.append(
                {
                    "monotonic_ns": now,
                    "fresh": True,
                    "ok": health_ok,
                    "status": health_status,
                    "error_kind": health_error,
                }
            )
            metrics, _metrics_error = _metrics(base_url, timeout)
            if "vllm:generation_tokens_total" in metrics:
                counter_samples.append(
                    {
                        "monotonic_ns": now,
                        "fresh": True,
                        "value": metrics["vllm:generation_tokens_total"],
                    }
                )
            if {
                "vllm:num_requests_running",
                "vllm:num_requests_waiting",
            } <= metrics.keys():
                demand_samples.append(
                    {
                        "monotonic_ns": now,
                        "fresh": True,
                        "running": metrics["vllm:num_requests_running"],
                        "waiting": metrics["vllm:num_requests_waiting"],
                    }
                )
        if pid is not None:
            identity = _process_identity(pid)
            process_samples.append(
                {
                    "monotonic_ns": now,
                    "fresh": True,
                    "alive": identity is not None,
                    "start_identity_value": identity[1] if identity else None,
                }
            )
        if now >= deadline:
            break
        remaining = max(0.0, (deadline - time.perf_counter_ns()) / 1e9)
        if remaining == 0:
            continue
        time.sleep(min(sample_interval, remaining))

    collection_end = health_samples[-1]["monotonic_ns"] if health_samples else now
    if process_samples:
        collection_end = max(collection_end, process_samples[-1]["monotonic_ns"])
    cutoff = collection_end - int(no_progress_window * 1_000_000_000)
    timeline = health_samples or process_samples
    evaluation_start = max(
        (
            sample["monotonic_ns"]
            for sample in timeline
            if sample["monotonic_ns"] <= cutoff
        ),
        default=timeline[0]["monotonic_ns"],
    )
    evaluation_end = collection_end
    if probe is not None:
        probe.stop()

    def within(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            sample
            for sample in samples
            if evaluation_start <= sample["monotonic_ns"] <= evaluation_end
        ]

    client_available = probe is not None and probe.available
    client_started = probe.request_started_ns if probe is not None else 0
    client_completed = probe.request_completed_ns if probe is not None else None
    client_chunks = within(probe.chunks) if probe is not None else []
    if not client_available:
        client_chunks = []
    stack_evidence = (
        capture_stack(pid, output_dir / "private" / "stacks.txt", timeout)
        if stack and pid is not None
        else disabled_stack()
    )
    server_counter_input = {
        "producer_available": bool(counter_samples),
        "evaluation_start_ns": evaluation_start,
        "evaluation_end_ns": evaluation_end,
        "minimum_span_ns": int(no_progress_window * 1_000_000_000),
        "minimum_fresh_samples": 2,
        "samples": within(counter_samples),
    }
    observations = {
        "schema_version": OBSERVATION_SCHEMA,
        "targets": {
            "process": {
                "supplied": pid is not None,
                "pid": pid,
                "start_identity": (
                    {"kind": start_identity[0], "value": start_identity[1]}
                    if start_identity
                    else None
                ),
                "identity_source": "operator_supplied",
            },
            "endpoint": {
                "supplied": base_url is not None,
                "endpoint_id": _endpoint_id(base_url) if base_url else None,
                "identity_source": "operator_supplied",
            },
            "relation": "operator_asserted_same_incident",
        },
        "intervals": {
            "collection": {"start_ns": collection_start, "end_ns": collection_end},
            "evaluation": {
                "start_ns": evaluation_start,
                "end_ns": evaluation_end,
                "minimum_duration_ns": int(no_progress_window * 1_000_000_000),
            },
        },
        "process": {"samples": within(process_samples)},
        "health": {
            "failure_threshold": unhealthy_samples,
            "samples": health_samples,
        },
        "producer_inputs": {
            "decision_source": decision_source,
            "corroborating_source": (
                "client_request"
                if decision_source == "server_counter" and probe is not None
                else "server_counter"
                if decision_source == "client_request"
                else None
            ),
            "client_request_sha256": probe.request_sha256 if probe else None,
            "server_counter": server_counter_input,
            "client_request": {
                "producer_available": client_available,
                "evaluation_start_ns": evaluation_start,
                "evaluation_end_ns": evaluation_end,
                "request_started_ns": client_started,
                "request_completed_ns": client_completed,
                "chunks": client_chunks,
            },
        },
        "demand_inputs": {
            "server_counter": {
                "evaluation_start_ns": evaluation_start,
                "evaluation_end_ns": evaluation_end,
                "minimum_fresh_samples": 2,
                "samples": within(demand_samples),
            },
            "client_request": {
                "evaluation_start_ns": evaluation_start,
                "evaluation_end_ns": evaluation_end,
                "request_started_ns": client_started,
                "request_completed_ns": client_completed,
            },
        },
        "stack": stack_evidence,
    }
    return write_bundle(output_dir, observations)
