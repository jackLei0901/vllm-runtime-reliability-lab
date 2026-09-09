from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

from dfxlab.external_schema import (
    ExternalObservation,
    GpuAggregate,
    HealthObservation,
    MetricsObservation,
    ProcessObservation,
    RuntimeInfo,
)
from dfxlab.prometheus import parse_samples, select_metrics
from dfxlab.schema import anonymize_id, utc_now

METRIC_FIELDS = {
    "vllm:kv_cache_usage_perc": "kv_cache_usage",
    "vllm:gpu_cache_usage_perc": "kv_cache_usage",
    "vllm:num_preemptions_total": "preemptions_total",
    "vllm:num_requests_running": "running_requests",
    "vllm:num_requests_waiting": "waiting_requests",
}


def _http_get(url: str, timeout: float) -> tuple[int | None, str, str | None]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8"), None
    except urllib.error.HTTPError as exc:
        status = exc.code
        exc.close()
        return status, "", None
    except TimeoutError:
        return None, "", "timeout"
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            return None, "", "timeout"
        return None, "", "connection"
    except OSError:
        return None, "", "os_error"


def collect_health(
    base_url: str, timeout: float
) -> tuple[HealthObservation, str | None]:
    status, _, error = _http_get(f"{base_url.rstrip('/')}/health", timeout)
    return HealthObservation(ok=status == 200, status=status), error


def collect_metrics(
    base_url: str, timeout: float
) -> tuple[MetricsObservation, str | None]:
    status, body, error = _http_get(f"{base_url.rstrip('/')}/metrics", timeout)
    if error:
        return MetricsObservation(), error
    if status != 200:
        return MetricsObservation(), "http_status"
    parsed = parse_samples(body)
    if body.strip() and not parsed:
        return MetricsObservation(), "malformed_metrics"
    selected = select_metrics(body, set(METRIC_FIELDS))
    normalized: dict[str, float] = {}
    for metric_name, value in selected.items():
        field_name = METRIC_FIELDS[metric_name]
        if field_name == "kv_cache_usage" and field_name in normalized:
            continue
        normalized[field_name] = value
    return MetricsObservation(**normalized), None


def process_snapshot(pid: int | None) -> ProcessObservation:
    if pid is None:
        return ProcessObservation(tracked=False, alive=None)
    if os.name == "nt":
        alive = _windows_pid_alive(pid)
    else:
        try:
            os.kill(pid, 0)
            alive = True
        except PermissionError:
            alive = True
        except (OSError, ProcessLookupError):
            alive = False
    if not alive:
        return ProcessObservation(tracked=True, alive=False)

    rss_bytes: int | None = None
    vms_bytes: int | None = None
    threads: int | None = None
    status_path = Path(f"/proc/{pid}/status")
    if status_path.exists():
        wanted = {"VmRSS": "rss", "VmSize": "vms", "Threads": "threads"}
        values: dict[str, int] = {}
        for line in status_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            key, _, value = line.partition(":")
            if key in wanted:
                values[wanted[key]] = int(value.strip().split()[0])
        rss_bytes = values.get("rss", 0) * 1024 if "rss" in values else None
        vms_bytes = values.get("vms", 0) * 1024 if "vms" in values else None
        threads = values.get("threads")
    return ProcessObservation(
        tracked=True,
        alive=True,
        rss_bytes=rss_bytes,
        vms_bytes=vms_bytes,
        threads=threads,
    )


def _windows_pid_alive(pid: int) -> bool:
    import ctypes
    from ctypes import wintypes

    process_query_limited_information = 0x1000
    still_active = 259
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return False
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def gpu_snapshot(timeout: float = 2.0) -> tuple[GpuAggregate, tuple[str, ...]]:
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return GpuAggregate(device_count=0), ()
    fields = "memory.used,memory.total,utilization.gpu"
    try:
        completed = subprocess.run(
            [binary, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return GpuAggregate(device_count=0), ("gpu_timeout",)
    except (OSError, subprocess.SubprocessError):
        return GpuAggregate(device_count=0), ("gpu_error",)
    rows: list[tuple[int, int, int]] = []
    for line in completed.stdout.splitlines():
        values = [part.strip() for part in line.split(",")]
        if len(values) != 3:
            continue
        try:
            rows.append((int(values[0]), int(values[1]), int(values[2])))
        except ValueError:
            continue
    if not rows:
        return GpuAggregate(device_count=0), ("gpu_malformed",)
    return (
        GpuAggregate(
            device_count=len(rows),
            memory_used_mib=sum(row[0] for row in rows),
            memory_total_mib=sum(row[1] for row in rows),
            utilization_percent_mean=sum(row[2] for row in rows) / len(rows),
        ),
        (),
    )


def _module_version(module_name: str) -> str | None:
    try:
        return metadata.version(module_name)
    except metadata.PackageNotFoundError:
        return None


def runtime_allowlist(
    *,
    target_torch_version: str | None = None,
    target_vllm_version: str | None = None,
) -> RuntimeInfo:
    """Build shareable runtime metadata without guessing target versions.

    The recorder may run in a different environment from the observed server,
    so target package versions are recorded only when supplied explicitly.
    """
    gpu_models: tuple[str, ...] = ()
    binary = shutil.which("nvidia-smi")
    if binary:
        try:
            completed = subprocess.run(
                [binary, "--query-gpu=name", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            gpu_models = tuple(
                line.strip() for line in completed.stdout.splitlines() if line.strip()
            )
        except (OSError, subprocess.SubprocessError):
            pass
    return RuntimeInfo(
        python_version=platform.python_version(),
        platform=platform.system(),
        torch_version=target_torch_version,
        vllm_version=target_vllm_version,
        gpu_count=len(gpu_models),
        gpu_models=gpu_models,
    )


def environment_snapshot() -> dict[str, Any]:
    """Private lab metadata. Shareable artifacts use runtime_allowlist()."""
    snapshot: dict[str, Any] = {
        "captured_at": utc_now(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "torch": _module_version("torch"),
        "vllm": _module_version("vllm"),
    }
    binary = shutil.which("nvidia-smi")
    if binary:
        try:
            completed = subprocess.run(
                [
                    binary,
                    "--query-gpu=index,uuid,name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            snapshot["gpus"] = []
            for line in completed.stdout.splitlines():
                values = [part.strip() for part in line.split(",")]
                if len(values) == 4:
                    snapshot["gpus"].append(
                        {
                            "index": int(values[0]),
                            "uuid": anonymize_id(values[1]),
                            "name": values[2],
                            "memory_total_mib": int(values[3]),
                        }
                    )
        except (OSError, subprocess.SubprocessError, ValueError):
            snapshot["gpus"] = []
    return snapshot


class CadencedCollector:
    def __init__(
        self,
        base_url: str,
        pid: int | None,
        timeout: float,
        health_interval: float = 1.0,
        metrics_interval: float = 1.0,
        process_interval: float = 1.0,
        gpu_interval: float = 5.0,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        intervals = (health_interval, metrics_interval, process_interval, gpu_interval)
        if any(value <= 0 for value in intervals):
            raise ValueError("collector intervals must be positive")
        self.base_url = base_url
        self.pid = pid
        self.timeout = timeout
        self.clock_ns = clock_ns
        self.intervals_ns = {
            "health": int(health_interval * 1_000_000_000),
            "metrics": int(metrics_interval * 1_000_000_000),
            "process": int(process_interval * 1_000_000_000),
            "gpu": int(gpu_interval * 1_000_000_000),
        }
        self.next_due = {name: 0 for name in self.intervals_ns}
        self.health = HealthObservation(ok=False, status=None)
        self.metrics = MetricsObservation()
        self.process = ProcessObservation(tracked=pid is not None, alive=None)
        self.gpu = GpuAggregate(device_count=0)

    def collect(self, sequence: int) -> ExternalObservation:
        now_ns = self.clock_ns()
        errors: list[str] = []
        sampled_sources: list[str] = []
        if now_ns >= self.next_due["health"]:
            sampled_sources.append("health")
            self.health, error = collect_health(self.base_url, self.timeout)
            if error:
                errors.append(f"health_{error}")
            self.next_due["health"] = now_ns + self.intervals_ns["health"]
        if now_ns >= self.next_due["metrics"]:
            sampled_sources.append("metrics")
            self.metrics, error = collect_metrics(self.base_url, self.timeout)
            if error:
                errors.append(f"metrics_{error}")
            self.next_due["metrics"] = now_ns + self.intervals_ns["metrics"]
        if now_ns >= self.next_due["process"]:
            sampled_sources.append("process")
            self.process = process_snapshot(self.pid)
            self.next_due["process"] = now_ns + self.intervals_ns["process"]
        if now_ns >= self.next_due["gpu"]:
            sampled_sources.append("gpu")
            self.gpu, gpu_errors = gpu_snapshot(self.timeout)
            errors.extend(gpu_errors)
            self.next_due["gpu"] = now_ns + self.intervals_ns["gpu"]
        return ExternalObservation(
            sequence=sequence,
            observed_at=utc_now(),
            monotonic_ns=now_ns,
            health=self.health,
            process=self.process,
            metrics=self.metrics,
            gpu=self.gpu,
            sampled_sources=tuple(sampled_sources),
            collector_errors=tuple(errors),
        )
