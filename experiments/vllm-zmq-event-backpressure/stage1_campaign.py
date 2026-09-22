#!/usr/bin/env python3
"""External one-cell runner for the vLLM #53859 Stage 1 matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from stage1_contract import (
    RECOVERY_SECONDS,
    ProgressTrace,
    classify_cell,
    load_hook_error,
    load_queue_counts,
    load_ready,
    stack_matches,
)

BASE_TREE = "b7061e73a6ed4773e16bd2ae3acf47aebfd1342d"
FIX_TREE = "46bc6e191b14ce12a04827454b4588ea5d3a435f"
BASE_COMMIT = "22258a26bc090bccf5473cf681bbe9bac41bd035"
FIX_HEAD = "1a2b85306b6d13033bbecc693e6cb776acb4bcaa"
FIX_PATCH_SHA256 = "ebf0e35f53e6e3a74c79d608f6a2656d3f7537bcb3c648ce6885a8eac3423dfc"
READY_SECONDS = 180.0
REQUEST_SECONDS = 180.0
HEALTH_TIMEOUT_SECONDS = 2.0
STACK_TIMEOUT_SECONDS = 15.0
MIN_PROMPT_TOKENS = 16
MIN_COMPLETION_TOKENS = 32
HERE = Path(__file__).resolve().parent
PLUGIN_ROOT = HERE / "stage1_plugin"
CELL_ORDER = {
    1: ("base", "control"),
    2: ("fix", "control"),
    3: ("base", "pause"),
    4: ("fix", "pause"),
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def read_command(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError("server command must be a non-empty JSON array")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError("server command entries must be non-empty strings")
    return value


def command_option(command: list[str], name: str) -> str | None:
    for index, item in enumerate(command):
        if item == name:
            return command[index + 1] if index + 1 < len(command) else None
        prefix = f"{name}="
        if item.startswith(prefix):
            return item.removeprefix(prefix)
    return None


def command_option_values(command: list[str], name: str) -> list[str | None]:
    values: list[str | None] = []
    for index, item in enumerate(command):
        if item == name:
            values.append(command[index + 1] if index + 1 < len(command) else None)
        elif item.startswith(f"{name}="):
            values.append(item.removeprefix(f"{name}="))
    return values


def validate_server_command(command: list[str]) -> None:
    required_switches = {"--enable-prefix-caching", "--async-scheduling"}
    missing = sorted(
        option for option in required_switches if command.count(option) != 1
    )
    if missing:
        raise ValueError(
            f"required server switches must appear once: {', '.join(missing)}"
        )
    forbidden_switches = {"--no-enable-prefix-caching", "--no-async-scheduling"}
    present_forbidden = sorted(
        option for option in forbidden_switches if option in command
    )
    if present_forbidden:
        raise ValueError(
            f"negated server switches are forbidden: {', '.join(present_forbidden)}"
        )
    block_sizes = command_option_values(command, "--block-size")
    if block_sizes != ["16"]:
        raise ValueError("server command must set --block-size 16")
    raw_configs = command_option_values(command, "--kv-events-config")
    if len(raw_configs) != 1 or raw_configs[0] is None:
        raise ValueError("server command must set --kv-events-config exactly once")
    raw_config = raw_configs[0]
    config = json.loads(raw_config)
    if not isinstance(config, dict):
        raise ValueError("kv-events config must be a JSON object")
    if set(config) != {
        "enable_kv_cache_events",
        "endpoint",
        "max_queue_size",
        "publisher",
    }:
        raise ValueError("kv-events config has unexpected fields")
    if config.get("enable_kv_cache_events") is not True:
        raise ValueError("kv-events must be explicitly enabled")
    if config.get("publisher") != "zmq" or config.get("max_queue_size") != 1:
        raise ValueError("kv-events must use ZMQ with max_queue_size 1")
    if config.get("endpoint") != "tcp://127.0.0.1:5557":
        raise ValueError("kv-events endpoint must use fixed loopback TCP")


def validate_request(request_bytes: bytes) -> None:
    request = json.loads(request_bytes)
    if not isinstance(request, dict):
        raise ValueError("request must be a JSON object")
    if request.get("stream") is not True:
        raise ValueError("request must enable streaming")
    stream_options = request.get("stream_options")
    if (
        not isinstance(stream_options, dict)
        or stream_options.get("include_usage") is not True
    ):
        raise ValueError("request must include streaming usage")
    if request.get("temperature") != 0:
        raise ValueError("request must use deterministic temperature 0")
    max_tokens = request.get("max_tokens")
    if (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens < 32
    ):
        raise ValueError("request must allow at least 32 output tokens")


def source_tree(workdir: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=workdir,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def source_is_clean(workdir: Path) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=workdir,
        check=True,
        capture_output=True,
    )
    return not result.stdout


def git_file_sha256(workdir: Path, relative_path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"HEAD:{relative_path}"],
        cwd=workdir,
        check=True,
        capture_output=True,
    )
    return sha256_bytes(result.stdout)


def process_start_time_ticks(pid: int) -> int | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        closing = raw.rfind(")")
        if closing < 0:
            return None
        return int(raw[closing + 2 :].split()[19])
    except (FileNotFoundError, PermissionError, ValueError, IndexError):
        return None


def identity_is_live(pid: int, start_time_ticks: int) -> bool:
    return process_start_time_ticks(pid) == start_time_ticks


def implementation_hashes() -> dict[str, str]:
    paths = {
        "campaign": HERE / "stage1_campaign.py",
        "contract": HERE / "stage1_contract.py",
        "plugin": PLUGIN_ROOT / "src" / "dfx_stage1_backpressure" / "__init__.py",
        "plugin_package": PLUGIN_ROOT / "pyproject.toml",
        "protocol": HERE / "STAGE1_PROTOCOL_DRAFT.md",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def probe_environment(python: str, workdir: Path) -> dict[str, Any]:
    code = """
import json
import platform
import torch
import vllm
import vllm.distributed.kv_events as kv_events
record = {
    "python": platform.python_version(),
    "vllm": vllm.__version__,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu_count": torch.cuda.device_count(),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "gpu_capability": (
        list(torch.cuda.get_device_capability(0))
        if torch.cuda.is_available()
        else None
    ),
    "vllm_file": vllm.__file__,
    "kv_events_file": kv_events.__file__,
}
print(json.dumps(record, sort_keys=True))
"""
    result = subprocess.run(
        [python, "-c", code],
        cwd=workdir,
        check=True,
        capture_output=True,
        text=True,
    )
    record = json.loads(result.stdout)
    if set(record) != {
        "cuda",
        "cuda_available",
        "gpu_capability",
        "gpu_count",
        "gpu_name",
        "python",
        "torch",
        "vllm",
        "vllm_file",
        "kv_events_file",
    }:
        raise ValueError("unexpected environment probe fields")
    root = workdir.resolve()
    vllm_file = Path(record.pop("vllm_file")).resolve()
    kv_events_file = Path(record.pop("kv_events_file")).resolve()
    if not vllm_file.is_relative_to(root) or not kv_events_file.is_relative_to(root):
        raise RuntimeError("vLLM imports do not resolve inside the source worktree")
    vllm_relative = vllm_file.relative_to(root).as_posix()
    kv_events_relative = kv_events_file.relative_to(root).as_posix()
    if vllm_relative != "vllm/__init__.py":
        raise RuntimeError("unexpected imported vLLM package path")
    if kv_events_relative != "vllm/distributed/kv_events.py":
        raise RuntimeError("unexpected imported kv_events path")
    imported_hash = sha256_file(kv_events_file)
    tree_hash = git_file_sha256(workdir, kv_events_relative)
    if imported_hash != tree_hash:
        raise RuntimeError("imported kv_events bytes do not match the Git tree")
    record.update(
        {
            "vllm_relative_file": vllm_relative,
            "kv_events_relative_file": kv_events_relative,
            "kv_events_sha256": imported_hash,
            "tree_kv_events_sha256": tree_hash,
            "import_matches_tree": True,
        }
    )
    return record


def health_class(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=HEALTH_TIMEOUT_SECONDS) as response:
            return "2xx" if 200 <= response.status < 300 else "non_2xx"
    except urllib.error.HTTPError:
        return "non_2xx"
    except (OSError, urllib.error.URLError):
        return "unavailable"


class StreamObserver:
    def __init__(self) -> None:
        self.trace = ProgressTrace(started_at=time.monotonic())
        self._lock = threading.Lock()
        self.finished = threading.Event()
        self.error_kind: str | None = None
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None

    def observe(self) -> None:
        with self._lock:
            self.trace.observe(time.monotonic())

    def complete(self) -> None:
        with self._lock:
            self.trace.complete(time.monotonic())
        self.finished.set()

    def fail(self, kind: str) -> None:
        with self._lock:
            self.error_kind = kind
        self.finished.set()

    def record_usage(self, prompt_count: int, completion_count: int) -> None:
        with self._lock:
            self.prompt_tokens = prompt_count
            self.completion_tokens = completion_count

    def stalled(self) -> bool:
        with self._lock:
            return self.trace.stalled(time.monotonic())

    def public_trace(self) -> dict[str, Any]:
        with self._lock:
            return self.trace.public_record()

    def release_snapshot(self) -> tuple[float, int]:
        with self._lock:
            offset = time.monotonic() - self.trace.started_at
            return offset, len(self.trace.offsets)

    def recovered_after(self, progress_count: int) -> bool:
        with self._lock:
            return (
                self.trace.completed_at is not None
                and len(self.trace.offsets) > progress_count
            )

    def classify(
        self,
        *,
        source_arm: str,
        trigger: str,
        stalled_before_release: bool,
        stack_match: bool | None,
        recovered_after_release: bool | None,
        accepted_batch_count: int,
        dropped_batch_count: int,
    ) -> str:
        with self._lock:
            return classify_cell(
                source_arm=source_arm,
                trigger=trigger,
                trace=self.trace,
                stalled_before_release=stalled_before_release,
                stack_match=stack_match,
                recovered_after_release=recovered_after_release,
                accepted_batch_count=accepted_batch_count,
                dropped_batch_count=dropped_batch_count,
            )


def _has_stream_progress(payload: dict[str, Any]) -> bool:
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if isinstance(delta, dict) and delta.get("content"):
            return True
        if choice.get("text"):
            return True
    return False


def consume_stream(url: str, request_bytes: bytes, observer: StreamObserver) -> None:
    request = urllib.request.Request(
        url,
        data=request_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_SECONDS) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="strict").strip()
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    observer.complete()
                    return
                payload = json.loads(data)
                if _has_stream_progress(payload):
                    observer.observe()
                usage = payload.get("usage")
                if (
                    isinstance(usage, dict)
                    and isinstance(usage.get("prompt_tokens"), int)
                    and isinstance(usage.get("completion_tokens"), int)
                ):
                    observer.record_usage(
                        usage["prompt_tokens"], usage["completion_tokens"]
                    )
        observer.fail("stream_ended_without_done")
    except urllib.error.HTTPError:
        observer.fail("http_error")
    except TimeoutError:
        observer.fail("transport_timeout")
    except (OSError, urllib.error.URLError):
        observer.fail("transport_error")
    except (UnicodeError, json.JSONDecodeError, TypeError):
        observer.fail("malformed_stream")


def wait_for_file(path: Path, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.05)
    return False


def capture_stack(py_spy: str, pid: int, raw_path: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [py_spy, "dump", "--pid", str(pid)],
            check=False,
            capture_output=True,
            timeout=STACK_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"available": False, "match": None, "raw_sha256": None}
    raw = result.stdout + result.stderr
    raw_path.write_bytes(raw)
    return {
        "available": result.returncode == 0,
        "match": stack_matches(raw.decode("utf-8", errors="replace"))
        if result.returncode == 0
        else None,
        "raw_sha256": sha256_bytes(raw),
    }


def process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_process_group_exit(pgid: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not process_group_exists(pgid):
            return True
        time.sleep(0.05)
    return not process_group_exists(pgid)


def stop_process_group(
    process: subprocess.Popen[bytes],
    engine_identity: tuple[int, int] | None,
) -> dict[str, Any]:
    pgid = process.pid
    termination = "exited"
    try:
        os.killpg(pgid, signal.SIGTERM)
        termination = "terminated"
    except ProcessLookupError:
        pass
    except OSError:
        termination = "signal_failed"

    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass

    group_gone = wait_for_process_group_exit(pgid, 1)
    if not group_gone:
        try:
            os.killpg(pgid, signal.SIGKILL)
            termination = "killed"
        except ProcessLookupError:
            pass
        except OSError:
            termination = "signal_failed"
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            termination = "cleanup_timeout"
        group_gone = wait_for_process_group_exit(pgid, 10)
        if not group_gone:
            termination = "cleanup_timeout"

    engine_gone = (
        None
        if engine_identity is None
        else not identity_is_live(engine_identity[0], engine_identity[1])
    )
    return {
        "engine_core_gone": engine_gone,
        "process_group_gone": group_gone,
        "termination": termination,
    }


def bounded_error_kind(error: Exception) -> str:
    if isinstance(error, json.JSONDecodeError):
        return "invalid_json"
    if isinstance(error, FileNotFoundError):
        return "missing_file"
    if isinstance(error, PermissionError):
        return "permission_error"
    if isinstance(error, (ValueError, RuntimeError)):
        return "contract_error"
    if isinstance(error, OSError):
        return "os_error"
    return "unexpected_error"


def prioritize_classification(
    candidate: str,
    *,
    hook_integrity_failed: bool,
    stream_error_kind: str | None,
    release_observed: bool,
    request_sized: bool,
) -> str:
    if hook_integrity_failed:
        return "hook_failed"
    if stream_error_kind is not None:
        return "stream_failed"
    if not release_observed:
        return "hook_release_missing"
    if not request_sized:
        return "trigger_not_reached"
    return candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-arm", choices=("base", "fix"), required=True)
    parser.add_argument("--trigger", choices=("control", "pause"), required=True)
    parser.add_argument("--cell-index", type=int, choices=CELL_ORDER, required=True)
    parser.add_argument("--server-workdir", type=Path, required=True)
    parser.add_argument("--server-command-json", type=Path, required=True)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--build-identity-json", type=Path, required=True)
    parser.add_argument("--health-url", required=True)
    parser.add_argument("--stream-url", required=True)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--py-spy", default="py-spy")
    parser.add_argument(
        "--capture-control-stack",
        action="store_true",
        help=(
            "capture through the observer process before the control request; "
            "reserved for the Stage B native-producer capability run"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if os.name != "posix":
        raise SystemExit("Stage 1 campaign requires POSIX process groups")
    if args.capture_control_stack and args.trigger != "control":
        raise SystemExit("--capture-control-stack requires --trigger control")
    if CELL_ORDER[args.cell_index] != (args.source_arm, args.trigger):
        raise SystemExit("cell index does not match the frozen execution order")
    expected_tree = BASE_TREE if args.source_arm == "base" else FIX_TREE
    actual_tree = source_tree(args.server_workdir)
    if actual_tree != expected_tree:
        raise SystemExit(f"source tree mismatch: {actual_tree}")
    if not source_is_clean(args.server_workdir):
        raise SystemExit("vLLM source worktree is not clean")

    command = read_command(args.server_command_json)
    if not Path(command[0]).name.startswith("python"):
        raise SystemExit("server command must start with its Python interpreter")
    validate_server_command(command)
    runtime_environment = probe_environment(command[0], args.server_workdir)
    build_identity_bytes = args.build_identity_json.read_bytes()
    build_identity = json.loads(build_identity_bytes)
    if build_identity.get("source_tree") != actual_tree:
        raise SystemExit("build identity source tree mismatch")
    wheel_binaries = build_identity.get("installed_wheel_binaries")
    if not isinstance(wheel_binaries, dict) or not wheel_binaries:
        raise SystemExit("build identity has no installed wheel binaries")
    request_bytes = args.request_json.read_bytes()
    validate_request(request_bytes)
    args.private_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    control_dir = args.private_dir / "control"
    control_dir.mkdir()
    if args.trigger == "control":
        (control_dir / "release").touch()

    environment = os.environ.copy()
    environment.update(
        {
            "VLLM_PLUGINS": "dfx_stage1_backpressure",
            "DFX_STAGE1_ENABLE": "1",
            "DFX_STAGE1_CONTROL_DIR": str(control_dir.resolve()),
            "DFX_STAGE1_OBSERVER_PID": str(os.getpid()),
        }
    )
    server_log = args.private_dir / "server.log"
    implementation = implementation_hashes()
    with server_log.open("wb") as raw_server:
        process = subprocess.Popen(
            command,
            cwd=args.server_workdir,
            env=environment,
            stdout=raw_server,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        summary: dict[str, Any] = {
            "campaign_error_kind": None,
            "schema_version": 1,
            "cell_index": args.cell_index,
            "source_arm": args.source_arm,
            "trigger": args.trigger,
            "source": {
                "base_commit": BASE_COMMIT,
                "fix_head": FIX_HEAD,
                "fix_patch_sha256": FIX_PATCH_SHA256,
                "tree": actual_tree,
            },
            "implementation_sha256": implementation,
            "environment": runtime_environment,
            "build_identity_sha256": sha256_bytes(build_identity_bytes),
            "server_command_sha256": sha256_file(args.server_command_json),
            "request_sha256": sha256_bytes(request_bytes),
        }
        return_code = 1
        engine_identity: tuple[int, int] | None = None
        try:
            health_ready = False
            ready_paths: list[Path] = []
            hook_error_paths: list[Path] = []
            deadline = time.monotonic() + READY_SECONDS
            while time.monotonic() < deadline and process.poll() is None:
                health_ready = health_class(args.health_url) == "2xx"
                ready_paths = sorted(control_dir.glob("ready-*.json"))
                hook_error_paths = sorted(control_dir.glob("hook-error-*.json"))
                if hook_error_paths or len(ready_paths) > 1:
                    break
                if health_ready and len(ready_paths) == 1:
                    break
                time.sleep(0.25)
            summary["hook_instance_count"] = len(ready_paths)
            summary["hook_error_kind"] = None
            if hook_error_paths:
                kinds = [load_hook_error(path) for path in hook_error_paths]
                summary["hook_error_kind"] = (
                    kinds[0] if len(kinds) == 1 else "multiple_hook_errors"
                )
                summary["classification"] = "hook_failed"
                return_code = 1
            elif len(ready_paths) != 1:
                summary["classification"] = (
                    "hook_failed" if len(ready_paths) > 1 else "server_not_ready"
                )
                return_code = 1
            elif not health_ready:
                summary["classification"] = "server_not_ready"
                return_code = 1
            else:
                ready = load_ready(ready_paths[0])
                if ready["plugin_sha256"] != implementation["plugin"]:
                    raise RuntimeError(
                        "loaded plugin bytes do not match the lab plugin"
                    )
                if (
                    ready["kv_events_sha256"]
                    != runtime_environment["tree_kv_events_sha256"]
                ):
                    raise RuntimeError(
                        "EngineCore kv_events bytes do not match the source tree"
                    )
                mapped_binaries = ready["mapped_worktree_binaries"]
                if not mapped_binaries:
                    raise RuntimeError("EngineCore mapped no worktree binaries")
                if any(
                    relative not in wheel_binaries
                    or wheel_binaries[relative].get("sha256") != digest
                    for relative, digest in mapped_binaries.items()
                ):
                    raise RuntimeError("EngineCore mapped binary identity mismatch")
                identity_bound = identity_is_live(
                    ready["pid"], ready["start_time_ticks"]
                )
                if not identity_bound:
                    raise RuntimeError(
                        "EngineCore process identity changed before request"
                    )
                engine_identity = (ready["pid"], ready["start_time_ticks"])
                stack = {"available": None, "match": None, "raw_sha256": None}
                if args.trigger == "control" and args.capture_control_stack:
                    stack = capture_stack(
                        args.py_spy,
                        ready["pid"],
                        args.private_dir / "enginecore-stack.txt",
                    )
                observer = StreamObserver()
                stream_thread = threading.Thread(
                    target=consume_stream,
                    args=(args.stream_url, request_bytes, observer),
                    daemon=True,
                )
                stream_thread.start()

                stalled = False
                health_during_stall: str | None = None
                release_offset: float | None = None
                progress_count_at_release = 0
                if args.source_arm == "base" and args.trigger == "pause":
                    deadline = time.monotonic() + REQUEST_SECONDS
                    while (
                        time.monotonic() < deadline and not observer.finished.is_set()
                    ):
                        if observer.stalled():
                            stalled = True
                            break
                        time.sleep(0.05)
                    if stalled:
                        health_during_stall = health_class(args.health_url)
                        if identity_is_live(ready["pid"], ready["start_time_ticks"]):
                            stack = capture_stack(
                                args.py_spy,
                                ready["pid"],
                                args.private_dir / "enginecore-stack.txt",
                            )
                        else:
                            stack = {
                                "available": False,
                                "match": None,
                                "raw_sha256": None,
                            }
                    release_offset, progress_count_at_release = (
                        observer.release_snapshot()
                    )
                    (control_dir / "release").touch()
                    observer.finished.wait(RECOVERY_SECONDS)
                    recovered = observer.recovered_after(progress_count_at_release)
                elif args.source_arm == "fix" and args.trigger == "pause":
                    recovered = None
                    deadline = time.monotonic() + REQUEST_SECONDS
                    while (
                        time.monotonic() < deadline and not observer.finished.is_set()
                    ):
                        if observer.stalled():
                            stalled = True
                            break
                        time.sleep(0.05)
                    release_offset, progress_count_at_release = (
                        observer.release_snapshot()
                    )
                    (control_dir / "release").touch()
                else:
                    recovered = None
                    observer.finished.wait(REQUEST_SECONDS)

                stream_thread.join(timeout=5)
                release_observed = wait_for_file(
                    control_dir / f"released-{ready['pid']}.json", seconds=5
                )
                counts = load_queue_counts(
                    control_dir / f"queue-counts-{ready['pid']}.json"
                )
                classification = observer.classify(
                    source_arm=args.source_arm,
                    trigger=args.trigger,
                    stalled_before_release=stalled,
                    stack_match=stack["match"],
                    recovered_after_release=recovered,
                    accepted_batch_count=counts["accepted_batch_count"],
                    dropped_batch_count=counts["dropped_batch_count"],
                )
                final_ready_paths = sorted(control_dir.glob("ready-*.json"))
                final_hook_error_paths = sorted(control_dir.glob("hook-error-*.json"))
                summary["hook_instance_count"] = len(final_ready_paths)
                if final_hook_error_paths:
                    kinds = [load_hook_error(path) for path in final_hook_error_paths]
                    summary["hook_error_kind"] = (
                        kinds[0] if len(kinds) == 1 else "multiple_hook_errors"
                    )
                hook_integrity_failed = (
                    len(final_ready_paths) != 1
                    or final_ready_paths[0] != ready_paths[0]
                    or bool(final_hook_error_paths)
                )
                request_sized = not (
                    observer.prompt_tokens is None
                    or observer.prompt_tokens < MIN_PROMPT_TOKENS
                    or observer.completion_tokens is None
                    or observer.completion_tokens < MIN_COMPLETION_TOKENS
                )
                classification = prioritize_classification(
                    classification,
                    hook_integrity_failed=hook_integrity_failed,
                    stream_error_kind=observer.error_kind,
                    release_observed=release_observed,
                    request_sized=request_sized,
                )
                summary.update(
                    {
                        "accepted_batch_count": counts["accepted_batch_count"],
                        "classification": classification,
                        "completion_tokens": observer.completion_tokens,
                        "dropped_batch_count": counts["dropped_batch_count"],
                        "engine_core_bound": identity_bound,
                        "engine_core_kv_events_sha256": ready["kv_events_sha256"],
                        "engine_core_mapped_worktree_binaries": mapped_binaries,
                        "health_during_stall": health_during_stall,
                        "hook_ready": True,
                        "observer_authorization": ready["authorization"],
                        "progress_count_at_release": progress_count_at_release,
                        "prompt_tokens": observer.prompt_tokens,
                        "release_offset_seconds": release_offset,
                        "release_observed": release_observed,
                        "progress": observer.public_trace(),
                        "recovered_after_release": recovered,
                        "stack": stack,
                        "stalled_before_release": stalled,
                        "stream_error_kind": observer.error_kind,
                        "yama_ptrace_scope": ready["yama_ptrace_scope"],
                    }
                )
                return_code = 0 if classification == "pass" else 1
        except Exception as error:
            summary["campaign_error_kind"] = bounded_error_kind(error)
            summary["classification"] = "campaign_error"
            return_code = 1
        finally:
            if args.trigger == "pause":
                (control_dir / "release").touch(exist_ok=True)
            cleanup = stop_process_group(process, engine_identity)
            summary["cleanup"] = cleanup
            if (
                cleanup["termination"] not in {"exited", "terminated", "killed"}
                or cleanup["process_group_gone"] is not True
                or (
                    engine_identity is not None
                    and cleanup["engine_core_gone"] is not True
                )
            ):
                summary["classification"] = "cleanup_failed"
                return_code = 1
            raw_server.flush()
            os.fsync(raw_server.fileno())
            summary["private_server_log_sha256"] = sha256_file(server_log)
            atomic_json(args.summary, summary)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
