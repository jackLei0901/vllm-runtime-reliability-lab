#!/usr/bin/env python3
"""Measure top-level DPSupervisor status after normal and abnormal exits."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import importlib.util
import json
import logging
import os
import platform
import signal
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

SUBJECT_PREFIX = "GATE0_SUBJECT_JSON="


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PopenProcessAdapter:
    """Expose the BaseProcess surface used by DPSupervisor cleanup."""

    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self._process = process
        self.name = "gate0-child"

    @property
    def pid(self) -> int:
        return self._process.pid

    @property
    def exitcode(self) -> int | None:
        return self._process.poll()

    def is_alive(self) -> bool:
        return self._process.poll() is None

    def join(self, timeout: float | None = None) -> None:
        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return


def make_args() -> argparse.Namespace:
    return argparse.Namespace(
        host=None,
        port=18000,
        data_parallel_multi_port_external_lb=True,
        data_parallel_supervisor_port=19256,
        dp_supervisor_probe_interval_s=0.02,
        dp_supervisor_probe_timeout_s=0.1,
        dp_supervisor_probe_failure_threshold=1,
        data_parallel_size=2,
        data_parallel_size_local=2,
        data_parallel_start_rank=0,
        data_parallel_rank=None,
        data_parallel_external_lb=False,
        data_parallel_hybrid_lb=False,
        enable_fault_tolerance=False,
        api_server_count=None,
        headless=False,
        grpc=False,
        uds=None,
        ssl_keyfile=None,
        ssl_certfile=None,
        ssl_ca_certs=None,
        ssl_cert_reqs=0,
        ssl_ciphers=None,
        node_rank=0,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        uvicorn_log_level="warning",
        shutdown_timeout=0.0,
        disable_uvicorn_access_log=False,
        disable_access_log_for_endpoints=None,
        log_config_file=None,
    )


def install_vllm_import_stubs() -> None:
    """Install only the vLLM names needed to load the lifecycle module."""

    def package(name: str) -> types.ModuleType:
        module = types.ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = module
        return module

    package("vllm")
    package("vllm.entrypoints")
    package("vllm.entrypoints.launchers")
    package("vllm.entrypoints.launchers.utils")
    package("vllm.utils")
    package("vllm.v1")
    package("vllm.v1.engine")

    envs = types.ModuleType("vllm.envs")
    envs.VLLM_USE_RUST_FRONTEND = False
    envs.VLLM_RUST_FRONTEND_PATH = None
    sys.modules[envs.__name__] = envs

    logger_module = types.ModuleType("vllm.logger")
    logger_module.init_logger = logging.getLogger
    sys.modules[logger_module.__name__] = logger_module

    system_utils = types.ModuleType("vllm.utils.system_utils")
    system_utils.decorate_logs = lambda *_args: None
    system_utils.set_process_title = lambda *_args: None

    def kill_process_tree(pid: int) -> None:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    system_utils.kill_process_tree = kill_process_tree
    sys.modules[system_utils.__name__] = system_utils

    engine_utils = types.ModuleType("vllm.v1.engine.utils")
    engine_utils.get_engine_process_shutdown_timeout = (
        lambda request_timeout, _manager_timeout: request_timeout
    )
    sys.modules[engine_utils.__name__] = engine_utils

    launcher = types.ModuleType("vllm.entrypoints.launchers.launcher")

    import uvicorn

    class MinimalNoSignalServer(uvicorn.Server):
        @contextlib.contextmanager
        def capture_signals(self):
            yield

    launcher.NoSignalServer = MinimalNoSignalServer
    sys.modules[launcher.__name__] = launcher

    server_utils = types.ModuleType(
        "vllm.entrypoints.launchers.utils.server_utils"
    )
    server_utils.get_uvicorn_log_config = lambda _args: None
    sys.modules[server_utils.__name__] = server_utils


def load_subject_module(subject_source: Path) -> types.ModuleType:
    install_vllm_import_stubs()
    name = "vllm.entrypoints.launchers.dp_supervisor"
    spec = importlib.util.spec_from_file_location(name, subject_source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load subject: {subject_source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_subject(case: str, subject_source: Path) -> None:
    dp_sup = load_subject_module(subject_source)

    real_supervisor = dp_sup.DPSupervisor

    class ProbeSupervisor(real_supervisor):
        instance: "ProbeSupervisor | None" = None

        def __init__(self, args: argparse.Namespace) -> None:
            super().__init__(args)
            ProbeSupervisor.instance = self
            self.server_started = False
            self.signal_sent = False
            self.handled_signals: list[str] = []
            self.probe_failure_triggered = False
            self._signal_task: asyncio.Task[None] | None = None
            self._release_file = Path(f"/tmp/gate0-release-{os.getpid()}")
            self._release_file.unlink(missing_ok=True)

        def _start_children(self) -> None:
            if case == "abnormal-child-exit-before-ready":
                code = "import os,time; time.sleep(0.15); os._exit(17)"
            elif case == "abnormal-child-exit-after-ready":
                code = (
                    "import os,time,pathlib; "
                    f"p=pathlib.Path({str(self._release_file)!r}); "
                    "deadline=time.monotonic()+10; "
                    "\nwhile not p.exists() and time.monotonic()<deadline: "
                    "time.sleep(0.01)\n"
                    "os._exit(17 if p.exists() else 18)"
                )
            else:
                code = "import time; time.sleep(30)"
            child = subprocess.Popen(
                [sys.executable, "-c", code],
                start_new_session=True,
            )
            self._processes = [PopenProcessAdapter(child)]
            if case == "intentional-sigterm-after-ready":
                self._signal_task = asyncio.create_task(
                    self._send_signal_after_server_start()
                )

        async def _send_signal_after_server_start(self) -> None:
            while not self.server_started:
                await asyncio.sleep(0.01)
            self.signal_sent = True
            os.kill(os.getpid(), signal.SIGTERM)

        def _handle_signal(self, signum: int) -> None:
            self.handled_signals.append(signal.Signals(signum).name)
            super()._handle_signal(signum)

        async def _start_server(self):
            server, task = await super()._start_server()
            self.server_started = True
            if case == "abnormal-child-exit-after-ready":
                self._release_file.write_text("release\n")
            return server, task

        async def _probe_all_children(self) -> None:
            if case == "abnormal-child-exit-before-ready":
                await self._shutdown_event.wait()
                return
            await super()._probe_all_children()

    async def fake_probe_endpoint(*_args, **_kwargs) -> bool:
        instance = ProbeSupervisor.instance
        if (
            case == "probe-failure-after-ready"
            and instance is not None
            and instance.server_started
        ):
            instance.probe_failure_triggered = True
            return False
        return True

    dp_sup.DPSupervisor = ProbeSupervisor
    dp_sup._probe_endpoint = fake_probe_endpoint
    returned = False
    error: str | None = None
    try:
        dp_sup.run_dp_supervisor(make_args())
        returned = True
    except BaseException as exc:  # retained as closed-shape evidence
        error = f"{type(exc).__name__}: {exc}"

    instance = ProbeSupervisor.instance
    child_exitcode = None
    if instance is not None and instance._processes:
        child_exitcode = instance._processes[0].exitcode
        instance._release_file.unlink(missing_ok=True)
    print(
        SUBJECT_PREFIX
        + json.dumps(
            {
                "case": case,
                "child_exitcode": child_exitcode,
                "entry_returned": returned,
                "error": error,
                "probe_failure_triggered": (
                    instance.probe_failure_triggered if instance else False
                ),
                "handled_signals": (
                    instance.handled_signals if instance else []
                ),
                "server_started": instance.server_started if instance else False,
                "signal_sent": instance.signal_sent if instance else False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not returned:
        raise SystemExit(70)


def parse_subject(stdout: str) -> dict[str, Any] | None:
    records = []
    for line in stdout.splitlines():
        if SUBJECT_PREFIX in line:
            records.append(json.loads(line.split(SUBJECT_PREFIX, 1)[1]))
    if len(records) != 1:
        return None
    return records[0]


def run_cell(subject_source: Path, case: str, timeout: float) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--subject",
        case,
        "--subject-source",
        str(subject_source),
    ]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=subject_source.parent,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        subject = parse_subject(completed.stdout)
        return {
            "case": case,
            "process_returncode": completed.returncode,
            "timed_out": False,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "subject": subject,
            "stdout_sha256": hashlib.sha256(
                completed.stdout.encode("utf-8")
            ).hexdigest(),
            "stderr_sha256": hashlib.sha256(
                completed.stderr.encode("utf-8")
            ).hexdigest(),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "case": case,
            "process_returncode": None,
            "timed_out": True,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "subject": None,
            "stdout_sha256": hashlib.sha256(
                (exc.stdout or "").encode("utf-8")
                if isinstance(exc.stdout, str)
                else (exc.stdout or b"")
            ).hexdigest(),
            "stderr_sha256": hashlib.sha256(
                (exc.stderr or "").encode("utf-8")
                if isinstance(exc.stderr, str)
                else (exc.stderr or b"")
            ).hexdigest(),
        }


def run_campaign(
    source: Path,
    subject_commit: str,
    output: Path,
    timeout: float,
) -> None:
    if not source.is_file():
        raise SystemExit(f"missing subject source: {source}")
    protocol = Path(__file__).with_name("GATE0_PROTOCOL.md")
    results = [
        run_cell(source, "intentional-sigterm-after-ready", timeout),
        run_cell(source, "abnormal-child-exit-before-ready", timeout),
        run_cell(source, "abnormal-child-exit-after-ready", timeout),
        run_cell(source, "probe-failure-after-ready", timeout),
    ]
    summary = {
        "schema_version": 1,
        "gate": "vllm-dp-supervisor-exit-gate0",
        "vllm_commit": subject_commit,
        "subject_source_sha256": sha256_file(source),
        "protocol_sha256": sha256_file(protocol),
        "runner_sha256": sha256_file(Path(__file__)),
        "python": sys.version.split()[0],
        "fault_tolerance_enabled": False,
        "timeout_seconds": timeout,
        "platform": {
            "machine": platform.machine(),
            "release": platform.release(),
            "system": platform.system(),
        },
        "cells": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject",
        choices=(
            "intentional-sigterm-after-ready",
            "abnormal-child-exit-before-ready",
            "abnormal-child-exit-after-ready",
            "probe-failure-after-ready",
        ),
    )
    parser.add_argument("--subject-source", type=Path)
    parser.add_argument("--subject-commit")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()
    if args.subject:
        if args.subject_source is None:
            parser.error("--subject-source is required in subject mode")
        run_subject(args.subject, args.subject_source.resolve())
        return
    if (
        args.subject_source is None
        or args.subject_commit is None
        or args.output is None
    ):
        parser.error(
            "--subject-source, --subject-commit, and --output are required"
        )
    run_campaign(
        args.subject_source.resolve(),
        args.subject_commit,
        args.output.resolve(),
        args.timeout,
    )


if __name__ == "__main__":
    main()
