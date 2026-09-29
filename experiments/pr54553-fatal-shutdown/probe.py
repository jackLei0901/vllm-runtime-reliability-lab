"""CPU-only, fail-closed process check of PR #54553's real outer EngineCore path.

The fake EngineCore supplies the fault and teardown; run_engine_core is imported
unchanged from each pinned vLLM checkout. Raw child stderr remains in --out.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

BASE_PREFIX = "810bc3250c945829c64a745b4f695ddfd8f9a598"
PATCH_PREFIX = "41dddf7"
CASES = ("fatal_hold", "fatal_quick", "clean_quick")


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, encoding="utf-8"
    ).strip()


def _check_source(root: Path, expected: str) -> str:
    head = _git(root, "rev-parse", "HEAD")
    if not head.startswith(expected):
        raise ValueError(f"wrong commit in {root}: {head}")
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError(f"checkout is not clean in {root}")
    path = root / "vllm" / "v1" / "engine" / "core.py"
    if _git(root, "hash-object", str(path)) != _git(
        root, "rev-parse", "HEAD:vllm/v1/engine/core.py"
    ):
        raise ValueError(f"core.py differs from the commit in {root}")
    return head


def _mark(path: Path, marker: str) -> None:
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(fd, (marker + "\n").encode("ascii"))
    finally:
        os.close(fd)


def _child(args: argparse.Namespace) -> None:
    import vllm.v1.engine.core as core

    loaded = Path(core.__file__).resolve()
    expected = (args.src / "vllm" / "v1" / "engine" / "core.py").resolve()
    if loaded != expected:
        raise RuntimeError(f"imported unexpected core.py: {loaded}")
    run_engine_core = core.EngineCoreProc.run_engine_core
    _mark(args.markers, "source_verified")

    class FakeCore:
        def __init__(self, *unused: object, **kwargs: object) -> None:
            self.vllm_config = kwargs["vllm_config"]
            self.shutdown_state = core.EngineShutdownState.SHUTTING_DOWN
            self.input_queue = queue.Queue()

        def run_busy_loop(self) -> None:
            if args.case == "clean_quick":
                raise SystemExit(0)
            raise RuntimeError("controlled fatal fault")

        def _send_engine_dead(self) -> None:
            _mark(args.markers, "engine_dead_sent")

        def has_work(self) -> bool:
            return False

        def shutdown(self) -> None:
            _mark(args.markers, "teardown_entered")
            if args.case == "fatal_hold":
                threading.Event().wait(3600)
            else:
                time.sleep(0.1)
                _mark(args.markers, "teardown_completed")

    class FakeSignalCallback:
        def __init__(self, *unused: object) -> None:
            pass

        def stop(self) -> None:
            pass

    parallel = SimpleNamespace(
        data_parallel_size=1,
        numa_bind=False,
        reconfigure_for_independent_dp_rank=lambda: None,
    )
    config = SimpleNamespace(
        parallel_config=parallel, kv_transfer_config=None, shutdown_timeout=0
    )
    core.EngineCoreProc = FakeCore
    core.SignalCallback = FakeSignalCallback
    core.maybe_register_config_serialize_by_value = lambda: None
    core.set_process_title = lambda *unused: None
    core.maybe_init_worker_tracer = lambda *unused: None
    core.decorate_logs = lambda: None
    _mark(args.markers, "outer_path_entered")
    run_engine_core(vllm_config=config)


def _one_case(python: Path, src: Path, case: str, directory: Path) -> dict[str, object]:
    directory.mkdir(parents=True)
    markers = directory / "markers.txt"
    stderr_path = directory / "stderr.txt"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(src)
    environment["VLLM_TARGET_DEVICE"] = "cpu"
    environment["VLLM_ENGINE_SHUTDOWN_TIMEOUT_SECONDS"] = "1"
    with stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            [
                str(python),
                str(Path(__file__).resolve()),
                "child",
                "--src",
                str(src),
                "--case",
                case,
                "--markers",
                str(markers),
            ],
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=stderr,
        )
        try:
            setup_limit = time.monotonic() + 45
            while time.monotonic() < setup_limit:
                lines = markers.read_text().splitlines() if markers.exists() else []
                if "teardown_entered" in lines or process.poll() is not None:
                    break
                time.sleep(0.05)
            entered = "teardown_entered" in lines
            start = time.monotonic()
            if entered:
                try:
                    code = process.wait(timeout=4)
                    timed_out = False
                except subprocess.TimeoutExpired:
                    timed_out = True
                    process.kill()
                    code = process.wait(timeout=5)
            else:
                timed_out = False
                if process.poll() is None:
                    process.kill()
                code = process.wait(timeout=5)
            elapsed = round(time.monotonic() - start, 3)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
    lines = markers.read_text().splitlines() if markers.exists() else []
    return {
        "case": case,
        "exit_code": code,
        "parent_timeout": timed_out,
        "elapsed_after_teardown_s": elapsed,
        "markers": lines,
    }


def score(base: dict[str, dict], patch: dict[str, dict]) -> str:
    for cells in (base, patch):
        for case in CASES:
            row = cells[case]
            if not {
                "source_verified",
                "outer_path_entered",
                "teardown_entered",
            }.issubset(row["markers"]):
                return "unscored"
            if case != "clean_quick" and "engine_dead_sent" not in row["markers"]:
                return "unscored"
            if case != "fatal_hold" and "teardown_completed" not in row["markers"]:
                return "unscored"
    bh, ph = base["fatal_hold"], patch["fatal_hold"]
    if bh["parent_timeout"] and not ph["parent_timeout"] and ph["exit_code"] == 1:
        for cells in (base, patch):
            fq, cq = cells["fatal_quick"], cells["clean_quick"]
            if (
                fq["parent_timeout"]
                or cq["parent_timeout"]
                or fq["exit_code"] != 1
                or cq["exit_code"] != 0
            ):
                return "refuted"
        if 0.7 <= ph["elapsed_after_teardown_s"] <= 3:
            return "supported"
    return "refuted"


def _run(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise ValueError("output directory must not exist")
    identities = {
        "base": _check_source(args.base_src, BASE_PREFIX),
        "patch": _check_source(args.patch_src, PATCH_PREFIX),
    }
    for path in (args.base_python, args.patch_python):
        if not path.is_file():
            raise ValueError(f"missing interpreter: {path}")
    args.out.mkdir(parents=True)
    cells: dict[str, dict[str, dict]] = {}
    for name, python, src in (
        ("base", args.base_python, args.base_src),
        ("patch", args.patch_python, args.patch_src),
    ):
        cells[name] = {
            case: _one_case(python, src, case, args.out / name / case) for case in CASES
        }
    result = {"identities": identities, "cells": cells, "outcome": score(**cells)}
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"outcome": result["outcome"], "identities": identities}))


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    for name in ("base-src", "patch-src", "base-python", "patch-python", "out"):
        run.add_argument("--" + name, required=True, type=Path)
    child = commands.add_parser("child")
    child.add_argument("--src", required=True, type=Path)
    child.add_argument("--case", required=True, choices=CASES)
    child.add_argument("--markers", required=True, type=Path)
    args = parser.parse_args()
    _run(args) if args.command == "run" else _child(args)


if __name__ == "__main__":
    main()
