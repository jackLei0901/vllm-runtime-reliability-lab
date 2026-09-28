"""K1: does ``vllm.v1.utils.shutdown(procs, timeout=0)`` leave any SIGTERM grace?

Inventory finding C1 (docs/ENGINE_LIVENESS_CONTRACT_INVENTORY_2026-09-27.md):
``--shutdown-timeout`` (default 0) is passed down to the EngineCore process
manager as its kill grace. From source, ``shutdown`` then sends SIGTERM and
immediately SIGKILLs the process tree. This script exercises the installed
helper against a fake EngineCore whose SIGTERM handler needs 100 ms of
teardown and owns one child process (standing in for a worker).

CPU only, Linux only. It does not start vLLM, touch a GPU, or prove any
downstream consequence (leaked shared memory, skipped NCCL teardown).
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import multiprocessing as mp
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

TEARDOWN_DELAY_S = 0.1
READY_TIMEOUT_S = 60.0
PINNED_VERSION_FRAGMENT = "gc8602c79"


def _grandchild() -> None:
    while True:
        time.sleep(0.2)


def _fake_engine(marker_dir: str, ready) -> None:
    """Stand-in EngineCore: needs TEARDOWN_DELAY_S after SIGTERM to finish."""
    directory = Path(marker_dir)
    worker = mp.get_context("fork").Process(target=_grandchild, daemon=False)
    worker.start()
    (directory / "worker.pid").write_text(str(worker.pid))

    def on_sigterm(signum, frame) -> None:
        (directory / "sigterm_received").write_text(str(time.monotonic()))
        time.sleep(TEARDOWN_DELAY_S)
        worker.terminate()
        worker.join(1)
        (directory / "teardown_done").write_text(str(time.monotonic()))
        os._exit(0)

    signal.signal(signal.SIGTERM, on_sigterm)
    ready.set()
    while True:
        time.sleep(0.2)


def _pid_alive(pid: int) -> bool:
    import psutil

    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _cleanup(pid: int) -> None:
    if _pid_alive(pid):
        os.kill(pid, signal.SIGKILL)


def _run_cell(shutdown, label: str, timeout: float | None) -> dict:
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix=f"k1-{label}-") as marker_dir:
        ready = ctx.Event()
        engine = ctx.Process(
            target=_fake_engine, args=(marker_dir, ready), name=f"FakeEngine-{label}"
        )
        engine.start()
        if not ready.wait(READY_TIMEOUT_S):
            engine.kill()
            return {"cell": label, "result": "apparatus_not_ready"}
        directory = Path(marker_dir)
        worker_pid = int((directory / "worker.pid").read_text())

        started = time.monotonic()
        shutdown([engine], timeout=timeout)
        returned = time.monotonic() - started
        engine.join(5)

        worker_alive = _pid_alive(worker_pid)
        _cleanup(worker_pid)
        return {
            "cell": label,
            "timeout_arg": timeout,
            "shutdown_returned_after_s": round(returned, 3),
            "engine_exitcode": engine.exitcode,
            "engine_sigkilled": engine.exitcode == -signal.SIGKILL,
            "sigterm_handler_started": (directory / "sigterm_received").exists(),
            "teardown_done": (directory / "teardown_done").exists(),
            "worker_alive_after_shutdown": worker_alive,
        }


def _file_sha256(path: str | None) -> str | None:
    if path is None:
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _provenance() -> dict:
    import vllm
    import vllm.v1.utils as v1_utils

    record: dict = {
        "python": sys.version.split()[0],
        "vllm_version": getattr(vllm, "__version__", "unknown"),
        "v1_utils_sha256": _file_sha256(inspect.getsourcefile(v1_utils)),
        "shutdown_source_sha256": hashlib.sha256(
            inspect.getsource(v1_utils.shutdown).encode()
        ).hexdigest(),
    }
    record["pinned_version_match"] = PINNED_VERSION_FRAGMENT in record["vllm_version"]
    try:
        import dataclasses

        from vllm.config import VllmConfig

        field = {f.name: f for f in dataclasses.fields(VllmConfig)}["shutdown_timeout"]
        default = field.default
        record["default_shutdown_timeout"] = getattr(default, "default", default)
    except Exception as exc:  # recorded, not scored
        record["default_shutdown_timeout"] = f"unavailable: {type(exc).__name__}"
    try:
        import vllm.entrypoints.launcher as launcher

        record["launcher_passes_shutdown_timeout"] = (
            "engine_client.shutdown, timeout=timeout" in inspect.getsource(launcher)
        )
    except Exception as exc:
        unavailable = f"unavailable: {type(exc).__name__}"
        record["launcher_passes_shutdown_timeout"] = unavailable
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if not sys.platform.startswith("linux"):
        print("K1 requires Linux (POSIX signals, fork, psutil process tree).")
        return 2

    from vllm.v1.utils import shutdown

    provenance = _provenance()
    cells = [
        _run_cell(shutdown, "zero", 0),
        _run_cell(shutdown, "fallback_none", None),
        _run_cell(shutdown, "one_second", 1.0),
    ]
    by_label = {cell["cell"]: cell for cell in cells}

    controls_ok = all(
        by_label[label].get("teardown_done") is True
        and by_label[label].get("engine_exitcode") == 0
        for label in ("fallback_none", "one_second")
    )
    zero = by_label["zero"]
    zero_grace = (
        zero.get("teardown_done") is False
        and zero.get("engine_sigkilled") is True
        and zero.get("worker_alive_after_shutdown") is False
    )
    if not controls_ok:
        verdict = "apparatus_failed"
    elif zero_grace:
        verdict = "c1_zero_grace_confirmed"
    else:
        verdict = "c1_not_reproduced"

    for cell in cells:
        print("K1_CELL " + json.dumps(cell, sort_keys=True))
    print(
        "K1_RESULT "
        + json.dumps({"verdict": verdict, "provenance": provenance}, sort_keys=True)
    )
    return 0 if verdict != "apparatus_failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
