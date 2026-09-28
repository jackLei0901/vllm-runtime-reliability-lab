"""K2: can the executor's worker-shutdown schedule finish inside the client's grace?

Inventory finding C2 (docs/ENGINE_LIVENESS_CONTRACT_INVENTORY_2026-09-27.md):
``VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS`` (= x) bounds two nested levels. The
client gives the EngineCore process x seconds before killing its tree
(``core_client.py:433-436`` via ``vllm.v1.utils.shutdown``). Inside EngineCore,
``MultiprocExecutor._ensure_worker_termination`` waits x for workers, then
SIGTERMs them, waits 4 s, then SIGKILLs. The inner schedule needs about x + 4 s,
so from source it can never complete inside the outer x.

This script runs the two *installed* vLLM functions in that nesting: the parent
calls ``shutdown([fake_engine], timeout=outer)``; the fake engine, on SIGTERM,
calls the real ``_ensure_worker_termination`` on workers that are slow to exit
(they record SIGTERM but keep running). A control cell gives the outer level
enough time, proving the apparatus can observe a completed inner schedule.
The finite sampled x values demonstrate this particular slow-worker case;
the general timeout-budget relation is a source-level argument.

CPU only, Linux only. The fake engine omits EngineCore's own pre-executor
teardown; ``--engine-teardown-delay`` models it (default 0, the least
favourable case for the finding).
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
import threading
import time
from pathlib import Path

READY_TIMEOUT_S = 180.0  # the spawned fake engine imports vLLM before READY
WORKER_READY_TIMEOUT_S = 30.0
INNER_SIGTERM_WAIT_S = 4.0  # literal in _ensure_worker_termination at c8602c7
PINNED_VERSION_FRAGMENT = "gc8602c79"


def _slow_worker(marker_dir: str, ready) -> None:
    """Records SIGTERM, then keeps running: a worker slow to finish teardown."""
    directory = Path(marker_dir)

    def on_sigterm(signum, frame) -> None:
        path = directory / f"worker_sigterm.{os.getpid()}"
        path.write_text(str(time.monotonic()))

    signal.signal(signal.SIGTERM, on_sigterm)
    ready.set()
    while True:
        time.sleep(0.2)


def _fake_engine(
    marker_dir: str, inner_x: int, teardown_delay: float, n_workers: int, ready
) -> None:
    os.environ["VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS"] = str(inner_x)
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    directory = Path(marker_dir)
    ctx = mp.get_context("fork")
    worker_ready = [ctx.Event() for _ in range(n_workers)]
    workers = [
        ctx.Process(
            target=_slow_worker,
            args=(marker_dir, worker_ready[i]),
            name=f"FakeWorker-{i}",
        )
        for i in range(n_workers)
    ]
    for worker in workers:
        worker.start()
    (directory / "worker.pids").write_text(
        "\n".join(str(worker.pid) for worker in workers)
    )
    if not all(event.wait(WORKER_READY_TIMEOUT_S) for event in worker_ready):
        for worker in workers:
            if worker.is_alive():
                worker.kill()
        os._exit(2)
    (directory / "worker_handlers_ready").write_text(str(n_workers))

    # Like EngineCore, the handler only records the request; the main thread
    # performs teardown.
    requested = threading.Event()
    signal.signal(signal.SIGTERM, lambda signum, frame: requested.set())
    ready.set()
    while not requested.wait(0.05):
        pass
    (directory / "engine_sigterm").write_text(str(time.monotonic()))
    time.sleep(teardown_delay)
    (directory / "inner_start").write_text(str(time.monotonic()))
    MultiprocExecutor._ensure_worker_termination(workers)
    (directory / "inner_done").write_text(str(time.monotonic()))
    os._exit(0)


def _pid_alive(pid: int) -> bool:
    import psutil

    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _offset(path: Path, origin: float) -> float | None:
    if not path.exists():
        return None
    return round(float(path.read_text()) - origin, 3)


def _wait_ready(engine, ready) -> bool:
    deadline = time.monotonic() + READY_TIMEOUT_S
    while time.monotonic() < deadline:
        if ready.wait(0.1):
            return True
        if not engine.is_alive():
            return False
    return False


def _run_cell(
    shutdown,
    label: str,
    inner_x: int,
    outer: float,
    delay: float,
    n_workers: int,
    engine_start_method: str,
) -> dict:
    ctx = mp.get_context(engine_start_method)
    with tempfile.TemporaryDirectory(prefix=f"k2-{label}-") as marker_dir:
        directory = Path(marker_dir)
        ready = ctx.Event()
        engine = ctx.Process(
            target=_fake_engine,
            args=(marker_dir, inner_x, delay, n_workers, ready),
            name=f"FakeEngine-{label}",
        )
        engine.start()
        if not _wait_ready(engine, ready):
            if engine.is_alive():
                engine.kill()
            engine.join(5)
            pids_file = directory / "worker.pids"
            if pids_file.exists():
                for pid in (int(line) for line in pids_file.read_text().split()):
                    if _pid_alive(pid):
                        os.kill(pid, signal.SIGKILL)
            return {"cell": label, "result": "apparatus_not_ready"}
        worker_pids = [
            int(line) for line in (directory / "worker.pids").read_text().split()
        ]

        origin = time.monotonic()
        shutdown([engine], timeout=outer)
        outer_returned = round(time.monotonic() - origin, 3)
        engine.join(5)

        worker_sigterms = [
            _offset(path, origin) for path in directory.glob("worker_sigterm.*")
        ]
        leftover = [pid for pid in worker_pids if _pid_alive(pid)]
        for pid in leftover:
            os.kill(pid, signal.SIGKILL)
        return {
            "cell": label,
            "engine_start_method": engine_start_method,
            "inner_x_s": inner_x,
            "outer_grace_s": outer,
            "engine_teardown_delay_s": delay,
            "outer_returned_s": outer_returned,
            "engine_exitcode": engine.exitcode,
            "engine_sigkilled_by_outer": engine.exitcode == -signal.SIGKILL,
            "worker_handlers_ready": (
                (directory / "worker_handlers_ready").read_text() == str(n_workers)
                if (directory / "worker_handlers_ready").exists()
                else False
            ),
            "engine_sigterm_s": _offset(directory / "engine_sigterm", origin),
            "inner_start_s": _offset(directory / "inner_start", origin),
            "inner_sigterm_step_reached": bool(worker_sigterms),
            "first_worker_sigterm_s": min(worker_sigterms) if worker_sigterms else None,
            "inner_done_s": _offset(directory / "inner_done", origin),
            "inner_schedule_completed": (directory / "inner_done").exists(),
            "workers_left_alive": len(leftover),
        }


def _file_sha256(path: str | None) -> str | None:
    if path is None:
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _provenance() -> dict:
    import vllm
    import vllm.v1.executor.multiproc_executor as executor_module
    import vllm.v1.utils as v1_utils

    termination_source = inspect.getsource(
        executor_module.MultiprocExecutor._ensure_worker_termination
    )
    record = {
        "python": sys.version.split()[0],
        "vllm_version": getattr(vllm, "__version__", "unknown"),
        "v1_utils_sha256": _file_sha256(inspect.getsourcefile(v1_utils)),
        "multiproc_executor_sha256": _file_sha256(
            inspect.getsourcefile(executor_module)
        ),
        "inner_sigterm_wait_literal_present": (
            f"wait_for_termination(active_procs(), {int(INNER_SIGTERM_WAIT_S)})"
            in termination_source
        ),
    }
    record["pinned_version_match"] = PINNED_VERSION_FRAGMENT in record["vllm_version"]
    return record


def classify_cells(cells: list[dict]) -> str:
    """Never score a nested budget before both levels demonstrably ran."""
    if len(cells) < 2 or cells[0].get("cell") != "control":
        return "apparatus_failed"
    control, nested = cells[0], cells[1:]
    if not all(
        cell.get("worker_handlers_ready") is True
        and cell.get("engine_sigterm_s") is not None
        and cell.get("inner_start_s") is not None
        for cell in cells
    ):
        return "apparatus_failed"
    if not (
        control.get("inner_schedule_completed") is True
        and control.get("inner_sigterm_step_reached") is True
        and control.get("engine_exitcode") == 0
    ):
        return "apparatus_failed"
    if all(
        cell.get("inner_schedule_completed") is False
        and cell.get("engine_sigkilled_by_outer") is True
        for cell in nested
    ):
        return "c2_tested_nested_budget_shortfall"
    return "c2_not_reproduced"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inner-x", type=int, nargs="+", default=[1, 2, 5])
    parser.add_argument("--engine-teardown-delay", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--engine-start-method",
        choices=("spawn", "fork"),
        default="spawn",
        help="Use fork only for this CPU-only Linux harness on a memory-limited host.",
    )
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        print("K2 requires Linux (POSIX signals, fork, psutil process tree).")
        return 2
    if (
        args.engine_teardown_delay < 0
        or not 1 <= args.workers <= 8
        or any(not 1 <= value <= 30 for value in args.inner_x)
    ):
        parser.error("delay >= 0, workers in [1, 8], and every inner x in [1, 30]")

    from vllm.v1.utils import shutdown

    provenance = _provenance()
    delay = args.engine_teardown_delay
    cells = []
    # Control: outer grace comfortably exceeds x + 4 s + delay.
    control_x = min(args.inner_x)
    control_outer = control_x + INNER_SIGTERM_WAIT_S + delay + 6.0
    cells.append(
        _run_cell(
            shutdown,
            "control",
            control_x,
            control_outer,
            delay,
            args.workers,
            args.engine_start_method,
        )
    )
    # Nested: the client's grace equals the executor's first wait, as in vLLM.
    for inner_x in args.inner_x:
        cells.append(
            _run_cell(
                shutdown,
                f"nested_x{inner_x}",
                inner_x,
                inner_x,
                delay,
                args.workers,
                args.engine_start_method,
            )
        )

    verdict = classify_cells(cells)

    for cell in cells:
        print("K2_CELL " + json.dumps(cell, sort_keys=True))
    print(
        "K2_RESULT "
        + json.dumps({"verdict": verdict, "provenance": provenance}, sort_keys=True)
    )
    return 0 if verdict != "apparatus_failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
