#!/usr/bin/env python3
"""CPU preflight for the real Stage 1 plugin and ZMQ publisher."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path

from stage1_campaign import (
    BASE_TREE,
    FIX_TREE,
    PLUGIN_ROOT,
    atomic_json,
    capture_stack,
    identity_is_live,
    load_queue_counts,
    load_ready,
    probe_environment,
    sha256_file,
    source_is_clean,
    source_tree,
    stop_process_group,
)

READY_SECONDS = 60.0
HERE = Path(__file__).resolve().parent


def subject(control_dir: Path) -> int:
    entries = list(
        importlib.metadata.entry_points(
            group="vllm.general_plugins", name="dfx_stage1_backpressure"
        )
    )
    if len(entries) != 1:
        raise RuntimeError("expected exactly one installed Stage 1 plugin")
    from vllm.plugins import load_general_plugins

    load_general_plugins()

    from vllm.distributed.kv_events import (
        AllBlocksCleared,
        EventBatch,
        ZmqEventPublisher,
    )

    publisher = ZmqEventPublisher(
        data_parallel_rank=0,
        endpoint=f"inproc://stage1a-{uuid.uuid4().hex}",
        max_queue_size=1,
    )
    second_returned = threading.Event()
    errors: list[str] = []
    first = EventBatch(ts=1.0, events=[AllBlocksCleared()])
    second = EventBatch(ts=2.0, events=[AllBlocksCleared()])
    try:
        ready = control_dir / f"ready-{os.getpid()}.json"
        deadline = time.monotonic() + READY_SECONDS
        while not ready.is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not ready.is_file():
            raise RuntimeError("publisher hook did not become ready")
        publisher.publish(first)

        def publish_second() -> None:
            try:
                atomic_json(
                    control_dir / f"second-started-{os.getpid()}.json",
                    {"pid": os.getpid()},
                )
                publisher.publish(second)
            except Exception as error:
                errors.append(type(error).__name__)
            finally:
                second_returned.set()

        caller = threading.Thread(
            target=publish_second, name="stage1a-scheduler-like-caller"
        )
        caller.start()
        if not (control_dir / "release").is_file():
            while not (control_dir / "release").is_file():
                time.sleep(0.05)
        caller.join(timeout=10)
        if caller.is_alive() or errors:
            return 1
        drain_deadline = time.monotonic() + 5
        while (
            publisher._event_queue.unfinished_tasks
            and time.monotonic() < drain_deadline
        ):
            time.sleep(0.01)
        if publisher._event_queue.unfinished_tasks:
            return 1
        return 0
    finally:
        try:
            publisher.shutdown()
        finally:
            publisher._ctx.term()


def wait_for_hook(control_dir: Path, process: subprocess.Popen[bytes]) -> list[Path]:
    deadline = time.monotonic() + READY_SECONDS
    while time.monotonic() < deadline and process.poll() is None:
        errors = sorted(control_dir.glob("hook-error-*.json"))
        ready = sorted(control_dir.glob("ready-*.json"))
        if errors:
            raise RuntimeError("plugin reported a bounded hook error")
        if len(ready) > 1:
            raise RuntimeError("more than one publisher hook was installed")
        if len(ready) == 1:
            return ready
        time.sleep(0.05)
    raise RuntimeError("plugin did not publish a ready record")


def wait_for_queue_state(
    control_dir: Path, pid: int, source_arm: str
) -> dict[str, int]:
    count_path = control_dir / f"queue-counts-{pid}.json"
    second_started = control_dir / f"second-started-{pid}.json"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if count_path.is_file() and second_started.is_file():
            counts = load_queue_counts(count_path)
            reached = counts["accepted_batch_count"] >= 1
            if source_arm == "fix":
                reached = reached and counts["dropped_batch_count"] >= 1
            if reached:
                return counts
        time.sleep(0.05)
    raise RuntimeError("scheduler-like caller did not reach the measured queue state")


def capture_expected_stack(
    py_spy: str, pid: int, private_dir: Path
) -> tuple[dict[str, object], int]:
    deadline = time.monotonic() + 10
    attempt = 0
    last = {"available": False, "match": None, "raw_sha256": None}
    while time.monotonic() < deadline:
        attempt += 1
        last = capture_stack(
            py_spy,
            pid,
            private_dir / f"subject-stack-{attempt}.txt",
        )
        if last["match"] is True:
            return last, attempt
        time.sleep(0.1)
    return last, attempt


def run(args: argparse.Namespace) -> int:
    expected_tree = BASE_TREE if args.source_arm == "base" else FIX_TREE
    if source_tree(args.server_workdir) != expected_tree:
        raise RuntimeError("source tree mismatch")
    if not source_is_clean(args.server_workdir):
        raise RuntimeError("source worktree is not clean")
    environment_record = probe_environment(args.python, args.server_workdir)

    args.private_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    control_dir = args.private_dir / "control"
    control_dir.mkdir()
    environment = os.environ.copy()
    environment.update(
        {
            "VLLM_PLUGINS": "dfx_stage1_backpressure",
            "PYTHONPATH": str(args.server_workdir.resolve()),
            "DFX_STAGE1_ENABLE": "1",
            "DFX_STAGE1_CONTROL_DIR": str(control_dir.resolve()),
            "DFX_STAGE1_OBSERVER_PID": str(os.getpid()),
        }
    )
    raw_log = args.private_dir / "subject.log"
    engine_identity: tuple[int, int] | None = None
    result_record: dict[str, object] | None = None
    passed = False
    with raw_log.open("wb") as stream:
        process = subprocess.Popen(
            [
                args.python,
                str(Path(__file__).resolve()),
                "--subject",
                "--control-dir",
                str(control_dir),
            ],
            cwd=args.server_workdir,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            ready_path = wait_for_hook(control_dir, process)[0]
            ready = load_ready(ready_path)
            plugin_path = (
                PLUGIN_ROOT / "src" / "dfx_stage1_backpressure" / "__init__.py"
            )
            expected_plugin_hash = sha256_file(plugin_path)
            if ready["plugin_sha256"] != expected_plugin_hash:
                raise RuntimeError("loaded plugin bytes do not match the lab plugin")
            if ready["kv_events_sha256"] != environment_record["tree_kv_events_sha256"]:
                raise RuntimeError("EngineCore kv_events bytes do not match the tree")
            engine_identity = (ready["pid"], ready["start_time_ticks"])
            if not identity_is_live(*engine_identity):
                raise RuntimeError("subject identity changed before capture")
            counts_before = wait_for_queue_state(
                control_dir, ready["pid"], args.source_arm
            )
            if args.source_arm == "base":
                time.sleep(1.0)
                stack, stack_attempt_count = capture_expected_stack(
                    args.py_spy,
                    ready["pid"],
                    args.private_dir,
                )
            else:
                stack = {"available": None, "match": None, "raw_sha256": None}
                stack_attempt_count = 0
            (control_dir / "release").touch()
            return_code = process.wait(timeout=30)
            counts_after = load_queue_counts(
                control_dir / f"queue-counts-{ready['pid']}.json"
            )
            expected_counts = (
                counts_before["accepted_batch_count"] >= 1
                and counts_before["dropped_batch_count"] == 0
                if args.source_arm == "base"
                else counts_before["accepted_batch_count"] >= 1
                and counts_before["dropped_batch_count"] >= 1
            )
            passed = (
                return_code == 0
                and expected_counts
                and (stack["match"] is True if args.source_arm == "base" else True)
                and isinstance(ready["yama_ptrace_scope"], int)
                and ready["yama_ptrace_scope"] >= 1
                and ready["authorization"] == "pr_set_ptracer_observer"
            )
            result_record = {
                "schema_version": 1,
                "source_arm": args.source_arm,
                "source_tree": expected_tree,
                "environment": environment_record,
                "implementation_sha256": {
                    "preflight": sha256_file(Path(__file__).resolve()),
                    "campaign": sha256_file(HERE / "stage1_campaign.py"),
                    "contract": sha256_file(HERE / "stage1_contract.py"),
                    "plugin": expected_plugin_hash,
                },
                "ready_authorization": ready["authorization"],
                "engine_core_kv_events_sha256": ready["kv_events_sha256"],
                "engine_core_mapped_worktree_binaries": ready[
                    "mapped_worktree_binaries"
                ],
                "yama_ptrace_scope": ready["yama_ptrace_scope"],
                "counts_before_release": counts_before,
                "counts_after_release": counts_after,
                "stack": stack,
                "stack_attempt_count": stack_attempt_count,
                "subject_return_code": return_code,
            }
        finally:
            (control_dir / "release").touch(exist_ok=True)
            cleanup = stop_process_group(process, engine_identity)
    if result_record is None:
        raise RuntimeError("preflight ended without a structured result")
    passed = (
        passed
        and cleanup["process_group_gone"] is True
        and cleanup["engine_core_gone"] is True
    )
    result_record["cleanup"] = cleanup
    result_record["verdict"] = "PASS" if passed else "FAIL"
    atomic_json(args.summary, result_record)
    return 0 if passed else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", action="store_true")
    parser.add_argument("--control-dir", type=Path)
    parser.add_argument("--source-arm", choices=("base", "fix"))
    parser.add_argument("--server-workdir", type=Path)
    parser.add_argument("--python")
    parser.add_argument("--py-spy", default="py-spy")
    parser.add_argument("--private-dir", type=Path)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.subject:
        if args.control_dir is None:
            raise SystemExit("--control-dir is required in subject mode")
        return subject(args.control_dir)
    required = (
        args.source_arm,
        args.server_workdir,
        args.python,
        args.private_dir,
        args.summary,
    )
    if any(value is None for value in required):
        raise SystemExit("outer mode requires source, Python, workdir and output paths")
    if os.name != "posix":
        raise SystemExit("Stage 1a requires Linux process groups and ptrace")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
