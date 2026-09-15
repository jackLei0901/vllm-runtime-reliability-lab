#!/usr/bin/env python3
"""Deterministic CPU-only component probe for vLLM issue #53859."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from vllm.distributed.kv_events import EventBatch, ZmqEventPublisher

PR_SET_PTRACER = 0x59616D61
YAMA_SCOPE = Path("/proc/sys/kernel/yama/ptrace_scope")


def authorize_observer_from_env() -> str:
    """Authorize the declared observer and its descendants under Yama."""
    if not YAMA_SCOPE.exists():
        return "yama_absent"
    observer_pid = int(os.environ.get("DFX_OBSERVER_PID", "0"))
    if observer_pid <= 0:
        raise RuntimeError("DFX_OBSERVER_PID must identify the observer parent")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_PTRACER, observer_pid, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    return "pr_set_ptracer_parent"


class PausedZmqEventPublisher(ZmqEventPublisher):
    """Use production construction while pausing only the consumer loop."""

    def __init__(self) -> None:
        self.consumer_paused = threading.Event()
        self.release_consumer = threading.Event()
        super().__init__(
            data_parallel_rank=0,
            endpoint=f"inproc://stage0-{uuid.uuid4().hex}",
            max_queue_size=1,
        )

    def _publisher_thread(self) -> None:
        self.consumer_paused.set()
        if not self.release_consumer.wait(timeout=300):
            raise RuntimeError("controlled consumer release was not observed")
        super()._publisher_thread()


def queued_object(publisher: PausedZmqEventPublisher) -> Any:
    queue = publisher._event_queue
    with queue.mutex:
        if len(queue.queue) != 1:
            raise RuntimeError(f"expected one queued item, found {len(queue.queue)}")
        return queue.queue[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect", choices=("blocking", "nonblocking"), required=True)
    parser.add_argument("--observation-seconds", type=float, default=1.0)
    parser.add_argument("--hold-seconds", type=float, default=20.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    observer_authorization = authorize_observer_from_env()
    publisher = PausedZmqEventPublisher()
    second_started = threading.Event()
    second_returned = threading.Event()
    second_error: list[str] = []

    first = EventBatch(ts=1.0, events=[])
    second = EventBatch(ts=2.0, events=[])

    try:
        if not publisher.consumer_paused.wait(timeout=5):
            raise RuntimeError("consumer did not reach controlled pause")

        publisher.publish(first)

        def publish_second() -> None:
            second_started.set()
            try:
                publisher.publish(second)
            except Exception as exc:  # retained as a bounded type/message
                second_error.append(f"{type(exc).__name__}: {exc}")
            finally:
                second_returned.set()

        caller = threading.Thread(target=publish_second, name="stage0-publish-caller")
        caller.start()
        if not second_started.wait(timeout=5):
            raise RuntimeError("second publisher thread did not start")

        returned_inside_bound = second_returned.wait(args.observation_seconds)
        first_retained = queued_object(publisher) is first

        interim = {
            "schema_version": 1,
            "subject_pid": os.getpid(),
            "expectation": args.expect,
            "consumer_paused": publisher.consumer_paused.is_set(),
            "observation_seconds": args.observation_seconds,
            "observer_authorization": observer_authorization,
            "second_returned_inside_bound": returned_inside_bound,
            "first_batch_retained_before_release": first_retained,
            "second_error": second_error or None,
        }
        print(json.dumps({"stage0_interim": interim}, sort_keys=True), flush=True)

        expected_return = args.expect == "nonblocking"
        matched_before_release = (
            returned_inside_bound == expected_return
            and first_retained
            and not second_error
        )

        if args.expect == "blocking" and args.hold_seconds > 0:
            time.sleep(args.hold_seconds)

        publisher.release_consumer.set()
        caller.join(timeout=5)
        recovered_after_release = not caller.is_alive() and not second_error

        drain_deadline = time.monotonic() + 5
        while (
            publisher._event_queue.unfinished_tasks
            and time.monotonic() < drain_deadline
        ):
            time.sleep(0.01)
        queue_drained_after_release = publisher._event_queue.unfinished_tasks == 0

        result = {
            **interim,
            "matched_before_release": matched_before_release,
            "recovered_after_release": recovered_after_release,
            "second_publish_completed_after_release": recovered_after_release,
            "queue_drained_after_release": queue_drained_after_release,
            "verdict": "PASS"
            if (
                matched_before_release
                and recovered_after_release
                and queue_drained_after_release
            )
            else "FAIL",
        }
        print(json.dumps({"stage0_result": result}, sort_keys=True), flush=True)
        return 0 if result["verdict"] == "PASS" else 1
    finally:
        publisher.release_consumer.set()
        try:
            publisher.shutdown()
        finally:
            # This standalone probe owns the only sockets in this process.
            publisher._ctx.term()


if __name__ == "__main__":
    raise SystemExit(main())
