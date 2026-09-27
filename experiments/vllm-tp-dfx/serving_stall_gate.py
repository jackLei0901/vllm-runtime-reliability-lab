"""Bounded TP=2 serving hold: private callbacks versus completed JSON.

The plugin, NCCL patch, exact model and source revision are preregistered in
VLLM_TP_INFLIGHT_SERVING_GATE_2026-09-27.md. Use an external hard timeout.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import stat
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from inflight_trace import MAX_LOG_BYTES, Event, read_rank_logs, score_triplet

_LOG_NAME = re.compile(r"^nccl\.(\d+)\.log$")
_RANK_LINE = re.compile(r"\[(\d+)\] NCCL INFO")


def _start_ticks(pid: int) -> int:
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    closing = raw.rfind(")")
    if closing < 0:
        raise ValueError("process identity unavailable")
    return int(raw[closing + 2 :].split()[19])


def _rank_identities(pattern: str) -> dict[int, tuple[int, int]]:
    identities: dict[int, tuple[int, int]] = {}
    paths = glob.glob(pattern)
    if len(paths) != 2:
        raise ValueError("expected exactly two NCCL logs")
    for name in paths:
        path = Path(name)
        match = _LOG_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError("NCCL log filename has no PID binding")
        pid = int(match[1])
        if path.stat().st_size > MAX_LOG_BYTES:
            raise ValueError("NCCL debug log exceeded private budget")
        ranks = set(_RANK_LINE.findall(path.read_text(errors="replace")))
        if len(ranks) != 1:
            raise ValueError("NCCL rank header ambiguous")
        rank = int(next(iter(ranks)))
        if rank in identities:
            raise ValueError("duplicate rank identity")
        identities[rank] = (pid, _start_ticks(pid))
    if set(identities) != {0, 1}:
        raise ValueError("rank identity set incomplete")
    if identities[0][0] == identities[1][0]:
        raise ValueError("two ranks share one PID")
    return identities


def _started(events: tuple[Event, ...]) -> dict[tuple[str, int], int]:
    counts: dict[tuple[str, int], int] = {}
    for event in events:
        if event.kind == "kernel_ch_start":
            counts[event.key] = counts.get(event.key, 0) + 1
    return counts


def _has_gap(before: dict, during: dict) -> bool:
    keys = {event.key for rank in (0, 1) for event in during[rank]}
    for key in keys:
        old = [_started(before[rank]).get(key, 0) for rank in (0, 1)]
        new = [_started(during[rank]).get(key, 0) for rank in (0, 1)]
        if old[0] == old[1] and new[0] > old[0] and new[1] == old[1]:
            return True
    return False


def _private_directory(path: Path) -> None:
    metadata = path.stat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("private evidence target is not a directory")
    if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise ValueError("private evidence directory must be owner-only")


def _create_arm(path: Path) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("control", "hold"), required=True)
    args = parser.parse_args()

    private = args.private_dir.resolve(strict=True)
    _private_directory(private)
    arm = private / "arm"
    entered = private / "entered"
    expected_environment = {
        "LLR_TP_ARM_FILE": str(arm),
        "LLR_TP_ENTER_FILE": str(entered),
        "VLLM_PLUGINS": "llr_tp_stall",
        "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
    }
    for name, value in expected_environment.items():
        if os.environ.get(name) != value:
            raise ValueError(f"experiment environment mismatch: {name}")
    if arm.exists() or entered.exists():
        raise ValueError("arm and entered markers must not pre-exist")
    hold_seconds = float(os.environ.get("LLR_TP_HOLD_SECONDS", "0"))
    if not 2 <= hold_seconds <= 5:
        raise ValueError("hold duration is outside the preregistered bound")
    debug_pattern = str(private / "nccl.*.log")
    inspector_dir = private / "inspector"
    _private_directory(inspector_dir)

    from ras_graph_baseline import inspector_counts
    from vllm import LLM, SamplingParams

    engine = LLM(
        model=args.model,
        tensor_parallel_size=2,
        disable_custom_all_reduce=True,
        gpu_memory_utilization=0.7,
        max_model_len=512,
        max_num_seqs=2,
        enforce_eager=False,
    )
    identities = _rank_identities(debug_pattern)
    before = read_rank_logs(debug_pattern)
    inspector_before = inspector_counts(str(inspector_dir))
    if args.mode == "hold":
        _create_arm(arm)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            engine.generate,
            ["Say hello."],
            SamplingParams(max_tokens=16, temperature=0),
        )
        held_during = False
        during = before
        inspector_during = inspector_before
        if args.mode == "hold":
            deadline = time.monotonic() + 15
            while not entered.exists() and not future.done() and time.monotonic() < deadline:
                time.sleep(0.05)
            if entered.exists():
                entered_ns = int(entered.read_text(encoding="ascii"))
                deadline = min(time.monotonic() + 1.5, entered_ns / 1e9 + hold_seconds - 0.25)
                while time.monotonic() < deadline and not future.done():
                    during = read_rank_logs(debug_pattern)
                    inspector_during = inspector_counts(str(inspector_dir))
                    held_during = (
                        time.monotonic_ns() - entered_ns < hold_seconds * 1e9
                        and not future.done()
                    )
                    if held_during and _has_gap(before, during):
                        break
                    time.sleep(0.05)
        output = future.result(timeout=60)

    after = read_rank_logs(debug_pattern)
    inspector_after = inspector_counts(str(inspector_dir))
    identity_stable = identities == _rank_identities(debug_pattern)
    tokens = len(output[0].outputs[0].token_ids)
    if args.mode == "hold":
        try:
            result = score_triplet(
                before,
                during,
                after,
                held_during_snapshot=held_during and identity_stable,
                request_completed_after_release=tokens == 16,
            )
        except ValueError:
            result = {"result": "unscored", "reason": "callback_history_invalid"}
    else:
        control_ok = tokens == 16 and identity_stable and not entered.exists()
        result = {
            "result": "healthy_control_complete" if control_ok else "unscored",
            "reason": None if control_ok else "control_precondition_failed",
        }
    print(
        "TP_INFLIGHT_SERVING_GATE "
        + json.dumps(
            {
                "schema": "tp-inflight-serving-gate-v0",
                "mode": args.mode,
                "tokens": tokens,
                "identity_stable": identity_stable,
                "held_during_snapshot": held_during,
                "inflight": result,
                "inspector_completed_records": {
                    "before": inspector_before,
                    "during": inspector_during,
                    "after": inspector_after,
                },
                "stock_same_key_comparison": "not_scored",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result["result"] in ("start_asymmetry_observed", "healthy_control_complete") else 2


if __name__ == "__main__":
    raise SystemExit(main())
