"""Bounded TP=2 V2 serving replay hold with per-rank activation witnesses.

Use only with the pinned protocol and an external hard timeout. Raw logs and
witness files remain in a fresh private directory; stdout is closed-shape.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from inflight_trace import read_rank_logs, score_triplet
from serving_stall_gate import (
    _create_arm,
    _has_gap,
    _private_directory,
    _rank_identities,
)
from v2_activation_witness import read_witness


def _witness_pair(directory: Path, identities: dict[int, tuple[int, int]]) -> dict:
    return {
        rank: read_witness(directory, identities[rank][0], rank) for rank in (0, 1)
    }


def _public_witness(pair: dict) -> dict:
    fields = (
        "manager_kind",
        "runner_v2",
        "configured_graph_mode",
        "breakable_enabled",
        "replay_calls",
        "armed_replay_calls",
        "full_cached_calls",
        "eligible_calls",
        "hold_entered",
    )
    return {rank: {field: pair[rank][field] for field in fields} for rank in (0, 1)}


def _unscored(mode: str, reason: str, binary_digest: str) -> int:
    print(
        "TP_V2_INFLIGHT_GATE "
        + json.dumps(
            {
                "schema": "tp-v2-inflight-gate-v1",
                "mode": mode,
                "inspector_binary_sha256": binary_digest,
                "inflight": {"result": "unscored", "reason": reason},
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 2


def _check_environment(private: Path) -> tuple[Path, Path, Path]:
    _private_directory(private)
    inspector = private / "inspector"
    witness = private / "witness"
    _private_directory(inspector)
    _private_directory(witness)
    if list(witness.iterdir()):
        raise ValueError("witness directory must start empty")
    arm, entered = private / "arm", private / "entered"
    if arm.exists() or entered.exists():
        raise ValueError("arm and entered markers must not pre-exist")
    expected = {
        "LLR_TP_ARM_FILE": str(arm),
        "LLR_TP_ENTER_FILE": str(entered),
        "LLR_TP_WITNESS_DIR": str(witness),
        "VLLM_PLUGINS": "llr_tp_v2_stall",
        "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
        "NCCL_DEBUG": "TRACE",
        "NCCL_DEBUG_SUBSYS": "INIT,PROFILE",
        "NCCL_DEBUG_FILE": str(private / "nccl.%p.log"),
        "NCCL_INSPECTOR_DUMP_DIR": str(inspector),
    }
    for name, value in expected.items():
        if os.environ.get(name) != value:
            raise ValueError(f"experiment environment mismatch: {name}")
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") is not None:
        raise ValueError("V2 must be selected by the pinned build, not forced")
    seconds = float(os.environ.get("LLR_TP_HOLD_SECONDS", "0"))
    if not 2 <= seconds <= 5:
        raise ValueError("hold duration is outside the preregistered bound")
    library = Path(os.environ.get("NCCL_PROFILER_PLUGIN", ""))
    if not library.is_absolute() or not library.is_file():
        raise ValueError("pinned Inspector library is unavailable")
    return arm, entered, witness


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--private-dir", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("control", "hold"))
    args = parser.parse_args()
    private = args.private_dir.resolve(strict=True)
    arm, entered, witness_dir = _check_environment(private)
    library = Path(os.environ["NCCL_PROFILER_PLUGIN"])
    library_digest = hashlib.sha256(library.read_bytes()).hexdigest()

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
    pattern = str(private / "nccl.*.log")
    try:
        identities = _rank_identities(pattern)
        before = read_rank_logs(pattern)
        witness_before = _witness_pair(witness_dir, identities)
    except (OSError, ValueError, UnicodeError):
        return _unscored(args.mode, "preflight_evidence_unavailable", library_digest)
    if any(witness_before[rank]["hold_entered"] for rank in (0, 1)):
        raise ValueError("hold occurred before the evaluation window")
    inspector_before = inspector_counts(str(private / "inspector"))
    if inspector_before.get("outcome") != "available":
        return _unscored(args.mode, "stock_inspector_unavailable", library_digest)
    if args.mode == "hold":
        _create_arm(arm)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            engine.generate,
            ["Say hello."],
            SamplingParams(max_tokens=16, temperature=0),
        )
        held_during = False
        during_error = False
        during = before
        inspector_during = inspector_before
        if args.mode == "hold":
            deadline = time.monotonic() + 15
            while not entered.exists() and not future.done() and time.monotonic() < deadline:
                time.sleep(0.05)
            if entered.exists():
                try:
                    entered_ns = int(entered.read_text(encoding="ascii"))
                    seconds = float(os.environ["LLR_TP_HOLD_SECONDS"])
                    deadline = min(
                        time.monotonic() + 1.5, entered_ns / 1e9 + seconds - 0.25
                    )
                    while time.monotonic() < deadline and not future.done():
                        during = read_rank_logs(pattern)
                        inspector_during = inspector_counts(str(private / "inspector"))
                        if inspector_during.get("outcome") != "available":
                            raise ValueError("stock snapshot unavailable")
                        held_during = (
                            time.monotonic_ns() - entered_ns < seconds * 1e9
                            and not future.done()
                        )
                        if held_during and _has_gap(before, during):
                            break
                        time.sleep(0.05)
                except (OSError, ValueError, UnicodeError):
                    during_error = True
        output = future.result(timeout=60)

    try:
        after = read_rank_logs(pattern)
        inspector_after = inspector_counts(str(private / "inspector"))
        identity_stable = identities == _rank_identities(pattern)
        witness_after = _witness_pair(witness_dir, identities)
    except (OSError, ValueError, UnicodeError):
        return _unscored(args.mode, "postflight_evidence_unavailable", library_digest)
    if inspector_after.get("outcome") != "available":
        return _unscored(args.mode, "stock_inspector_unavailable", library_digest)
    tokens = len(output[0].outputs[0].token_ids)
    witnessed_full = all(
        witness_after[rank]["replay_calls"] > 0
        and witness_after[rank]["full_cached_calls"] > 0
        for rank in (0, 1)
    )
    if args.mode == "control":
        okay = (
            tokens == 16
            and identity_stable
            and witnessed_full
            and not entered.exists()
            and all(not witness_after[rank]["hold_entered"] for rank in (0, 1))
            and all(witness_after[rank]["eligible_calls"] == 0 for rank in (0, 1))
        )
        result = {
            "result": "healthy_v2_full_replay_observed" if okay else "unscored",
            "reason": None if okay else "control_precondition_failed",
        }
    elif during_error:
        result = {"result": "unscored", "reason": "during_evidence_unavailable"}
    elif (
        not witnessed_full
        or not entered.exists()
        or witness_after[0]["hold_entered"]
        or not witness_after[1]["hold_entered"]
        or witness_after[1]["eligible_calls"] == 0
    ):
        result = {"result": "unscored", "reason": "v2_activation_unverified"}
    else:
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

    print(
        "TP_V2_INFLIGHT_GATE "
        + json.dumps(
            {
                "schema": "tp-v2-inflight-gate-v1",
                "mode": args.mode,
                "tokens": tokens,
                "identity_stable": identity_stable,
                "held_during_snapshot": held_during,
                "inspector_binary_sha256": library_digest,
                "activation": _public_witness(witness_after),
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
    return 0 if result["result"] in (
        "healthy_v2_full_replay_observed",
        "start_asymmetry_observed",
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
