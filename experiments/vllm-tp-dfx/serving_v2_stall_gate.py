"""Bounded TP=2 V2 serving replay hold with per-rank activation witnesses.

Use only with the pinned protocol and an external hard timeout. Raw logs and
witness files remain in a fresh private directory; stdout is closed-shape.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata
from pathlib import Path

from inflight_trace_v2 import (
    read_rank_logs,
    save_private_snapshot,
    score_triplet,
    validate_control_history,
)
from serving_stall_gate import (
    _create_arm,
    _has_gap,
    _private_directory,
    _rank_identities,
)
from v2_activation_witness import read_witness

NCCL_LIBRARY_NAME = re.compile(r"[/\\]libnccl\.so(?:\.\d+)*$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_nccl_digest(
    identities: dict[int, tuple[int, int]], *, proc_root: Path = Path("/proc")
) -> str:
    paths = set()
    for rank in (0, 1):
        mapped = set()
        content = (proc_root / str(identities[rank][0]) / "maps").read_text(
            encoding="utf-8", errors="replace"
        )
        for line in content.splitlines():
            parts = line.split(maxsplit=5)
            if len(parts) == 6 and NCCL_LIBRARY_NAME.search(parts[5]):
                mapped.add(Path(parts[5]).resolve(strict=True))
        if len(mapped) != 1:
            raise ValueError("runtime NCCL library mapping is ambiguous")
        paths.update(mapped)
    if len(paths) != 1:
        raise ValueError("ranks loaded different NCCL libraries")
    return _sha256_file(next(iter(paths)))


def _software_build_identity(vllm_module: object, torch_module: object) -> dict:
    vllm_root = Path(vllm_module.__file__).resolve(strict=True).parent
    return {
        "vllm_version": metadata.version("vllm"),
        "vllm_init_sha256": _sha256_file(vllm_root / "__init__.py"),
        "vllm_v2_runner_sha256": _sha256_file(
            vllm_root / "v1" / "worker" / "gpu" / "model_runner.py"
        ),
        "vllm_graph_manager_sha256": _sha256_file(
            vllm_root / "v1" / "worker" / "gpu" / "cudagraph_utils.py"
        ),
        "torch_version": torch_module.__version__,
        "torch_git_version": torch_module.version.git_version,
        "torch_cuda_version": torch_module.version.cuda,
        "torch_c_sha256": _sha256_file(Path(torch_module._C.__file__)),
    }


def _active_plugin_digest() -> str:
    spec = importlib.util.find_spec("llr_tp_v2_stall")
    if spec is None or spec.origin is None:
        raise ValueError("active V2 stall plugin source is unavailable")
    return _sha256_file(Path(spec.origin))


def _control_identity(
    model: str,
    inspector_digest: str,
    nccl_digest: str,
    software_build: dict,
    plugin_digest: str,
    patch_digest: str,
) -> dict:
    return {
        "schema": "tp-v2-control-receipt-v3",
        "result": "healthy_v2_full_replay_observed",
        "model": model,
        "inspector_binary_sha256": inspector_digest,
        "runtime_nccl_sha256": nccl_digest,
        "software_build": software_build,
        "plugin_sha256": plugin_digest,
        "inspector_patch_sha256": patch_digest,
        "witness_reader_sha256": _sha256_file(
            Path(__file__).with_name("v2_activation_witness.py")
        ),
        "runner_sha256": _sha256_file(Path(__file__)),
        "parser_sha256": _sha256_file(
            Path(__file__).with_name("inflight_trace_v2.py")
        ),
    }


def _check_control_receipt(path: Path, expected: dict) -> str:
    if path.stat().st_size > 4096:
        raise ValueError("control receipt exceeded budget")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    after_digest = receipt.pop("control_after_sha256", None)
    if receipt != expected or not isinstance(after_digest, str) or not re.fullmatch(
        r"[0-9a-f]{64}", after_digest
    ):
        raise ValueError("healthy-control receipt identity mismatch")
    if _sha256_file(path.with_name("callbacks-after.json")) != after_digest:
        raise ValueError("healthy-control snapshot digest mismatch")
    used = path.with_name("control-receipt.used")
    descriptor = os.open(used, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    return _sha256_file(path)


def _write_control_receipt(path: Path, identity: dict) -> None:
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _witness_pair(directory: Path, identities: dict[int, tuple[int, int]]) -> dict:
    return {
        rank: read_witness(directory, identities[rank][0], rank) for rank in (0, 1)
    }


def _public_witness(pair: dict) -> dict:
    fields = (
        "manager_kind",
        "process_origin",
        "runner_v2",
        "configured_graph_mode",
        "breakable_enabled",
        "replay_calls",
        "armed_replay_calls",
        "full_cached_calls",
        "observed_full_cached_calls",
        "eligible_calls",
        "hold_entered",
    )
    return {rank: {field: pair[rank][field] for field in fields} for rank in (0, 1)}


def _control_passed(
    pair: dict, *, tokens: int, identity_stable: bool, entered: bool
) -> bool:
    return (
        tokens == 16
        and identity_stable
        and not entered
        and all(
            pair[rank]["observed_full_cached_calls"] > 0
            and pair[rank]["replay_calls"] > 0
            and pair[rank]["full_cached_calls"] > 0
            and pair[rank]["eligible_calls"] == 0
            and not pair[rank]["hold_entered"]
            for rank in (0, 1)
        )
    )


def _unscored(mode: str, reason: str, binary_digest: str) -> int:
    print(
        "TP_V2_INFLIGHT_GATE "
        + json.dumps(
            {
                "schema": "tp-v2-inflight-gate-v2",
                "mode": mode,
                "inspector_binary_sha256": binary_digest,
                "inflight": {"result": "unscored", "reason": reason},
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 2


def _check_environment(private: Path) -> tuple[Path, Path, Path, Path]:
    _private_directory(private)
    inspector = private / "inspector"
    witness = private / "witness"
    _private_directory(inspector)
    _private_directory(witness)
    if list(witness.iterdir()):
        raise ValueError("witness directory must start empty")
    arm, entered, observe = private / "arm", private / "entered", private / "observe"
    if arm.exists() or entered.exists() or observe.exists():
        raise ValueError("experiment markers must not pre-exist")
    expected = {
        "LLR_TP_ARM_FILE": str(arm),
        "LLR_TP_ENTER_FILE": str(entered),
        "LLR_TP_OBSERVE_FILE": str(observe),
        "LLR_TP_WITNESS_DIR": str(witness),
        "VLLM_PLUGINS": "llr_tp_v2_stall",
        "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
        "NCCL_DEBUG": "TRACE",
        "NCCL_DEBUG_SUBSYS": "INIT,PROFILE",
        "NCCL_DEBUG_FILE": str(private / "nccl.%p.log"),
        "NCCL_INSPECTOR_DUMP_DIR": str(inspector),
        "NCCL_INSPECTOR_ENABLE": "1",
        "NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS": "500",
        "NCCL_INSPECTOR_DUMP_VERBOSE": "1",
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
    return arm, entered, observe, witness


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--private-dir", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("control", "hold"))
    parser.add_argument("--control-receipt", type=Path)
    args = parser.parse_args()
    private = args.private_dir.resolve(strict=True)
    if args.mode == "control" and args.control_receipt is not None:
        parser.error("control mode creates its own receipt")
    if args.mode == "hold" and args.control_receipt is None:
        parser.error("hold mode requires a passed healthy-control receipt")
    arm, entered, observe, witness_dir = _check_environment(private)
    library = Path(os.environ["NCCL_PROFILER_PLUGIN"])
    library_digest = _sha256_file(library)

    from ras_graph_baseline import inspector_counts
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    try:
        software_build = _software_build_identity(vllm, torch)
        plugin_digest = _active_plugin_digest()
        patch_digest = _sha256_file(
            Path(__file__).with_name(
                "inspector-inflight-occurrence-v2.29.7-1.patch"
            )
        )
    except (OSError, ValueError, TypeError, metadata.PackageNotFoundError):
        return _unscored(args.mode, "software_identity_unavailable", library_digest)

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
        nccl_digest = _runtime_nccl_digest(identities)
        before = read_rank_logs(pattern)
        witness_before = _witness_pair(witness_dir, identities)
    except (OSError, ValueError, UnicodeError):
        return _unscored(args.mode, "preflight_evidence_unavailable", library_digest)
    control_identity = _control_identity(
        args.model,
        library_digest,
        nccl_digest,
        software_build,
        plugin_digest,
        patch_digest,
    )
    control_receipt_digest = None
    if args.mode == "hold":
        try:
            control_receipt_digest = _check_control_receipt(
                args.control_receipt, control_identity
            )
        except (OSError, ValueError, TypeError):
            return _unscored(args.mode, "healthy_control_not_verified", library_digest)
    try:
        before_digest = save_private_snapshot(private, "before", before)
    except (OSError, ValueError):
        return _unscored(args.mode, "private_snapshot_unavailable", library_digest)
    if any(witness_before[rank]["hold_entered"] for rank in (0, 1)):
        raise ValueError("hold occurred before the evaluation window")
    if any(witness_before[rank]["observed_full_cached_calls"] for rank in (0, 1)):
        raise ValueError("observed replay before the evaluation window")
    inspector_before = inspector_counts(str(private / "inspector"))
    if inspector_before.get("outcome") != "available":
        return _unscored(args.mode, "stock_inspector_unavailable", library_digest)
    _create_arm(observe)
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
        try:
            during_digest = save_private_snapshot(private, "during", during)
        except (OSError, ValueError):
            during_error = True
            during_digest = None
        output = future.result(timeout=60)

    try:
        after = read_rank_logs(pattern)
        inspector_after = inspector_counts(str(private / "inspector"))
        identity_stable = identities == _rank_identities(pattern)
        nccl_stable = nccl_digest == _runtime_nccl_digest(identities)
        witness_after = _witness_pair(witness_dir, identities)
    except (OSError, ValueError, UnicodeError):
        return _unscored(args.mode, "postflight_evidence_unavailable", library_digest)
    if inspector_after.get("outcome") != "available":
        return _unscored(args.mode, "stock_inspector_unavailable", library_digest)
    if not nccl_stable:
        return _unscored(args.mode, "runtime_nccl_changed", library_digest)
    if during_digest is None:
        return _unscored(args.mode, "private_snapshot_unavailable", library_digest)
    try:
        after_digest = save_private_snapshot(private, "after", after)
    except (OSError, ValueError):
        return _unscored(args.mode, "private_snapshot_unavailable", library_digest)
    snapshot_digests = {
        "before": before_digest,
        "during": during_digest,
        "after": after_digest,
    }
    tokens = len(output[0].outputs[0].token_ids)
    witnessed_full = all(
        witness_after[rank]["replay_calls"] > 0
        and witness_after[rank]["full_cached_calls"] > 0
        and witness_after[rank]["observed_full_cached_calls"] > 0
        for rank in (0, 1)
    )
    if args.mode == "control":
        okay = _control_passed(
            witness_after,
            tokens=tokens,
            identity_stable=identity_stable,
            entered=entered.exists(),
        )
        try:
            validate_control_history(before, after)
        except ValueError:
            result = {"result": "unscored", "reason": "callback_history_invalid"}
        else:
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

    if args.mode == "control" and result["result"] == "healthy_v2_full_replay_observed":
        try:
            receipt_path = private / "control-receipt.json"
            _write_control_receipt(
                receipt_path,
                {**control_identity, "control_after_sha256": after_digest},
            )
            control_receipt_digest = _sha256_file(receipt_path)
        except OSError:
            return _unscored(args.mode, "control_receipt_unavailable", library_digest)

    print(
        "TP_V2_INFLIGHT_GATE "
        + json.dumps(
            {
                "schema": "tp-v2-inflight-gate-v2",
                "mode": args.mode,
                "tokens": tokens,
                "identity_stable": identity_stable,
                "held_during_snapshot": held_during,
                "inspector_binary_sha256": library_digest,
                "runtime_nccl_sha256": nccl_digest,
                "software_build": software_build,
                "plugin_sha256": plugin_digest,
                "inspector_patch_sha256": control_identity["inspector_patch_sha256"],
                "control_receipt_sha256": control_receipt_digest,
                "private_snapshot_sha256": snapshot_digests,
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
