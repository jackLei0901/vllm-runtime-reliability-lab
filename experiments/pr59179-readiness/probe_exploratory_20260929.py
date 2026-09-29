"""CPU-only check of PR #59179's DP dummy-batch readiness semantics.

Run this in a Python environment that can import the pinned vLLM checkout.
No server, model, CUDA device, or distributed process group is started.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import importlib.util
import inspect
import json
import queue
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

PIN = "b3687914aa7a1ded4bde8af67a35e17ac6ef2f95"
ARCHIVE_SHA256 = "e6dcdc927b75f9442019cd2fa5de1661637fc656dfdbead43cfe2ac1107d1f5f"
SOURCE_FILES = (
    "vllm/v1/engine/core.py",
    "vllm/v1/engine/core_client.py",
    "vllm/v1/engine/__init__.py",
    "vllm/v1/fault_tolerance/engine_core_sentinel.py",
)


def preflight(source: Path, archive: Path | None = None) -> dict[str, str]:
    if archive is None:
        head = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
        if head != PIN:
            raise ValueError(f"checkout is {head}, expected {PIN}")
        changed = subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain", "--", *SOURCE_FILES],
            text=True,
        ).strip()
        if changed:
            raise ValueError("pinned readiness source files have local changes")
        identity = {"source_kind": "git_checkout", "source_commit": head}
    else:
        archive = archive.resolve()
        with archive.open("rb") as archive_file:
            archive_digest = hashlib.file_digest(archive_file, "sha256").hexdigest()
        if archive_digest != ARCHIVE_SHA256:
            raise ValueError("source archive digest does not match the frozen receipt")
        prefix = f"vllm-{PIN}/"
        with tarfile.open(archive, "r:gz") as snapshot:
            for relative in SOURCE_FILES:
                member = snapshot.extractfile(prefix + relative)
                if member is None:
                    raise ValueError(f"pinned archive lacks {relative}")
                archived_digest = hashlib.sha256(member.read()).digest()
                disk_digest = hashlib.sha256((source / relative).read_bytes()).digest()
                if archived_digest != disk_digest:
                    raise ValueError(
                        f"extracted source differs from archive: {relative}"
                    )
        identity = {
            "source_kind": "github_commit_archive",
            "source_commit": PIN,
            "archive_sha256": archive_digest,
        }
    sys.path.insert(0, str(source))
    return identity


def run(
    source: Path,
    archive: Path | None = None,
    binary_package_dir: Path | None = None,
) -> dict[str, object]:
    identity = preflight(source, archive)
    if binary_package_dir is not None:
        binary_package_dir = binary_package_dir.resolve()
        if not binary_package_dir.is_dir():
            raise ValueError("binary package directory is missing")
        extensions = list(binary_package_dir.glob("_C_stable_libtorch*.so"))
        if len(extensions) != 1:
            raise ValueError("expected one existing vLLM stable ABI extension")
        with extensions[0].open("rb") as extension_file:
            extension_digest = hashlib.file_digest(extension_file, "sha256")
        spec = importlib.util.spec_from_file_location(
            "vllm",
            source / "vllm" / "__init__.py",
            submodule_search_locations=[str(source / "vllm"), str(binary_package_dir)],
        )
        if spec is None or spec.loader is None:
            raise ValueError("could not load pinned vLLM package")
        vllm = importlib.util.module_from_spec(spec)
        sys.modules["vllm"] = vllm
        spec.loader.exec_module(vllm)
        identity["binary_source"] = "existing_precompiled_venv"
        identity["binary_extension_sha256"] = extension_digest.hexdigest()
    from vllm.v1.engine import EngineCoreReadyState
    from vllm.v1.engine.core import DPEngineCoreProc
    from vllm.v1.engine.core_client import AsyncMPClient, EngineCoreReadyProgress

    for obj, relative in (
        (DPEngineCoreProc, SOURCE_FILES[0]),
        (AsyncMPClient, SOURCE_FILES[1]),
    ):
        if Path(inspect.getfile(obj)).resolve() != (source / relative).resolve():
            raise ValueError(f"{obj.__name__} was imported from the wrong checkout")

    # Model one admitted but unschedulable request, such as a remote-KV wait.
    # The actual DP loop, progress publisher and frontend consumer remain real.
    core = object.__new__(DPEngineCoreProc)
    core.enable_fault_tolerance = False
    core.engines_running = True
    core.eep_scaling_state = None
    core.has_coordinator = True
    core.dp_rank = 1
    core.batch_queue = None
    core.scheduler = SimpleNamespace(
        has_requests=Mock(return_value=True),
        has_unfinished_requests=Mock(return_value=True),
    )
    core.model_executor = SimpleNamespace(is_sleeping=False)
    core.is_sleeping = Mock(return_value=False)
    core._process_input_queue = Mock()
    core._maybe_publish_request_counts = Mock()
    core._has_global_unfinished_reqs = Mock(return_value=True)
    core.step_fn = Mock(return_value=(None, False))
    core.post_step = Mock()
    core.execute_dummy_batch = Mock()
    core.capture_iteration_details = lambda _output: contextlib.nullcontext(None)
    core.output_queue = queue.Queue()
    core._ready_progress_seq = 0
    core._last_ready_published_seq = 0
    core._last_ready_published_state = EngineCoreReadyState.BUSY
    core._last_ready_published_at = time.monotonic() - 2
    core._last_health_dummy_batch_at = 0.0

    client = object.__new__(AsyncMPClient)
    client._ready_progress = {
        0: EngineCoreReadyProgress(
            seq=0,
            state=EngineCoreReadyState.BUSY,
            last_progress_at=time.monotonic() - 61,
        )
    }
    before_each = []
    after_each = []
    for _ in range(2):
        # Advance the observation window without waiting 60 real seconds.
        client._ready_progress[0].last_progress_at = time.monotonic() - 61
        core._last_ready_published_at = time.monotonic() - 2
        before_each.append(client.get_stalled_engine_ranks(60))
        core._handle_shutdown = Mock(side_effect=[True, False])
        with contextlib.suppress(SystemExit):
            core.run_busy_loop()

        if core.output_queue.qsize() != 1:
            raise AssertionError("expected exactly one readiness broadcast")
        destination, output = core.output_queue.get_nowait()
        if destination != -2:
            raise AssertionError("readiness broadcast used an unexpected destination")
        asyncio.run(AsyncMPClient.process_engine_outputs(client, output))
        after_each.append(client.get_stalled_engine_ranks(60))

    if core.step_fn.call_count != 2 or core.execute_dummy_batch.call_count != 2:
        raise AssertionError("the no-real-step/dummy-step control did not execute")
    if before_each != [[0], [0]] or output.ready_state != EngineCoreReadyState.BUSY:
        raise AssertionError("the initial stalled-BUSY control was not established")
    if output.ready_progress_seq != 2:
        raise AssertionError("dummy work did not advance the readiness sequence")

    return {
        **identity,
        "pin": PIN,
        "real_request_steps": 0,
        "dummy_batches": core.execute_dummy_batch.call_count,
        "ready_progress_seq": output.ready_progress_seq,
        "stalled_ranks_before_each_dummy": before_each,
        "stalled_ranks_after_each_dummy": after_each,
        "result": "dummy_clears_frontend_stall"
        if after_each == [[], []]
        else "stall_remains_visible",
        "scope": "mocked unschedulable demand; real PR DP loop and frontend methods",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-src", required=True, type=Path)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--binary-package-dir", type=Path)
    args = parser.parse_args()
    try:
        result = run(
            args.vllm_src.resolve(), args.source_archive, args.binary_package_dir
        )
    except Exception as exc:  # Apparatus errors must never become a scored result.
        print(json.dumps({"result": "unscored", "reason": type(exc).__name__}))
        print(f"unscored: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
