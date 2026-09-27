"""One-shot two-rank NCCL profiler call-history control.

Run each variant in a fresh torchrun process. Raw identities stay in the
owner-only directory; stdout carries only digests and closed counts.
"""

from __future__ import annotations

import argparse
import json
import os
from importlib import metadata
from pathlib import Path

import torch
import torch.distributed as dist

from inflight_trace_v2 import save_private_snapshot, validate_rank_pair
from ptimer_standalone_replay import (
    COLLECTIVES_PER_PASS,
    REPLAY_COUNT,
    _assert_outputs,
    _capture_three,
    _loaded_nccl_digest,
    _snapshot_with_completed_collectives,
)
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.parallel_state import get_world_group, init_distributed_environment

VARIANTS = ("no_eager", "same_comm", "other_comm")
EXPECTED_TORCH = "2.13.0+cu130"
EXPECTED_VLLM = "0.1.dev586+gc8602c790.precompiled"
EXPECTED_NCCL_SHA256 = (
    "aa957cdfb91b516eae0d54a28e9ee5db52730d02e0ab45580efc3c19a68327a4"
)


def _new_starts(events) -> int:
    return sum(event.kind == "coll_start" for event in events)


def _prefix(before, after) -> None:
    validate_rank_pair(before)
    validate_rank_pair(after)
    for rank in (0, 1):
        if after[rank][: len(before[rank])] != before[rank]:
            raise ValueError("profiler history is not a prefix")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--private-dir", required=True, type=Path)
    parser.add_argument("--sleep-cycles", type=int, default=1_000_000)
    args = parser.parse_args()
    if int(os.environ["WORLD_SIZE"]) != 2:
        raise ValueError("requires exactly two torchrun ranks")
    if not 1 <= args.sleep_cycles <= 5_000_000:
        raise ValueError("sleep-cycles outside frozen bound")
    if torch.__version__ != EXPECTED_TORCH or metadata.version("vllm") != EXPECTED_VLLM:
        raise ValueError("runtime version differs from frozen gate")
    private = args.private_dir.resolve(strict=True)
    if private.stat().st_mode & 0o077:
        raise ValueError("private directory must be owner-only")
    pattern = str(private / "nccl.*.log")
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_distributed_environment()
    world = get_world_group()
    with torch.no_grad():
        primary = PyNcclCommunicator(world.cpu_group, device=world.device)
        secondary = PyNcclCommunicator(world.cpu_group, device=world.device)
        nccl_digest = _loaded_nccl_digest()
        if nccl_digest != EXPECTED_NCCL_SHA256:
            raise ValueError("mapped NCCL library differs from frozen gate")
        digests = ["", ""]
        dist.all_gather_object(digests, nccl_digest, group=world.cpu_group)
        if len(set(digests)) != 1:
            raise ValueError("rank NCCL library identities differ")
        source = torch.ones((4, 4), device=world.device)
        graph, outputs = _capture_three(primary, source, args.sleep_cycles)
        torch.cuda.synchronize()
        dist.barrier(group=world.cpu_group)
        if rank == 0:
            before = _snapshot_with_completed_collectives(pattern, minimum=5)
            before_sha = save_private_snapshot(private, "before", before)
            before_count = _new_starts(before[0])
        else:
            before_count = None
            before_sha = None
        dist.barrier(group=world.cpu_group)

        if args.variant != "no_eager":
            source.fill_(7)
            target = primary if args.variant == "same_comm" else secondary
            eager_output = target.all_reduce(source)
            torch.cuda.synchronize()
            _assert_outputs([eager_output], 14)
        dist.barrier(group=world.cpu_group)
        if rank == 0:
            assert before_count is not None
            during = _snapshot_with_completed_collectives(
                pattern,
                minimum=before_count + (args.variant != "no_eager"),
            )
            _prefix(before, during)
            if any(
                _new_starts(during[r][len(before[r]) :])
                != int(args.variant != "no_eager")
                for r in (0, 1)
            ):
                raise ValueError("unexpected intervening collective count")
            during_sha = save_private_snapshot(private, "during", during)
            during_count = _new_starts(during[0])
        else:
            during_count = None
            during_sha = None
        dist.barrier(group=world.cpu_group)

        for value in (2, 3):
            source.fill_(value)
            graph.replay()
            torch.cuda.synchronize()
            _assert_outputs(outputs, 2 * value)
        dist.barrier(group=world.cpu_group)
        if rank == 0:
            assert before_sha is not None and during_sha is not None
            assert during_count is not None
            after = _snapshot_with_completed_collectives(
                pattern, minimum=during_count + COLLECTIVES_PER_PASS * REPLAY_COUNT
            )
            _prefix(during, after)
            if any(
                _new_starts(after[r][len(during[r]) :])
                != COLLECTIVES_PER_PASS * REPLAY_COUNT
                for r in (0, 1)
            ):
                raise ValueError("unexpected graph collective count")
            after_sha = save_private_snapshot(private, "after", after)
            print(
                json.dumps(
                    {
                        "schema": "tp-ptimer-call-history-v1",
                        "variant": args.variant,
                        "before_sha256": before_sha,
                        "during_sha256": during_sha,
                        "after_sha256": after_sha,
                        "numeric_result": "pass",
                        "runtime_nccl_sha256": nccl_digest,
                        "torch_version": torch.__version__,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    dist.barrier(group=world.cpu_group)


if __name__ == "__main__":
    main()
