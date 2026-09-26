"""Bounded TP=2 PyNccl B candidate: both host calls issued, GPU work delayed.

Run a no-sleep control first. This is a device-stream delay, not a transport
hang. A B claim requires a pending sleep event across the during-hold RAS
query, equal issued counts, correct completion, and a calibrated duration.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.parallel_state import get_world_group, init_distributed_environment

from ras_graph_baseline import query_view, report_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ras-port", type=int, default=28028)
    parser.add_argument("--sleep-cycles", type=int, default=0)
    parser.add_argument("--private-dir")
    args = parser.parse_args()
    if int(os.environ["WORLD_SIZE"]) != 2 or not 0 <= args.sleep_cycles <= 5_000_000_000:
        raise ValueError("requires WORLD_SIZE=2 and 0 <= sleep-cycles <= 5e9")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    init_distributed_environment()
    world = get_world_group()
    with torch.no_grad():
        comm = PyNcclCommunicator(world.cpu_group, device=world.device)
        source = torch.ones((4, 4), device=f"cuda:{local_rank}")
        warmup = comm.all_reduce(source)
        torch.cuda.synchronize()
        if not torch.all(warmup == 2).item():
            raise AssertionError("warmup result differs from 2")

        dist.barrier(group=world.cpu_group)
        before = (
            query_view(args.ras_port, args.private_dir, "before.ras.json")
            if rank == 1
            else None
        )
        dist.barrier(group=world.cpu_group)

        sleep_start = torch.cuda.Event(enable_timing=True)
        sleep_end = torch.cuda.Event(enable_timing=True)
        if rank == 1:
            sleep_start.record()
            if args.sleep_cycles:
                torch.cuda._sleep(args.sleep_cycles)
            sleep_end.record()

        reduced = comm.all_reduce(source)
        # Gloo rendezvous: both host calls returned before rank 1 queries RAS.
        dist.barrier(group=world.cpu_group)
        if rank == 1:
            sleep_pending_before = not sleep_end.query()
            during = query_view(args.ras_port, args.private_dir, "during.ras.json")
            sleep_pending_after = not sleep_end.query()
        else:
            during = None
            sleep_pending_before = None
            sleep_pending_after = None

        sync_start = time.monotonic()
        torch.cuda.synchronize()
        sync_elapsed_ms = (time.monotonic() - sync_start) * 1000
        dist.barrier(group=world.cpu_group)
        after = (
            query_view(args.ras_port, args.private_dir, "after.ras.json")
            if rank == 1
            else None
        )
        dist.barrier(group=world.cpu_group)
        if not torch.all(reduced == 2).item():
            raise AssertionError("released all-reduce result differs from 2")

        if rank == 1:
            print(
                json.dumps(
                    {
                        "case": "all_issued_device_delay",
                        "sleep_cycles": args.sleep_cycles,
                        "issued_barrier_passed": True,
                        "sleep_pending_before_ras": sleep_pending_before,
                        "sleep_pending_after_ras": sleep_pending_after,
                        "sleep_elapsed_ms": sleep_start.elapsed_time(sleep_end),
                        "rank1_sync_elapsed_ms": sync_elapsed_ms,
                        "before_to_during": report_counts(before, during),
                        "during_to_after": report_counts(during, after),
                        "private_raw_sha256": {
                            "before": before["raw_sha256"],
                            "during": during["raw_sha256"],
                            "after": after["raw_sha256"],
                        },
                        "result": "pass",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    dist.barrier(group=world.cpu_group)


if __name__ == "__main__":
    main()
