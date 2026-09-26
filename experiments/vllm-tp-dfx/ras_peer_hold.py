"""Bounded TP=2 eager PyNccl peer-before-launch hold, with RAS snapshots.

This is one side of a proposed discrimination pair. It does not establish
that RAS can distinguish peer absence from an all-entered collective hang.
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

from ras_graph_baseline import inspector_counts, query_view, report_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ras-port", type=int, default=28028)
    parser.add_argument("--hold-seconds", type=float, default=3.0)
    parser.add_argument("--mode", choices=("eager", "graph"), default="eager")
    parser.add_argument("--private-dir")
    parser.add_argument("--inspector-dir")
    args = parser.parse_args()
    if int(os.environ["WORLD_SIZE"]) != 2 or not 1 <= args.hold_seconds <= 10:
        raise ValueError("requires WORLD_SIZE=2 and 1 <= hold-seconds <= 10")

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

        graph = None
        if args.mode == "graph":
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                reduced = comm.all_reduce(source)
            torch.cuda.synchronize()

        dist.barrier(group=world.cpu_group)
        before = (
            query_view(args.ras_port, args.private_dir, "before.ras.json")
            if rank == 1
            else None
        )
        inspector_before = inspector_counts(args.inspector_dir) if rank == 1 and args.inspector_dir else None
        dist.barrier(group=world.cpu_group)

        if graph is not None:
            source.fill_(2)
        if rank == 1:
            time.sleep(args.hold_seconds)
            during = query_view(args.ras_port, args.private_dir, "during.ras.json")
            inspector_during = inspector_counts(args.inspector_dir) if args.inspector_dir else None
        else:
            during = None
            inspector_during = None
        if graph is None:
            reduced = comm.all_reduce(source)
        else:
            graph.replay()
        torch.cuda.synchronize()
        dist.barrier(group=world.cpu_group)

        after = (
            query_view(args.ras_port, args.private_dir, "after.ras.json")
            if rank == 1
            else None
        )
        inspector_after = inspector_counts(args.inspector_dir) if rank == 1 and args.inspector_dir else None
        dist.barrier(group=world.cpu_group)
        expected = 4 if graph is not None else 2
        if not torch.all(reduced == expected).item():
            raise AssertionError(f"released all-reduce result differs from {expected}")
        if rank == 1:
            print(
                json.dumps(
                    {
                        "case": f"{args.mode}_peer_before_launch_hold",
                        "hold_seconds": args.hold_seconds,
                        "before_to_during": report_counts(before, during),
                        "during_to_after": report_counts(during, after),
                        "private_raw_sha256": {
                            "before": before["raw_sha256"],
                            "during": during["raw_sha256"],
                            "after": after["raw_sha256"],
                        },
                        "inspector": {
                            "before": inspector_before,
                            "during": inspector_during,
                            "after": inspector_after,
                        } if args.inspector_dir else None,
                        "result": "pass",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    dist.barrier(group=world.cpu_group)


if __name__ == "__main__":
    main()
