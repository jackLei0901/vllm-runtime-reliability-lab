"""TP=2 captured PyNccl all-reduce with optional bounded device-side delay.

This tests an all-issued graph replay condition, not a transport fault or a
vLLM serving workload. Run the zero-delay control before a delayed cell.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.parallel_state import get_world_group, init_distributed_environment

from ras_graph_baseline import inspector_counts, query_view, report_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ras-port", type=int, default=28028)
    parser.add_argument("--sleep-cycles", type=int, default=0)
    parser.add_argument("--private-dir")
    parser.add_argument("--inspector-dir")
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

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            if rank == 1 and args.sleep_cycles:
                torch.cuda._sleep(args.sleep_cycles)
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

        source.fill_(2)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        # Both host-side replay calls returned before the during snapshot.
        dist.barrier(group=world.cpu_group)
        if rank == 1:
            graph_pending_before = not end.query()
            during = query_view(args.ras_port, args.private_dir, "during.ras.json")
            inspector_during = inspector_counts(args.inspector_dir) if args.inspector_dir else None
            graph_pending_after = not end.query()
        else:
            during = None
            inspector_during = None
            graph_pending_before = None
            graph_pending_after = None

        torch.cuda.synchronize()
        dist.barrier(group=world.cpu_group)
        after = (
            query_view(args.ras_port, args.private_dir, "after.ras.json")
            if rank == 1
            else None
        )
        inspector_after = inspector_counts(args.inspector_dir) if rank == 1 and args.inspector_dir else None
        dist.barrier(group=world.cpu_group)
        if not torch.all(reduced == 4).item():
            raise AssertionError("replayed all-reduce result differs from 4")
        if rank == 1:
            print(
                json.dumps(
                    {
                        "case": "graph_all_issued_device_delay",
                        "sleep_cycles": args.sleep_cycles,
                        "issued_barrier_passed": True,
                        "graph_pending_before_ras": graph_pending_before,
                        "graph_pending_after_ras": graph_pending_after,
                        "graph_elapsed_ms": start.elapsed_time(end),
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
