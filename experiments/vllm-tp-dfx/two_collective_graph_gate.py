"""Two captured all-reduces with a bounded device delay between them.

This is a profiler-interface timing check, not a serving or fault campaign.
Only callback counts leave the private NCCL debug directory.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.parallel_state import get_world_group, init_distributed_environment

from ras_peer_hold import start_event_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nccl-debug-dir", required=True)
    parser.add_argument("--sleep-cycles", type=int, default=5_000_000_000)
    args = parser.parse_args()
    if int(os.environ["WORLD_SIZE"]) != 2 or not 1 <= args.sleep_cycles <= 5_000_000_000:
        raise ValueError("requires WORLD_SIZE=2 and 1 <= sleep-cycles <= 5e9")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    init_distributed_environment()
    world = get_world_group()
    pid_by_rank: list[int] = [0, 0]
    dist.all_gather_object(pid_by_rank, os.getpid(), group=world.cpu_group)

    with torch.no_grad():
        comm = PyNcclCommunicator(world.cpu_group, device=world.device)
        first_input = torch.ones((4, 4), device=world.device)
        second_input = torch.ones((4, 4), device=world.device)
        warmup = comm.all_reduce(first_input)
        torch.cuda.synchronize()
        if not torch.all(warmup == 2).item():
            raise AssertionError("warmup result differs from 2")

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            first_output = comm.all_reduce(first_input)
            if rank == 1:
                torch.cuda._sleep(args.sleep_cycles)
            second_output = comm.all_reduce(second_input)
        torch.cuda.synchronize()
        dist.barrier(group=world.cpu_group)
        before = start_event_counts(args.nccl_debug_dir, pid_by_rank) if rank == 1 else None
        dist.barrier(group=world.cpu_group)

        first_input.fill_(2)
        second_input.fill_(3)
        completed = torch.cuda.Event(enable_timing=True)
        graph.replay()
        completed.record()
        dist.barrier(group=world.cpu_group)
        if rank == 1:
            pending_before = not completed.query()
            during = start_event_counts(args.nccl_debug_dir, pid_by_rank)
            pending_after = not completed.query()
        else:
            pending_before = pending_after = during = None

        torch.cuda.synchronize()
        dist.barrier(group=world.cpu_group)
        after = start_event_counts(args.nccl_debug_dir, pid_by_rank) if rank == 1 else None
        dist.barrier(group=world.cpu_group)
        if not torch.all(first_output == 4).item() or not torch.all(second_output == 6).item():
            raise AssertionError("replayed all-reduce results differ")
        if rank == 1:
            print(
                json.dumps(
                    {
                        "case": "two_collective_graph_device_delay",
                        "sleep_cycles": args.sleep_cycles,
                        "pending_before_snapshot": pending_before,
                        "pending_after_snapshot": pending_after,
                        "before": before,
                        "during": during,
                        "after": after,
                        "result": "pass",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    dist.barrier(group=world.cpu_group)


if __name__ == "__main__":
    main()
