"""Five eager PyNccl all-reduces for the Inspector pTimer instrument check.

Run with two torchrun ranks and the same profiler plugin as the serving cells.
This is not a transport-hang reproduction or a device-timing oracle.
"""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.parallel_state import get_world_group, init_distributed_environment


def main() -> None:
    if int(os.environ["WORLD_SIZE"]) != 2:
        raise ValueError("eager control requires exactly two ranks")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    init_distributed_environment()
    world = get_world_group()
    with torch.no_grad():
        comm = PyNcclCommunicator(world.cpu_group, device=world.device)
        for iteration in range(5):
            source = torch.full((4, 4), float(iteration + 1), device=world.device)
            reduced = comm.all_reduce(source)
            torch.cuda.synchronize()
            if not torch.all(reduced == 2 * (iteration + 1)).item():
                raise AssertionError("eager all-reduce result mismatch")
    dist.barrier(group=world.cpu_group)
    if rank == 0:
        print(json.dumps({"case": "eager_ptimer_control", "iterations": 5, "result": "pass"}))


if __name__ == "__main__":
    main()
