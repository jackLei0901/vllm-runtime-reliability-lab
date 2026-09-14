"""Minimal FSDP2 reproducer for mixed fresh/accumulated gradient dtypes.

Examples:
  torchrun --standalone --nproc-per-node=1 reproducer.py --case control
  torchrun --standalone --nproc-per-node=1 reproducer.py --case last-microbatch
  torchrun --standalone --nproc-per-node=1 reproducer.py --case unused-placeholder
  torchrun --standalone --nproc-per-node=2 reproducer.py --case rank-divergent-unused
"""

from __future__ import annotations

import argparse
import datetime
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard


CASES = (
    "control",
    "last-microbatch",
    "unused-placeholder",
    "rank-divergent-unused",
)


class Model(nn.Module):
    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.always = nn.Linear(width, width)
        self.conditional = nn.Linear(width, width)

    def forward(self, x: torch.Tensor, use_conditional: bool) -> torch.Tensor:
        output = self.always(x)
        if use_conditional:
            output = output + self.conditional(x)
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--microbatches", type=int, default=4)
    args = parser.parse_args()
    if args.microbatches < 2:
        parser.error("--microbatches must be at least 2")
    return args


def uses_conditional(case: str, rank: int, microbatch: int, last: int) -> bool:
    if case == "control":
        return True
    if case == "last-microbatch":
        return microbatch == last
    if case == "unused-placeholder":
        return False
    if case == "rank-divergent-unused":
        return rank == 0
    raise AssertionError(f"unhandled case: {case}")


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if args.case == "rank-divergent-unused" and world_size < 2:
        raise ValueError("rank-divergent-unused requires at least 2 ranks")
    if not hasattr(FSDPModule, "set_reduce_scatter_unused_params"):
        raise RuntimeError("this PyTorch build lacks the required FSDP2 API")

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=30))
    device = torch.device("cuda", local_rank)

    print(
        f"REPRO_START rank={rank} world_size={world_size} case={args.case} "
        f"torch={torch.__version__}",
        flush=True,
    )

    try:
        torch.manual_seed(20260914)
        model = Model().to(device)
        fully_shard(
            model,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
            ),
        )
        if args.case in ("unused-placeholder", "rank-divergent-unused"):
            model.set_reduce_scatter_unused_params(True)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)

        last = args.microbatches - 1
        for microbatch in range(args.microbatches):
            is_last = microbatch == last
            model.set_is_last_backward(is_last)
            model.set_requires_gradient_sync(is_last)
            model.set_reshard_after_backward(is_last)
            x = torch.randn(4, 16, device=device, dtype=torch.bfloat16)
            use_conditional = uses_conditional(
                args.case, rank, microbatch, last
            )
            model(x, use_conditional).float().sum().backward()

        optimizer.step()
        dist.barrier()
        print(f"REPRO_RESULT rank={rank} status=completed", flush=True)
    except BaseException as error:
        print(
            f"REPRO_RESULT rank={rank} status=exception "
            f"type={type(error).__name__} message={error}",
            flush=True,
        )
        raise
    finally:
        print(f"REPRO_CLEANUP rank={rank} stage=enter", flush=True)
        dist.destroy_process_group()
        print(f"REPRO_CLEANUP rank={rank} stage=return", flush=True)


if __name__ == "__main__":
    main()
