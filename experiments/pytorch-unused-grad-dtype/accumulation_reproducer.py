"""Two-rank FSDP2 reproducer with pipeline-style gradient accumulation."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard


class ConditionalModel(nn.Module):
    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.always = nn.Linear(width, width)
        self.conditional = nn.Linear(width, width)

    def forward(self, value: torch.Tensor, use_conditional: bool) -> torch.Tensor:
        output = self.always(value)
        if use_conditional:
            output = output + self.conditional(value)
        return output


def _record_rank_pid(state_dir: Path, rank: int) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    target = state_dir / f"rank-{rank}.pid"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=state_dir, delete=False
    ) as handle:
        handle.write(f"{os.getpid()}\n")
        temporary = Path(handle.name)
    temporary.replace(target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("affected", "control"), required=True)
    parser.add_argument("--microbatches", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.microbatches < 2:
        raise ValueError("--microbatches must be at least two")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=30))
    _record_rank_pid(Path(os.environ["DFX_STATE_DIR"]), rank)
    device = torch.device("cuda", local_rank)

    try:
        torch.manual_seed(20260912)
        model = ConditionalModel().to(device)
        fully_shard(
            model,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
            ),
        )
        model.set_reduce_scatter_unused_params(True)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)

        for microbatch in range(args.microbatches):
            is_last = microbatch == args.microbatches - 1
            model.set_is_last_backward(is_last)
            model.set_requires_gradient_sync(is_last)
            model.set_reshard_after_backward(is_last)
            value = torch.randn(4, 16, device=device, dtype=torch.bfloat16)
            use_conditional = args.arm == "control" or rank == 0
            model(value, use_conditional=use_conditional).float().sum().backward()

        optimizer.step()
        dist.barrier()
        if rank == 0:
            print(
                "DFX_RESULT="
                + json.dumps(
                    {
                        "arm": args.arm,
                        "execution_mode": "accumulated",
                        "microbatches": args.microbatches,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        return 0
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
