"""Observe the exact dtype set passed to FSDP2 ``foreach_reduce``."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.fsdp._fully_shard._fsdp_param_group as fsdp_param_group
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard


class UnusedParameterModel(nn.Module):
    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.always = nn.Linear(width, width)
        self.conditional = nn.Linear(width, width)

    def forward(self, value: torch.Tensor, rank: int) -> torch.Tensor:
        output = self.always(value)
        if rank == 0:
            output = output + self.conditional(value)
        return output


class ForcedMixedGradientModel(nn.Module):
    """Positive control adapted from Garrett Goon's public two-rank repro."""

    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.bf16_path = nn.Linear(width, width, bias=False)
        self.fp32_path = nn.Linear(width, width, bias=False)

    def forward(self, value: torch.Tensor, rank: int) -> torch.Tensor:
        del rank
        output = self.bf16_path(value)
        original_dtype = self.fp32_path.weight.dtype
        self.fp32_path.to(torch.float32)
        output = self.fp32_path(output.to(torch.float32))
        self.fp32_path.to(original_dtype)
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


def _install_dtype_probe(rank: int) -> None:
    original = fsdp_param_group.foreach_reduce

    def observed(fsdp_params, unsharded_grads, *args, **kwargs):
        payload = {
            "rank": rank,
            "grad_count": len(unsharded_grads),
            "grad_dtypes": sorted(str(grad.dtype) for grad in unsharded_grads),
        }
        print("DFX_REDUCE_DTYPES=" + json.dumps(payload, sort_keys=True), flush=True)
        return original(fsdp_params, unsharded_grads, *args, **kwargs)

    fsdp_param_group.foreach_reduce = observed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case", choices=("unused-parameter", "forced-mixed-gradient"), required=True
    )
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=30))
    _record_rank_pid(Path(os.environ["DFX_STATE_DIR"]), rank)
    device = torch.device("cuda", local_rank)

    try:
        torch.manual_seed(20260912)
        model: nn.Module
        if args.case == "unused-parameter":
            model = UnusedParameterModel().to(device)
        else:
            model = ForcedMixedGradientModel().to(device)
        fully_shard(
            model,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
            ),
        )
        if args.case == "unused-parameter":
            model.set_reduce_scatter_unused_params(True)  # type: ignore[operator]
        _install_dtype_probe(rank)

        value = torch.randn(4, 16, device=device, dtype=torch.bfloat16)
        model(value, rank).float().sum().backward()
        dist.barrier()
        if rank == 0:
            print(
                "DFX_RESULT="
                + json.dumps({"case": args.case, "completed": True}, sort_keys=True),
                flush=True,
            )
        return 0
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
