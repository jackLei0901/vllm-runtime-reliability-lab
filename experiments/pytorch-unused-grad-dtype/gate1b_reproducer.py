"""Per-rank probe for the accumulated unused-gradient failure."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.fsdp._fully_shard._fsdp_param_group as fsdp_param_group
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

ASSERTION_MARKER = "FSDP reduce-scatter expects uniform gradient dtype"


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


def emit(prefix: str, payload: dict) -> None:
    print(prefix + json.dumps(payload, sort_keys=True), flush=True)


def record_rank_pid(state_dir: Path, rank: int) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    target = state_dir / f"rank-{rank}.pid"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=state_dir, delete=False
    ) as handle:
        handle.write(f"{os.getpid()}\n")
        temporary = Path(handle.name)
    temporary.replace(target)


def install_probe(rank: int, started: float) -> dict[str, bool]:
    original = fsdp_param_group.foreach_reduce
    state = {"terminal_emitted": False}
    call_index = 0

    def observed(fsdp_params, unsharded_grads, *args, **kwargs):
        nonlocal call_index
        call_index += 1
        emit(
            "DFX_RANK_REDUCE=",
            {
                "rank": rank,
                "call_index": call_index,
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "grad_count": len(unsharded_grads),
                "grad_dtypes": [str(grad.dtype) for grad in unsharded_grads],
            },
        )
        try:
            result = original(fsdp_params, unsharded_grads, *args, **kwargs)
        except BaseException as exc:
            outcome = (
                "local_uniformity_assertion"
                if ASSERTION_MARKER in str(exc)
                else "local_unexpected_exception"
            )
            emit(
                "DFX_RANK_OUTCOME=",
                {
                    "rank": rank,
                    "outcome": outcome,
                    "exception_type": type(exc).__name__,
                    "elapsed_seconds": round(time.monotonic() - started, 6),
                },
            )
            state["terminal_emitted"] = True
            raise
        emit(
            "DFX_RANK_REDUCE_RETURN=",
            {
                "rank": rank,
                "call_index": call_index,
                "elapsed_seconds": round(time.monotonic() - started, 6),
            },
        )
        return result

    fsdp_param_group.foreach_reduce = observed
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("affected", "control"), required=True)
    parser.add_argument("--microbatches", type=int, default=4)
    args = parser.parse_args()
    if args.microbatches < 2:
        parser.error("--microbatches must be at least two")

    started = time.monotonic()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=30))
    record_rank_pid(Path(os.environ["DFX_STATE_DIR"]), rank)
    device = torch.device("cuda", local_rank)
    probe_state = {"terminal_emitted": False}

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
        probe_state = install_probe(rank, started)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)

        for microbatch in range(args.microbatches):
            is_last = microbatch == args.microbatches - 1
            model.set_is_last_backward(is_last)
            model.set_requires_gradient_sync(is_last)
            model.set_reshard_after_backward(is_last)
            value = torch.randn(4, 16, device=device, dtype=torch.bfloat16)
            use_conditional = args.arm == "control" or rank == 0
            model(value, use_conditional).float().sum().backward()

        optimizer.step()
        dist.barrier()
        emit(
            "DFX_RANK_OUTCOME=",
            {
                "rank": rank,
                "outcome": "completed",
                "elapsed_seconds": round(time.monotonic() - started, 6),
            },
        )
        probe_state["terminal_emitted"] = True
        return 0
    except BaseException as exc:
        if not probe_state["terminal_emitted"]:
            emit(
                "DFX_RANK_OUTCOME=",
                {
                    "rank": rank,
                    "outcome": "outside_reduce_exception",
                    "exception_type": type(exc).__name__,
                    "elapsed_seconds": round(time.monotonic() - started, 6),
                },
            )
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
