"""Two-rank ungrouped PyNccl eager/graph callback-clock reproduction.

Run each mode in a fresh process and private directory. This isolates the
callback symptom from vLLM serving; it is not a timing oracle or a hang test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from importlib import metadata
from pathlib import Path

import torch
import torch.distributed as dist

from inflight_trace_v2 import read_rank_logs, save_private_snapshot, validate_rank_pair
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.parallel_state import get_world_group, init_distributed_environment

COLLECTIVES_PER_PASS = 3
REPLAY_COUNT = 2
SNAPSHOT_WAIT_SECONDS = 5.0
NCCL_LIBRARY_NAME = re.compile(r"[/\\]libnccl\.so(?:\.\d+)*$")


def _loaded_nccl_digest() -> str:
    mapped = set()
    for line in Path("/proc/self/maps").read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and NCCL_LIBRARY_NAME.search(parts[5]):
            mapped.add(Path(parts[5]).resolve(strict=True))
    if len(mapped) != 1:
        raise ValueError("runtime NCCL mapping is ambiguous")
    digest = hashlib.sha256()
    with next(iter(mapped)).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_with_completed_collectives(pattern: str, minimum: int) -> dict:
    """Wait briefly for line-buffered profiler callbacks after device sync."""
    deadline = time.monotonic() + SNAPSHOT_WAIT_SECONDS
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            snapshot = read_rank_logs(pattern)
            validate_rank_pair(snapshot)
            for events in snapshot.values():
                starts = sum(event.kind == "coll_start" for event in events)
                stops = sum(event.kind == "kernel_ch_stop" for event in events)
                if starts < minimum or stops < minimum:
                    raise ValueError("profiler callback history is incomplete")
            return snapshot
        except ValueError as exc:
            last_error = exc
            time.sleep(0.05)
    raise ValueError("bounded profiler snapshot unavailable") from last_error


def _capture_three(comm: PyNcclCommunicator, tensor: torch.Tensor, cycles: int):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = []
        for index in range(COLLECTIVES_PER_PASS):
            if index:
                torch.cuda._sleep(cycles)
            outputs.append(comm.all_reduce(tensor))
    return graph, outputs


def _assert_outputs(outputs: list[torch.Tensor], value: float) -> None:
    for output in outputs:
        if not torch.all(output == value).item():
            raise AssertionError("all-reduce result mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("eager", "graph"))
    parser.add_argument("--private-dir", required=True, type=Path)
    parser.add_argument("--sleep-cycles", type=int, default=1_000_000)
    args = parser.parse_args()
    if int(os.environ["WORLD_SIZE"]) != 2:
        raise ValueError("requires exactly two torchrun ranks")
    if not 1 <= args.sleep_cycles <= 5_000_000:
        raise ValueError("sleep-cycles must be within [1, 5000000]")
    private = args.private_dir.resolve(strict=True)
    if private.stat().st_mode & 0o077:
        raise ValueError("private directory must be owner-only")
    pattern = str(private / "nccl.*.log")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    init_distributed_environment()
    world = get_world_group()
    with torch.no_grad():
        comm = PyNcclCommunicator(world.cpu_group, device=world.device)
        local_nccl_digest = _loaded_nccl_digest()
        nccl_digests: list[str] = ["", ""]
        dist.all_gather_object(
            nccl_digests, local_nccl_digest, group=world.cpu_group
        )
        if len(set(nccl_digests)) != 1:
            raise ValueError("rank NCCL library identities differ")
        source = torch.ones((4, 4), device=world.device)
        if args.mode == "graph":
            graph, outputs = _capture_three(comm, source, args.sleep_cycles)
            torch.cuda.synchronize()
        dist.barrier(group=world.cpu_group)
        if rank == 0:
            before = _snapshot_with_completed_collectives(pattern, minimum=1)
            before_count = sum(
                event.kind == "coll_start" for event in before[0]
            )
            before_digest = save_private_snapshot(private, "before", before)
        else:
            before_count = None
            before_digest = None
        dist.barrier(group=world.cpu_group)

        if args.mode == "eager":
            for index in range(COLLECTIVES_PER_PASS):
                source.fill_(index + 2)
                output = comm.all_reduce(source)
                torch.cuda.synchronize()
                _assert_outputs([output], 2 * (index + 2))
            expected_new = COLLECTIVES_PER_PASS
        else:
            for value in (2, 3):
                source.fill_(value)
                graph.replay()
                torch.cuda.synchronize()
                _assert_outputs(outputs, 2 * value)
            expected_new = COLLECTIVES_PER_PASS * REPLAY_COUNT

        dist.barrier(group=world.cpu_group)
        if rank == 0:
            assert before_count is not None and before_digest is not None
            after = _snapshot_with_completed_collectives(
                pattern, minimum=before_count + expected_new
            )
            if after[0][: len(before[0])] != before[0] or after[1][: len(before[1])] != before[1]:
                raise ValueError("callback history is not a prefix")
            new_counts = {
                str(binding): sum(
                    event.kind == "coll_start"
                    for event in after[binding][len(before[binding]) :]
                )
                for binding in (0, 1)
            }
            if any(count != expected_new for count in new_counts.values()):
                raise ValueError("unexpected number of ungrouped collective callbacks")
            after_digest = save_private_snapshot(private, "after", after)
            print(
                json.dumps(
                    {
                        "schema": "tp-ptimer-standalone-v1",
                        "mode": args.mode,
                        "collectives_per_pass": COLLECTIVES_PER_PASS,
                        "replay_count": REPLAY_COUNT if args.mode == "graph" else 0,
                        "sleep_cycles": args.sleep_cycles if args.mode == "graph" else 0,
                        "new_coll_start_by_rank": new_counts,
                        "before_sha256": before_digest,
                        "after_sha256": after_digest,
                        "numeric_result": "pass",
                        "runtime_nccl_sha256": local_nccl_digest,
                        "torch_version": torch.__version__,
                        "vllm_version": metadata.version("vllm"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    dist.barrier(group=world.cpu_group)


if __name__ == "__main__":
    main()
