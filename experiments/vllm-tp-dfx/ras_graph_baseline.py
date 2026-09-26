"""Bounded TP=2 PyNccl CUDA-graph baseline; prints privacy-bounded RAS counts.

Run with torchrun from an installed vLLM source checkout. This is a healthy
producer-capability test, not a hang detector or a fault-injection campaign.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import socket
import stat
import time

import torch
import torch.distributed as dist

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.parallel_state import get_world_group, init_distributed_environment


def query_ras(port: int, private_output_path: str | None = None) -> dict:
    """Fetch RAS JSON; retain raw bytes only in a private, explicit file."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=3.0) as sock:
            sock.settimeout(3.0)
            sock.sendall(b"SET FORMAT json\nSTATUS\n")
            response = bytearray()
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline and len(response) < 2_000_000:
                try:
                    chunk = sock.recv(65536)
                except socket.timeout:
                    continue
                if not chunk:
                    break
                response.extend(chunk)
                opening = response.find(b"{")
                if opening >= 0:
                    try:
                        payload, _ = json.JSONDecoder().raw_decode(
                            response[opening:].decode("utf-8")
                        )
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    result = {"outcome": "available", "payload": payload}
                    if private_output_path is not None:
                        raw = bytes(response)
                        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
                        try:
                            fd = os.open(private_output_path, flags, 0o600)
                            with os.fdopen(fd, "wb") as output:
                                output.write(raw)
                        except OSError as exc:
                            raise RuntimeError(
                                "private RAS evidence write failed"
                            ) from exc
                        result["raw_sha256"] = hashlib.sha256(raw).hexdigest()
                    return result
            return {"outcome": "malformed_or_timed_out"}
    except (OSError, TimeoutError) as exc:
        return {"outcome": "unavailable", "error_type": type(exc).__name__}


def count_view(snapshot: dict) -> dict:
    """Discard hostnames, IPs, PIDs and raw communicator hashes."""
    if snapshot["outcome"] != "available":
        return {key: value for key, value in snapshot.items() if key != "payload"}
    payload = snapshot["payload"]
    communicators = []
    for comm in payload.get("communicators", []):
        ranks = []
        for rank in comm.get("ranks", []):
            ranks.append(
                {
                    "rank": rank.get("rank"),
                    "all_reduce_count": rank.get("collective_counts", {}).get(
                        "AllReduce"
                    ),
                    "status": {
                        key: rank.get("status", {}).get(key)
                        for key in ("init_state", "async_error", "abort_flag")
                    },
                }
            )
        communicators.append(
            {
                "identity": (comm.get("hash"), comm.get("secondary_hash")),
                "size": comm.get("size"),
                "ranks_count": comm.get("ranks_count"),
                "missing_ranks_count": comm.get("missing_ranks_count"),
                "ranks": sorted(ranks, key=lambda item: item["rank"]),
            }
        )
    return {
        "outcome": "available",
        "communicators": communicators,
        "raw_sha256": snapshot.get("raw_sha256"),
    }


def query_view(port: int, private_dir: str | None, name: str) -> dict:
    if private_dir is not None:
        directory = os.stat(private_dir)
        if not stat.S_ISDIR(directory.st_mode):
            raise ValueError("private evidence target is not a directory")
        if directory.st_uid != os.geteuid() or stat.S_IMODE(directory.st_mode) != 0o700:
            raise ValueError("private evidence directory must be owned, mode 0700")
        private_path = os.path.join(private_dir, name)
    else:
        private_path = None
    return count_view(query_ras(port, private_path))


def report_counts(before: dict, after: dict) -> dict:
    if before["outcome"] != "available" or after["outcome"] != "available":
        return {
            "outcome": "unscored",
            "before": before["outcome"],
            "after": after["outcome"],
        }
    previous = {item["identity"]: item for item in before["communicators"]}
    current = {item["identity"]: item for item in after["communicators"]}
    results = []
    for ordinal, identity in enumerate(sorted(set(previous) | set(current))):
        first = previous.get(identity)
        last = current.get(identity)
        first_counts = (
            {r["rank"]: r["all_reduce_count"] for r in first["ranks"]}
            if first
            else {}
        )
        last_counts = (
            {r["rank"]: r["all_reduce_count"] for r in last["ranks"]}
            if last
            else {}
        )
        deltas = {
            rank: last_counts[rank] - first_counts[rank]
            for rank in first_counts.keys() & last_counts.keys()
            if isinstance(first_counts[rank], int)
            and isinstance(last_counts[rank], int)
        }
        results.append(
            {
                "communicator_ordinal": ordinal,
                "ranks_before": sorted(first_counts),
                "ranks_after": sorted(last_counts),
                "all_reduce_counts_before": first_counts,
                "all_reduce_counts_after": last_counts,
                "all_reduce_deltas": deltas,
                "missing_ranks_after": last["missing_ranks_count"] if last else None,
            }
        )
    return {"outcome": "observed", "communicators": results}


def inspector_counts(private_dir: str) -> dict:
    """Read only complete Inspector JSON lines; disclose ranks and counts only."""
    directory = os.stat(private_dir)
    if not stat.S_ISDIR(directory.st_mode) or directory.st_uid != os.geteuid():
        raise ValueError("Inspector evidence directory must be owned by this user")
    if stat.S_IMODE(directory.st_mode) != 0o700:
        raise ValueError("Inspector evidence directory must have mode 0700")
    counts: dict[int, int] = {}
    for path in glob.glob(os.path.join(private_dir, "*.log")):
        with open(path, "rb") as stream:
            lines = stream.read().splitlines()
        complete = []
        for line in lines:
            try:
                complete.append(json.loads(line))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue  # Writer may be appending the final line.
        if not complete:
            continue
        ranks = {item.get("header", {}).get("rank") for item in complete}
        if len(ranks) != 1 or not isinstance(next(iter(ranks)), int):
            return {"outcome": "malformed"}
        rank = next(iter(ranks))
        if rank in counts:
            return {"outcome": "ambiguous_rank"}
        counts[rank] = sum(
            item.get("coll_perf", {}).get("coll") == "AllReduce"
            for item in complete
        )
    return {"outcome": "available", "all_reduce_records": counts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ras-port", type=int, default=28028)
    parser.add_argument("--mode", choices=("eager", "graph"), default="graph")
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--private-dir")
    args = parser.parse_args()
    if int(os.environ["WORLD_SIZE"]) != 2 or args.replays < 1 or args.replays > 100:
        raise ValueError("requires WORLD_SIZE=2 and 1 <= replays <= 100")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    init_distributed_environment()
    world = get_world_group()
    with torch.no_grad():
        comm = PyNcclCommunicator(world.cpu_group, device=world.device)
        source = torch.ones((4, 4), device=f"cuda:{local_rank}")
        graph = None
        if args.mode == "graph":
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                reduced = comm.all_reduce(source)
        else:
            reduced = comm.all_reduce(source)
        torch.cuda.synchronize()
        dist.barrier(group=world.cpu_group)
        before = (
            query_view(args.ras_port, args.private_dir, "before.ras.json")
            if rank == 0
            else None
        )
        dist.barrier(group=world.cpu_group)
        for iteration in range(args.replays):
            value = iteration + 2
            source.fill_(value)
            if graph is None:
                reduced = comm.all_reduce(source)
            else:
                graph.replay()
            torch.cuda.synchronize()
            if not torch.all(reduced == 2 * value).item():
                raise AssertionError(
                    f"replay {iteration} result differs from {2 * value}"
                )
            time.sleep(0.05)
        dist.barrier(group=world.cpu_group)
        after = (
            query_view(args.ras_port, args.private_dir, "after.ras.json")
            if rank == 0
            else None
        )
        dist.barrier(group=world.cpu_group)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "case": f"healthy_pynccl_{args.mode}",
                        "replays": args.replays,
                        "ras": report_counts(before, after),
                        "private_raw_sha256": {
                            "before": before["raw_sha256"],
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
