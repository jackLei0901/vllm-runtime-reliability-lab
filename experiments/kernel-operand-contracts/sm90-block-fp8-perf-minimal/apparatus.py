"""Read-only Linux preflight. An unreadable process inventory fails closed."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

IDLE_MEMORY_MIB = 128
GIB = 1024**3
E4_MIN_LIMIT_BYTES = 96 * GIB
E4_MIN_AVAILABLE_BYTES = 64 * GIB


def blocked_process(command):
    if not command:
        return False
    names = {Path(arg).name for arg in command}
    return bool(
        names
        & {
            "nvcc",
            "ninja",
            "cmake",
            "cc1plus",
            "cc1",
            "g++",
            "c++",
            "gcc",
            "clang++",
            "nsys",
            "build_perf.py",
            "build_pair.py",
            "serve_trace.py",
            "identity_probe.py",
            "replay_check.py",
        }
    ) or any(
        "vllm.entrypoints.openai.api_server" in arg or "VLLM::EngineCore" in arg
        for arg in command
    )


def idle_receipt(proc_root=Path("/proc"), own_pid=None):
    own_pid = os.getpid() if own_pid is None else own_pid
    inventory = []
    if not proc_root.is_dir():
        raise ValueError("Linux process inventory required")
    for directory in proc_root.iterdir():
        if not directory.name.isdecimal() or int(directory.name) == own_pid:
            continue
        try:
            command = [
                part.decode("utf-8", errors="replace")
                for part in (directory / "cmdline").read_bytes().split(b"\0")
                if part
            ]
        except FileNotFoundError:
            continue  # process exited during snapshot
        if blocked_process(command):
            inventory.append({"pid": int(directory.name), "command": command})
    query = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        encoding="utf-8",
        timeout=10,
    )
    gpu_pids = []
    for line in query.splitlines():
        if not line.strip().isdecimal():
            raise ValueError("GPU PID inventory unavailable")
        gpu_pids.append(int(line.strip()))
    memory_query = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,memory.used",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
            timeout=10,
        )
        .strip()
        .splitlines()
    )
    if len(memory_query) != 1:
        raise ValueError("exactly one physical GPU inventory required")
    fields = [field.strip() for field in memory_query[0].split(",")]
    if (
        len(fields) != 2
        or not fields[0].startswith("GPU-")
        or not fields[1].isdecimal()
    ):
        raise ValueError("GPU memory inventory unavailable")
    memory_mib = int(fields[1])
    if inventory or gpu_pids or memory_mib > IDLE_MEMORY_MIB:
        raise ValueError("GPU not idle or compiler/server/profiler remains")
    return {
        "checked_at_unix_ns": time.time_ns(),
        "own_pid": own_pid,
        "blocked_processes": inventory,
        "gpu_compute_pids": gpu_pids,
        "gpu_uuid": fields[0],
        "memory_used_mib": memory_mib,
        "idle_memory_max_mib": IDLE_MEMORY_MIB,
        "cuda_context_expected": "none; no PID namespace translation required",
        "status": "idle_snapshot",
        "limit": "point-in-time inventory, not a guarantee against later external load",
    }


def memory_headroom(proc_root=Path("/proc"), cgroup_root=Path("/sys/fs/cgroup")):
    """Conservative E4 admission, not a guarantee of peak compiler memory."""
    info = {}
    for line in (proc_root / "meminfo").read_text().splitlines():
        name, _, value = line.partition(":")
        if name in ("MemTotal", "MemAvailable"):
            info[name] = int(value.split()[0]) * 1024
    total, available = info["MemTotal"], info["MemAvailable"]
    pairs = (
        (cgroup_root / "memory.max", cgroup_root / "memory.current"),
        (
            cgroup_root / "memory/memory.limit_in_bytes",
            cgroup_root / "memory/memory.usage_in_bytes",
        ),
    )
    limits = []
    observed = []
    for maximum, current in pairs:
        if maximum.is_file() and current.is_file():
            raw = maximum.read_text().strip()
            used = int(current.read_text().strip())
            limit = total if raw == "max" else min(int(raw), total)
            if used < 0 or limit <= 0:
                raise ValueError("invalid memory limits")
            observed.append({"limit": raw, "current_bytes": used})
            limits.append(limit)
            available = min(available, max(0, limit - used))
    effective = min(limits) if limits else None
    eligible = (
        effective is not None
        and effective >= E4_MIN_LIMIT_BYTES
        and available >= E4_MIN_AVAILABLE_BYTES
    )
    return {
        "physical_total_bytes": total,
        "cgroup_observations": observed,
        "effective_limit_bytes": effective,
        "available_bytes": available,
        "e4_eligible": eligible,
        "e4_min_limit_bytes": E4_MIN_LIMIT_BYTES,
        "e4_min_available_bytes": E4_MIN_AVAILABLE_BYTES,
        "limit": "conservative snapshot; E4 skipped if unknown or below thresholds",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--freeze-commit", required=True)
    parser.add_argument("--idle-only", action="store_true")
    args = parser.parse_args()
    import runtime

    out = runtime.fresh(args.out)
    result = {"status": "unscored", "packet": runtime.packet_hashes()}
    try:
        result["public_freeze"] = runtime.public_freeze(args.freeze_commit)
        result["idle"] = idle_receipt()
        if args.idle_only:
            result["status"] = "idle_passed"
        else:
            result["memory"] = memory_headroom()
        # Build resources must be known; supporting E4 has the higher gate.
        if not args.idle_only and (
            result["memory"]["effective_limit_bytes"] is None
            or result["memory"]["available_bytes"] < 32 * GIB
        ):
            raise ValueError("insufficient or unknown build memory headroom")
        if not args.idle_only:
            result["status"] = "preflight_passed"
    except Exception as exc:  # noqa: BLE001 - fail cheaply in identity stage
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        runtime.write(out / "identity.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "e4_eligible": result.get("memory", {}).get("e4_eligible", False),
            }
        )
    )
    if result["status"] not in ("preflight_passed", "idle_passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
