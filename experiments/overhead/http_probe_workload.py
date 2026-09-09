from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.request


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--interval", type=float, default=0.02)
    args = parser.parse_args()
    identity = {
        "kind": "http-health-probe-v1",
        "requests": args.requests,
        "interval": args.interval,
    }
    signature = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    latencies: list[float] = []
    success = 0
    start = time.perf_counter()
    for _ in range(args.requests):
        request_start = time.perf_counter()
        with urllib.request.urlopen(f"{args.base_url}/health", timeout=2) as response:
            success += response.status == 200
        latencies.append((time.perf_counter() - request_start) * 1000)
        time.sleep(args.interval)
    wall = time.perf_counter() - start
    result = {
        "workload_signature": signature,
        "request_count": args.requests,
        "success_count": success,
        "metrics": {
            "wall_seconds": wall,
            "request_throughput_per_second": args.requests / wall,
            "latency_mean_ms": statistics.fmean(latencies),
            "latency_p50_ms": percentile(latencies, 0.50),
            "latency_p95_ms": percentile(latencies, 0.95),
        },
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if success == args.requests else 1


if __name__ == "__main__":
    raise SystemExit(main())
