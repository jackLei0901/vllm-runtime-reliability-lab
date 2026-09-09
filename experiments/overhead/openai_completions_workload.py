from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return ordered[index]


@dataclass(frozen=True)
class RequestResult:
    ok: bool
    e2e_ms: float
    ttft_ms: float | None
    tpot_ms: float | None
    completion_tokens: int
    error_kind: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "e2e_ms": self.e2e_ms,
            "ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "completion_tokens": self.completion_tokens,
            "error_kind": self.error_kind,
        }


def request_once(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    seed: int,
    timeout: float,
) -> RequestResult:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": seed,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    first_token_at: float | None = None
    completion_tokens = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                choices = chunk.get("choices") or []
                if choices and choices[0].get("text") and first_token_at is None:
                    first_token_at = time.perf_counter()
                usage = chunk.get("usage")
                if (
                    isinstance(usage, dict)
                    and usage.get("completion_tokens") is not None
                ):
                    completion_tokens = int(usage["completion_tokens"])
    except urllib.error.HTTPError:
        elapsed = (time.perf_counter() - start) * 1000
        return RequestResult(False, elapsed, None, None, 0, "http_error")
    except urllib.error.URLError:
        elapsed = (time.perf_counter() - start) * 1000
        return RequestResult(False, elapsed, None, None, 0, "transport_error")
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        elapsed = (time.perf_counter() - start) * 1000
        return RequestResult(False, elapsed, None, None, 0, "malformed_stream")
    end = time.perf_counter()
    e2e_ms = (end - start) * 1000
    if first_token_at is None:
        return RequestResult(False, e2e_ms, None, None, completion_tokens, "no_token")
    ttft_ms = (first_token_at - start) * 1000
    tpot_ms = None
    if completion_tokens > 1:
        tpot_ms = (e2e_ms - ttft_ms) / (completion_tokens - 1)
    return RequestResult(True, e2e_ms, ttft_ms, tpot_ms, completion_tokens, None)


def numeric_summary(name: str, values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        f"{name}_mean": statistics.fmean(values),
        f"{name}_p50": percentile(values, 0.50),
        f"{name}_p95": percentile(values, 0.95),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup-requests", type=int, default=10)
    parser.add_argument("--prompt-repeat", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if args.requests <= 0 or args.concurrency <= 0 or args.prompt_repeat <= 0:
        parser.error("requests, concurrency and prompt-repeat must be positive")
    prompt = "runtime reliability evidence " * args.prompt_repeat
    identity = {
        "kind": "openai-completions-stream-v1",
        "model": args.model,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "warmup_requests": args.warmup_requests,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }
    signature = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    for index in range(args.warmup_requests):
        warmup = request_once(
            args.base_url,
            args.model,
            prompt,
            args.max_tokens,
            args.seed + index,
            args.timeout,
        )
        if not warmup.ok:
            print(json.dumps({"error": "warmup_failed", "detail": warmup.to_dict()}))
            return 1

    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                request_once,
                args.base_url,
                args.model,
                prompt,
                args.max_tokens,
                args.seed + args.warmup_requests + index,
                args.timeout,
            )
            for index in range(args.requests)
        ]
        results = [future.result() for future in futures]
    wall = time.perf_counter() - start
    successful = [result for result in results if result.ok]
    ttft = [result.ttft_ms for result in successful if result.ttft_ms is not None]
    tpot = [result.tpot_ms for result in successful if result.tpot_ms is not None]
    e2e = [result.e2e_ms for result in successful]
    completion_tokens = sum(result.completion_tokens for result in successful)
    metrics: dict[str, float] = {
        "wall_seconds": wall,
        "successful_requests": float(len(successful)),
        "failed_requests": float(len(results) - len(successful)),
        "request_throughput_per_second": len(successful) / wall,
        "completion_tokens": float(completion_tokens),
        "output_token_throughput_per_second": completion_tokens / wall,
    }
    metrics.update(numeric_summary("ttft_ms", ttft))
    metrics.update(numeric_summary("tpot_ms", tpot))
    metrics.update(numeric_summary("e2e_ms", e2e))
    output = {
        "workload_signature": signature,
        "identity": identity,
        "request_count": args.requests,
        "success_count": len(successful),
        "metrics": metrics,
        "requests": [result.to_dict() for result in results],
    }
    print(json.dumps(output, sort_keys=True))
    return 0 if len(successful) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
