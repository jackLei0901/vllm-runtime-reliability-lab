"""Bounded TP=2 serving callback check; keep raw NCCL logs private."""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

RANK_LINE = re.compile(r"\[(\d+)\] NCCL INFO")
MARKERS = ("LLR_TP_EVT coll_start", "LLR_TP_EVT kernel_ch_start")


def snapshot(pattern: str) -> dict[int, dict[str, int]]:
    observed: dict[int, dict[str, int]] = {}
    for name in sorted(glob.glob(pattern)):
        data = Path(name).read_text(errors="replace")
        match = RANK_LINE.search(data)
        if match is None:
            raise ValueError("NCCL rank unavailable in a debug log")
        rank = int(match.group(1))
        if rank in observed:
            raise ValueError("duplicate NCCL rank debug log")
        observed[rank] = {
            "coll_start": data.count(MARKERS[0]),
            "kernel_ch_start": data.count(MARKERS[1]),
        }
    if set(observed) != {0, 1}:
        raise ValueError("expected exactly two rank-bound NCCL logs")
    return observed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--debug-glob", required=True)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    engine = LLM(
        model=args.model,
        tensor_parallel_size=2,
        disable_custom_all_reduce=True,
        gpu_memory_utilization=0.7,
        max_model_len=512,
        max_num_seqs=2,
        enforce_eager=False,
    )
    before = snapshot(args.debug_glob)
    output = engine.generate(
        ["Say hello."], SamplingParams(max_tokens=16, temperature=0)
    )
    after = snapshot(args.debug_glob)
    delta = {
        rank: {key: after[rank][key] - before[rank][key] for key in before[rank]}
        for rank in (0, 1)
    }
    result = {
        "schema": "tp-serving-callback-gate-v1",
        "tokens": len(output[0].outputs[0].token_ids),
        "before": before,
        "after": after,
        "delta": delta,
        "scorable": all(value > 0 for row in delta.values() for value in row.values()),
    }
    print("TP_SERVING_GATE " + json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["scorable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
