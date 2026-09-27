"""Closed-shape aggregate of legacy private Inspector callbacks.

This audits a failed identity assumption; it never recovers an in-flight
verdict and never emits communicator keys, PIDs, paths or raw log lines.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

RANK = re.compile(r"PROFILER/Plugin: init .*\bnranks: 2 rank: (\d+)(?:\s|$)")
COLL = re.compile(
    r"LLR_TP_EVT coll_start comm=([0-9a-f]{16}) seq=(\d+) "
    r"channels=(\d+)(?:\s|$)"
)
MAX_LOG_BYTES = 32 * 1024 * 1024


def summarize(pattern: str) -> dict[int, dict]:
    paths = sorted(glob.glob(pattern))
    if len(paths) != 2:
        raise ValueError("expected exactly two rank logs")
    result = {}
    for name in paths:
        path = Path(name)
        if path.stat().st_size > MAX_LOG_BYTES:
            raise ValueError("private NCCL log exceeded budget")
        content = path.read_text(encoding="utf-8", errors="replace")
        ranks = set(RANK.findall(content))
        if len(ranks) != 1 or int(next(iter(ranks))) not in (0, 1):
            raise ValueError("rank binding unavailable")
        rank = int(next(iter(ranks)))
        if rank in result:
            raise ValueError("duplicate rank log")
        by_key = defaultdict(list)
        for line in content.splitlines(keepends=True):
            if not line.endswith("\n") or "LLR_TP_EVT coll_start" not in line:
                continue
            match = COLL.search(line)
            if match is None:
                raise ValueError("malformed legacy collective marker")
            by_key[(match[1], int(match[2]))].append(int(match[3]))
        conflicts = sorted(
            (values for values in by_key.values() if len(set(values)) > 1),
            key=len,
            reverse=True,
        )
        if not conflicts:
            largest_runs = []
            histogram = {}
        else:
            largest_runs = []
            previous = None
            for channel_count in conflicts[0]:
                if channel_count != previous:
                    largest_runs.append(1)
                    previous = channel_count
                else:
                    largest_runs[-1] += 1
            histogram = dict(sorted(Counter(conflicts[0]).items()))
        result[rank] = {
            "coll_start_total": sum(map(len, by_key.values())),
            "distinct_legacy_keys": len(by_key),
            "conflicting_legacy_keys": len(conflicts),
            "conflict_sizes": [len(values) for values in conflicts],
            "largest_conflict_run_lengths": largest_runs,
            "largest_conflict_channel_histogram": histogram,
        }
    if set(result) != {0, 1}:
        raise ValueError("rank binding unavailable")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", required=True, help="private NCCL log glob")
    args = parser.parse_args()
    print(
        json.dumps(
            {"schema": "tp-legacy-callback-aggregate-v1", "ranks": summarize(args.logs)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
