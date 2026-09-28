"""Freeze the preregistered Q4 closed-issue candidate IDs, without issue bodies.

Usage: python scripts/freeze_pain_point_candidates.py --protocol-commit COMMIT

The output is a resumable, fail-closed acquisition record. No issue body, title,
or comment is written to disk. A completed output must never be resumed.
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


REPO = "vllm-project/vllm"
TERMS = ("hang", "stuck", "deadlock", "crash", "timeout", "health", '"exit code"', '"no progress"')
CREATED_MONTHS = tuple(range(1, 9))
CLOSED = "2026-01-01..2026-09-25"
SEED = "dfxlab-coverage-20260926-v1:"
PER_PAGE = 100
OUTPUT = Path("data/pain-point-sample/candidate_snapshot_2026-09-28.json")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def digest_numbers(numbers: list[int]) -> str:
    payload = "".join(f"{n}\n" for n in numbers).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def save(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def query(term: str, start: date, end: date) -> str:
    return (
        f"repo:{REPO} is:issue is:closed in:title,body {term} "
        f"created:{start.isoformat()}..{end.isoformat()} closed:{CLOSED}"
    )


def api_page(q: str, page: int) -> dict:
    command = [
        "gh", "api", "-X", "GET", "search/issues", "-f", f"q={q}",
        "-f", f"per_page={PER_PAGE}", "-f", f"page={page}",
        "--jq", '{total_count,incomplete_results,items:[.items[]|{number,created_at,closed_at,is_pr:has("pull_request")}]}'
    ]
    for attempt in range(1, 5):
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        if result.returncode == 0:
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON for {q!r} page {page}") from exc
        if attempt == 4:
            raise RuntimeError(f"gh api failed for {q!r} page {page}: {result.stderr.strip()}")
        time.sleep(2 * attempt)
    raise AssertionError("unreachable")


def fetch(q: str, delay: float) -> dict:
    first = api_page(q, 1)
    total = first["total_count"]
    if first["incomplete_results"]:
        raise RuntimeError(f"incomplete_results=true for {q!r}")
    if total >= 900:
        return {"split_required": True, "query": q, "retrieved_at_utc": now(), "total_count": total,
                "page_count": 1, "incomplete_results": False}
    pages = max(1, (total + PER_PAGE - 1) // PER_PAGE)
    all_items = first["items"]
    page_times = [now()]
    for page in range(2, pages + 1):
        time.sleep(delay)
        part = api_page(q, page)
        if part["incomplete_results"] or part["total_count"] != total:
            raise RuntimeError(f"Incomplete or changing result for {q!r} page {page}")
        all_items.extend(part["items"])
        page_times.append(now())
    numbers = [item["number"] for item in all_items]
    if len(numbers) != total or len(set(numbers)) != total:
        raise RuntimeError(f"Total or duplicate mismatch for {q!r}: got {len(numbers)}, expected {total}")
    for item in all_items:
        if item["is_pr"] or not item["created_at"] or not item["closed_at"]:
            raise RuntimeError(f"Non-issue or missing date in {q!r}")
    ordered = sorted(numbers)
    return {
        "query": q, "retrieved_at_utc_by_page": page_times, "total_count": total,
        "page_count": pages, "incomplete_results": False,
        "numbers": ordered, "numbers_sha256_newline_decimal": digest_numbers(ordered),
    }


def fetch_span(term: str, start: date, end: date, delay: float) -> tuple[list[dict], list[dict]]:
    q = query(term, start, end)
    result = fetch(q, delay)
    if not result.get("split_required"):
        return [result], []
    if start == end:
        raise RuntimeError(f"Single-day Search result is near the 1,000-result ceiling: {q!r}")
    print(f"split high-count shard ({result['total_count']}): {q}", flush=True)
    cursor = start
    shards = []
    split_probes = [result]
    while cursor <= end:
        day_shards, day_probes = fetch_span(term, cursor, cursor, delay)
        shards.extend(day_shards)
        split_probes.extend(day_probes)
        cursor += timedelta(days=1)
        time.sleep(delay)
    return shards, split_probes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-commit", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--interval-seconds", type=float, default=2.2)
    args = parser.parse_args()
    if args.interval_seconds < 2.2:
        parser.error("interval-seconds must be at least 2.2 to pace Search requests")
    version = subprocess.run(["gh", "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    if args.output.exists():
        record = json.loads(args.output.read_text(encoding="utf-8"))
        if record["status"] == "complete" or record["protocol_commit"] != args.protocol_commit:
            raise RuntimeError("Existing snapshot is complete or belongs to a different protocol commit")
    else:
        record = {
            "status": "incomplete", "started_at_utc": now(), "protocol_commit": args.protocol_commit,
            "repository": REPO, "api": "GET /search/issues via gh api", "gh_version": version,
            "created_months": "2026-01..2026-08", "closed": CLOSED, "terms": list(TERMS),
            "per_page": PER_PAGE, "search_shards": [], "split_probes": [],
            "completed_term_month_keys": [],
            "candidate_numbers": [], "candidate_numbers_sha256_newline_decimal": None,
            "random_seed": SEED, "randomized_numbers": [], "randomized_numbers_sha256_newline_decimal": None,
        }
        save(args.output, record)
    completed = set(record["completed_term_month_keys"])
    for term in TERMS:
        for month in CREATED_MONTHS:
            key = f"{term}|2026-{month:02d}"
            if key in completed:
                continue
            start = date(2026, month, 1)
            end = date(2026, month, calendar.monthrange(2026, month)[1])
            shards, split_probes = fetch_span(term, start, end, args.interval_seconds)
            record["search_shards"].extend(shards)
            record["split_probes"].extend(split_probes)
            record["completed_term_month_keys"].append(key)
            save(args.output, record)
            completed.add(key)
            print(f"{len(completed)}/64 {key}: {sum(s['total_count'] for s in shards)} results", flush=True)
            time.sleep(args.interval_seconds)
    all_numbers = sorted({n for shard in record["search_shards"] for n in shard["numbers"]})
    randomized = sorted(all_numbers, key=lambda n: (hashlib.sha256(f"{SEED}{n}".encode("ascii")).hexdigest(), n))
    record["candidate_numbers"] = all_numbers
    record["candidate_numbers_sha256_newline_decimal"] = digest_numbers(all_numbers)
    record["randomized_numbers"] = randomized
    record["randomized_numbers_sha256_newline_decimal"] = digest_numbers(randomized)
    record["finished_at_utc"] = now()
    record["status"] = "complete"
    save(args.output, record)
    print(f"COMPLETE: {len(all_numbers)} candidates; sha256={record['candidate_numbers_sha256_newline_decimal']}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"INCOMPLETE: {exc}", file=sys.stderr)
        sys.exit(1)
