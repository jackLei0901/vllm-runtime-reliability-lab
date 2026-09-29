"""Select the only cases the protocol's single fresh-process rerun may execute.

Inputs: node ids saved from ``pytest --collect-only -q`` before the first run
(summary lines are ignored) and that run's JSONL receipts. Each collected case
is classified from its receipts:

* ``scored``            has a ``result``                       -> never rerun
* ``failed_no_result``  ``start`` and ``end``, no ``result``   -> unscored, not rerun
* ``aborted``           ``start``, no ``end`` and no ``result`` -> unscored, not rerun
* ``context_skipped``   ``skipped_context_unusable``           -> rerun
* ``not_started``       no receipt at all                      -> rerun only if an
                        aborted case exists earlier in collection order

A case is eligible only for one of the two permitted causes: skipped after a
verified broken CUDA context (exactly one ``result`` with
``cuda_usable_after: false``, earlier in collection order), or never started
after a verified hard abort (exactly one ``aborted`` case, earlier in
collection order, and it is the last case that started). Anything else fails
closed: exit status 2, nothing printed, no rerun. On success, prints eligible
node ids in collection order; an empty output means stop, do not invoke pytest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EVENTS = {"start", "result", "end", "skipped_context_unusable"}


class NoRerun(ValueError):
    """The receipts do not prove a permitted rerun cause."""


def classify(collected: list[str], receipts: list[dict]) -> dict[str, str]:
    if len(set(collected)) != len(collected):
        raise NoRerun("duplicate collected node ids")
    events: dict[str, list[str]] = {case: [] for case in collected}
    for record in receipts:
        case, event = record.get("case"), record.get("event")
        if case not in events:
            raise NoRerun(f"receipt for a case that was not collected: {case}")
        if event not in EVENTS:
            raise NoRerun(f"unknown receipt event {event!r} for {case}")
        events[case].append(event)
    kinds = {}
    for case, seen in events.items():
        if any(seen.count(event) > 1 for event in EVENTS):
            raise NoRerun(f"repeated receipt event for {case}: {seen}")
        if "skipped_context_unusable" in seen:
            if len(seen) != 1:
                raise NoRerun(f"skipped case also has other receipts: {case}")
            kinds[case] = "context_skipped"
        elif "result" in seen:
            if "start" not in seen:
                raise NoRerun(f"result without start: {case}")
            kinds[case] = "scored"
        elif "start" in seen:
            kinds[case] = "failed_no_result" if "end" in seen else "aborted"
        elif seen:
            raise NoRerun(f"end without start: {case}")
        else:
            kinds[case] = "not_started"
    return kinds


def eligible(collected: list[str], receipts: list[dict]) -> list[str]:
    kinds = classify(collected, receipts)
    order = {case: index for index, case in enumerate(collected)}
    breakers = [
        r["case"]
        for r in receipts
        if r.get("event") == "result" and r.get("cuda_usable_after") is False
    ]
    aborted = [case for case, kind in kinds.items() if kind == "aborted"]
    started = [r["case"] for r in receipts if r.get("event") == "start"]
    if len(breakers) > 1 or len(aborted) > 1:
        raise NoRerun(f"more than one breaking case: {breakers} {aborted}")
    if aborted and started[-1] != aborted[0]:
        raise NoRerun(f"aborted case {aborted[0]} is not the last case that started")
    selected = []
    for case in collected:
        kind = kinds[case]
        if kind == "context_skipped":
            if not breakers or order[case] < order[breakers[0]]:
                raise NoRerun(f"{case} skipped without an earlier broken context")
            selected.append(case)
        elif kind == "not_started":
            if not aborted or order[case] < order[aborted[0]]:
                raise NoRerun(f"{case} never started without an earlier hard abort")
            selected.append(case)
    return selected


def read_collected(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if "::" in line and not line.startswith(" ")
    ]


def read_receipts(path: Path) -> list[dict]:
    try:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise NoRerun(f"unreadable receipts: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("collected", type=Path)
    parser.add_argument("receipts", type=Path)
    args = parser.parse_args()
    try:
        selected = eligible(
            read_collected(args.collected), read_receipts(args.receipts)
        )
    except NoRerun as exc:
        print(f"no rerun: {exc}", file=sys.stderr)
        return 2
    for case in selected:
        print(case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
