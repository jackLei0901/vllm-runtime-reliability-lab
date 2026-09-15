"""Normalize Flight Recorder records while preserving producer absence.

Version 1 is retained byte-for-byte because historical evidence freezes its
hash. Version 2 requires an explicit expected producer set and distinguishes a
rank that produced no artifact from a producer whose artifact has no matching
collective entry.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from typing import Any


def _members(pg_config: dict[str, Any], pg_id: str) -> list[int]:
    config = pg_config.get(str(pg_id), {})
    value = config.get("ranks", config.get("global_ranks", config.get("members")))
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or not value:
        raise ValueError(f"process group {pg_id} has no global-rank membership")
    return sorted(int(rank) for rank in value)


def _operation(entry: dict[str, Any]) -> str:
    name = str(entry.get("profiling_name", "unknown"))
    return name.rsplit(":", 1)[-1].upper()


def _process_group_id(entry: dict[str, Any]) -> str:
    process_group = entry.get("process_group")
    if isinstance(process_group, (list, tuple)) and process_group:
        return str(process_group[0])
    return str(entry["pg_id"])


def _producer_sets(
    artifacts: list[dict[str, Any]], expected_producer_ranks: Iterable[int]
) -> tuple[list[int], list[int], list[int]]:
    expected_values = [int(rank) for rank in expected_producer_ranks]
    expected = sorted(set(expected_values))
    if not expected or len(expected) != len(expected_values):
        raise ValueError("expected producer ranks must be non-empty and unique")

    present_values = [int(artifact["rank"]) for artifact in artifacts]
    present = sorted(set(present_values))
    if len(present) != len(present_values):
        raise ValueError("each producer rank must have exactly one artifact")

    unexpected = sorted(set(present) - set(expected))
    if unexpected:
        raise ValueError(f"unexpected producer ranks: {unexpected}")
    missing = sorted(set(expected) - set(present))
    return expected, present, missing


def _absence_reason(missing_producers: list[int], missing_members: list[int]) -> str:
    if missing_producers and missing_members:
        return "producer_and_member_missing"
    if missing_producers:
        return "producer_missing"
    if missing_members:
        return "member_missing"
    raise ValueError("absence reason requested without an absent rank")


def normalize_rank_artifacts(
    artifacts: list[dict[str, Any]], *, expected_producer_ranks: Iterable[int]
) -> dict[str, Any]:
    """Return a bounded divergence summary from decoded per-rank artifacts.

    ``expected_producer_ranks`` is capture-plane knowledge supplied by the
    caller. It must not be inferred from the dumps that happened to arrive.
    """
    expected, present_producers, missing_producers = _producer_sets(
        artifacts, expected_producer_ranks
    )
    missing_producer_set = set(missing_producers)
    expected_set = set(expected)

    grouped: dict[tuple[tuple[int, ...], bool, int], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    completed_before = 0
    for artifact in artifacts:
        rank = int(artifact["rank"])
        pg_config = artifact["pg_config"]
        for entry in sorted(artifact["entries"], key=lambda value: value["record_id"]):
            operation = _operation(entry)
            if bool(entry.get("is_p2p", False)) or operation == "COALESCED":
                continue
            pg_id = _process_group_id(entry)
            members = _members(pg_config, pg_id)
            undeclared = sorted(set(members) - expected_set)
            if undeclared:
                raise ValueError(
                    f"process group {pg_id} contains undeclared producers: {undeclared}"
                )
            sequence = int(entry["collective_seq_id"])
            grouped[(tuple(members), False, sequence)].append(
                {
                    "rank": rank,
                    "operation": operation,
                    "input_sizes": entry.get("input_sizes", []),
                    "input_dtypes": entry.get("input_dtypes", []),
                    "output_sizes": entry.get("output_sizes", []),
                    "output_dtypes": entry.get("output_dtypes", []),
                    "state": entry.get("state", "unknown"),
                    "record_id": int(entry["record_id"]),
                    "thread_id": entry.get("thread_id"),
                    "thread_name": entry.get("thread_name"),
                }
            )
            if entry.get("state") == "completed":
                completed_before += 1

    primary = None
    secondary = []
    for (members, is_p2p, sequence), participants in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])
    ):
        by_rank: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for participant in participants:
            by_rank[participant["rank"]].append(participant)
        present = sorted(by_rank)
        absent = sorted(set(members) - set(present))
        absent_producers = sorted(set(absent) & missing_producer_set)
        absent_members = sorted(set(absent) - missing_producer_set)
        absence_fields = {
            "missing_producer_ranks": absent_producers,
            "missing_member_ranks": absent_members,
        }

        ambiguous_ranks = [rank for rank, values in by_rank.items() if len(values) != 1]
        if ambiguous_ranks:
            secondary.append(
                {
                    "sequence_number": sequence,
                    "group_members": list(members),
                    "present_ranks": present,
                    "is_p2p": is_p2p,
                    "reason": "ambiguous_rank_entries",
                    **absence_fields,
                }
            )
            continue

        participants = [by_rank[rank][0] for rank in present]
        signature_values = {
            (
                item["operation"],
                json.dumps(item["input_sizes"], sort_keys=True),
                json.dumps(item["input_dtypes"], sort_keys=True),
                json.dumps(item["output_sizes"], sort_keys=True),
                json.dumps(item["output_dtypes"], sort_keys=True),
            )
            for item in participants
        }
        all_members_present = not absent
        if all_members_present and len(signature_values) > 1 and primary is None:
            operations = {item["operation"] for item in participants}
            primary = {
                "sequence_number": sequence,
                "operation": operations.pop() if len(operations) == 1 else "MIXED",
                "group_members": list(members),
                "is_p2p": is_p2p,
                "input_shapes_by_rank": [
                    {"rank": item["rank"], "shapes": item["input_sizes"]}
                    for item in sorted(participants, key=lambda value: value["rank"])
                ],
                "input_dtypes_by_rank": [
                    {"rank": item["rank"], "dtypes": item["input_dtypes"]}
                    for item in sorted(participants, key=lambda value: value["rank"])
                ],
            }
            continue

        all_pending = all(item["state"] != "completed" for item in participants)
        if absent or all_pending:
            secondary.append(
                {
                    "sequence_number": sequence,
                    "group_members": list(members),
                    "present_ranks": present,
                    "is_p2p": is_p2p,
                    "reason": (
                        _absence_reason(absent_producers, absent_members)
                        if absent
                        else "all_pending"
                    ),
                    **absence_fields,
                }
            )

    bounded = {
        "expected_producer_ranks": expected,
        "present_producer_ranks": present_producers,
        "missing_producer_ranks": missing_producers,
        "primary_divergence": primary,
        "secondary_divergences": secondary,
        "completed_collective_entries": completed_before,
    }
    canonical = json.dumps(bounded, sort_keys=True, separators=(",", ":"))
    return {
        **bounded,
        "normalized_fingerprint": hashlib.sha256(canonical.encode()).hexdigest(),
    }
