"""Validate a private Q4 label ledger against the frozen candidate snapshot.

This program never fetches an issue or decides its label. It reads local JSON
metadata only and prints aggregate validation results, not issue numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median

EXCLUSIONS = (
    "install_build",
    "accuracy_only",
    "feature_request",
    "performance_only_without_reliability_failure",
    "outside_runtime_scope",
    "not_an_incident_report",
)
V_LABELS = {"V1", "V2", "V3", "none_of_v0", "insufficient_information"}
RELATIONS = {"mapped", "outside_model", "model_gap", "insufficient_information"}
PARTS = {f"M{i}" for i in range(1, 7)}
GRADES = {"report_log_or_output", "reporter_narrative"}
ENTRY_KEYS = {
    "number",
    "decision",
    "title_sha256",
    "body_sha256",
    "matched_terms",
    "updated_at",
    "labelled_at",
    "active_seconds",
    "exclusion_code",
    "v_label",
    "model_relation",
    "model_parts",
    "evidence",
    "model_gap",
    "ping_detectable",
    "ping_basis",
    "duplicate_cluster",
    "closure_mode",
    "root_cause_status",
    "guide_issue",
    "comments",
    "signal_note",
    "incidental_observation",
}
RELABEL_KEYS = {
    "number",
    "labelled_at",
    "title_sha256",
    "body_sha256",
    "updated_at",
    "v_label",
    "model_relation",
    "model_parts",
    "model_gap",
    "evidence",
}


class LedgerError(ValueError):
    """A ledger cannot support the claimed sampling state."""


def _object(value: object, keys: set[str], where: str) -> dict:
    if not isinstance(value, dict) or set(value) - keys:
        raise LedgerError(f"{where}: unexpected object shape")
    return value


def _digest(value: object, where: str) -> None:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    ):
        raise LedgerError(f"{where}: expected lowercase SHA-256")


def _time(value: object, where: str) -> datetime:
    if not isinstance(value, str):
        raise LedgerError(f"{where}: timestamp missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LedgerError(f"{where}: invalid timestamp") from exc
    if parsed.utcoffset() != timedelta(0):
        raise LedgerError(f"{where}: UTC timestamp required")
    return parsed


def _text(value: object, where: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise LedgerError(f"{where}: nonempty text required")


def _parts(value: object, where: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(part, str) or part not in PARTS for part in value
    ):
        raise LedgerError(f"{where}: invalid model part")
    if len(set(value)) != len(value):
        raise LedgerError(f"{where}: duplicate model part")
    return value


def _model_evidence(record: dict, where: str) -> None:
    relation = record.get("model_relation")
    if relation not in RELATIONS:
        raise LedgerError(f"{where}: invalid model relation")
    parts = _parts(record.get("model_parts"), f"{where}.model_parts")
    if relation == "mapped" and not parts:
        raise LedgerError(f"{where}: mapped relation needs a model part")
    if relation in {"outside_model", "insufficient_information"} and parts:
        raise LedgerError(f"{where}: relation cannot have model parts")
    if relation == "model_gap":
        _text(record.get("model_gap"), f"{where}.model_gap")
    elif "model_gap" in record:
        raise LedgerError(f"{where}: model_gap requires gap relation")
    required = set(parts) | {"V"}
    if relation == "model_gap":
        required.add("gap")
    evidence = record.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != required:
        raise LedgerError(f"{where}: evidence must cover V and each model part")
    for ref in evidence.values():
        ref = _object(ref, {"pointer", "grade"}, "evidence")
        _text(ref.get("pointer"), "evidence.pointer")
        if ref.get("grade") not in GRADES:
            raise LedgerError("evidence: invalid grade")


def validate(snapshot_bytes: bytes, ledger: object) -> dict[str, object]:
    snapshot = json.loads(snapshot_bytes)
    if snapshot.get("status") != "complete":
        raise LedgerError("snapshot: incomplete acquisition")
    order = snapshot.get("randomized_numbers")
    candidates = snapshot.get("candidate_numbers")
    if not isinstance(order, list) or not isinstance(candidates, list):
        raise LedgerError("snapshot: candidate lists missing")
    if len(order) != len(candidates) or set(order) != set(candidates):
        raise LedgerError("snapshot: candidate membership differs")
    if len(set(order)) != len(order) or any(type(n) is not int for n in order):
        raise LedgerError("snapshot: duplicate or invalid number")
    snapshot_finished = _time(
        snapshot.get("finished_at_utc"), "snapshot.finished_at_utc"
    )
    seed = snapshot.get("random_seed")
    if not isinstance(seed, str) or order != sorted(
        candidates,
        key=lambda n: (hashlib.sha256(f"{seed}{n}".encode()).hexdigest(), n),
    ):
        raise LedgerError("snapshot: review order differs from frozen seed")
    terms = snapshot.get("terms")
    shards = snapshot.get("search_shards")
    if (
        not isinstance(terms, list)
        or not terms
        or any(not isinstance(term, str) or not term for term in terms)
        or len(set(terms)) != len(terms)
        or not isinstance(shards, list)
    ):
        raise LedgerError("snapshot: missing term provenance")
    term_matches: dict[int, set[str]] = {number: set() for number in candidates}
    for shard in shards:
        if not isinstance(shard, dict) or not isinstance(shard.get("query"), str):
            raise LedgerError("snapshot: invalid search shard")
        matched = [
            term for term in terms if f"in:title,body {term} created:" in shard["query"]
        ]
        if len(matched) != 1 or not isinstance(shard.get("numbers"), list):
            raise LedgerError("snapshot: ambiguous shard term")
        for number in shard["numbers"]:
            if type(number) is not int or number not in term_matches:
                raise LedgerError("snapshot: shard contains unknown candidate")
            term_matches[number].add(matched[0])
    if any(not matched for matched in term_matches.values()):
        raise LedgerError("snapshot: candidate has no matched term")

    doc = _object(
        ledger,
        {
            "schema_version",
            "snapshot_sha256",
            "protocol_commit",
            "status",
            "entries",
            "relabels",
        },
        "ledger",
    )
    if doc.get("schema_version") != "q4-label-ledger-v1":
        raise LedgerError("ledger: schema version mismatch")
    if doc.get("snapshot_sha256") != hashlib.sha256(snapshot_bytes).hexdigest():
        raise LedgerError("ledger: snapshot bytes mismatch")
    if doc.get("protocol_commit") != snapshot.get("protocol_commit"):
        raise LedgerError("ledger: protocol commit mismatch")
    status = doc.get("status")
    if status not in {"in_progress", "complete", "no_sample"}:
        raise LedgerError("ledger: invalid status")
    entries = doc.get("entries")
    if not isinstance(entries, list) or len(entries) > len(order):
        raise LedgerError("ledger: invalid reviewed prefix")

    included = []
    missing_times = 0
    guide_issues = 0
    previous_labelled = snapshot_finished
    for index, raw in enumerate(entries):
        entry = _object(raw, ENTRY_KEYS, "entry")
        if entry.get("number") != order[index]:
            raise LedgerError("entry: not the frozen review-order prefix")
        _digest(entry.get("title_sha256"), "entry.title_sha256")
        _digest(entry.get("body_sha256"), "entry.body_sha256")
        if entry.get("matched_terms") != [
            term for term in terms if term in term_matches[entry["number"]]
        ]:
            raise LedgerError("entry: matched terms differ from snapshot")
        updated = _time(entry.get("updated_at"), "entry.updated_at")
        labelled = _time(entry.get("labelled_at"), "entry.labelled_at")
        if labelled <= snapshot_finished:
            raise LedgerError("entry: labelled before snapshot finished")
        if labelled < previous_labelled:
            raise LedgerError("entry: labelled_at decreases along review order")
        previous_labelled = labelled
        if updated > labelled:
            raise LedgerError("entry: source updated after label")
        comments = entry.get("comments", [])
        if not isinstance(comments, list):
            raise LedgerError("entry: comments must be a list")
        comment_ids = set()
        for comment in comments:
            comment = _object(comment, {"id", "updated_at", "body_sha256"}, "comment")
            if type(comment.get("id")) is not int or comment["id"] <= 0:
                raise LedgerError("comment: invalid id")
            if comment["id"] in comment_ids:
                raise LedgerError("comment: duplicate id")
            comment_ids.add(comment["id"])
            if _time(comment.get("updated_at"), "comment.updated_at") > labelled:
                raise LedgerError("comment: source updated after label")
            _digest(comment.get("body_sha256"), "comment.body_sha256")
        seconds = entry.get("active_seconds")
        if seconds is None:
            missing_times += 1
        elif type(seconds) not in {int, float} or not 0 <= seconds <= 86400:
            raise LedgerError("entry: invalid active_seconds")
        decision = entry.get("decision")
        if decision == "exclude":
            if entry.get("exclusion_code") not in EXCLUSIONS:
                raise LedgerError("entry: invalid exclusion code")
            if set(entry) - {
                "number",
                "decision",
                "title_sha256",
                "body_sha256",
                "matched_terms",
                "updated_at",
                "labelled_at",
                "active_seconds",
                "exclusion_code",
                "comments",
            }:
                raise LedgerError("entry: excluded report has label fields")
            continue
        if decision != "include" or "exclusion_code" in entry:
            raise LedgerError("entry: invalid inclusion decision")
        if len(included) == 40:
            raise LedgerError("ledger: reviewed beyond the 40th eligible report")
        if entry.get("v_label") not in V_LABELS:
            raise LedgerError("entry: invalid V label")
        _model_evidence(entry, "entry")
        if entry.get("ping_detectable") not in {"yes", "no", "unknown"}:
            raise LedgerError("entry: invalid ping judgement")
        _text(entry.get("ping_basis"), "entry.ping_basis")
        _text(entry.get("closure_mode"), "entry.closure_mode")
        if entry.get("root_cause_status") not in {"known", "unknown"}:
            raise LedgerError("entry: invalid root-cause status")
        if "duplicate_cluster" in entry and entry["duplicate_cluster"] is not None:
            _text(entry["duplicate_cluster"], "entry.duplicate_cluster")
        if "guide_issue" in entry:
            _text(entry["guide_issue"], "entry.guide_issue")
            guide_issues += 1
        for note in ("signal_note", "incidental_observation"):
            if note in entry:
                _text(entry[note], f"entry.{note}")
        included.append(entry)

    if status == "complete" and (
        len(included) != 40 or entries[-1]["decision"] != "include"
    ):
        raise LedgerError("ledger: complete requires stopping at eligible item 40")
    if status == "no_sample" and (len(entries) != len(order) or len(included) >= 40):
        raise LedgerError("ledger: NO-SAMPLE requires candidate exhaustion")
    if status == "in_progress" and (len(entries) == len(order) or len(included) >= 40):
        raise LedgerError("ledger: progress status contradicts stop condition")

    relabels = doc.get("relabels", [])
    if not isinstance(relabels, list) or len(relabels) > 10:
        raise LedgerError("relabels: invalid list")
    if relabels and len(included) < 10:
        raise LedgerError("relabels: first ten eligible items not yet known")
    agreement = 0
    changed_sources = 0
    if relabels:
        earliest = max(
            _time(e["labelled_at"], "entry.labelled_at") for e in included[:10]
        )
        earliest += timedelta(days=7)
        for index, raw in enumerate(relabels):
            relabel = _object(raw, RELABEL_KEYS, "relabel")
            original = included[index]
            if relabel.get("number") != original["number"]:
                raise LedgerError("relabels: not the first-ten eligible prefix")
            relabelled = _time(relabel.get("labelled_at"), "relabel.labelled_at")
            if relabelled < earliest:
                raise LedgerError("relabels: seven-day wait not met")
            if (
                relabel.get("v_label") not in V_LABELS
                or relabel.get("model_relation") not in RELATIONS
            ):
                raise LedgerError("relabels: invalid label")
            _digest(relabel.get("body_sha256"), "relabel.body_sha256")
            _digest(relabel.get("title_sha256"), "relabel.title_sha256")
            if _time(relabel.get("updated_at"), "relabel.updated_at") > relabelled:
                raise LedgerError("relabel: source updated after label")
            _model_evidence(relabel, "relabel")
            if (
                relabel["body_sha256"] != original["body_sha256"]
                or relabel["title_sha256"] != original["title_sha256"]
                or relabel["updated_at"] != original["updated_at"]
            ):
                changed_sources += 1
            else:
                agreement += relabel["model_relation"] == original["model_relation"]

    timed_eligible = [
        e["active_seconds"] for e in included if e.get("active_seconds") is not None
    ]
    cost_scored = len(included) == 40 and len(timed_eligible) >= 35
    return {
        "status": status,
        "reviewed": len(entries),
        "eligible": len(included),
        "excluded": len(entries) - len(included),
        "guide_issues": guide_issues,
        "eligible_missing_time": sum(e.get("active_seconds") is None for e in included),
        "relabelled": len(relabels),
        "relabel_source_changed": changed_sources,
        "relation_agreement": (
            agreement if len(relabels) == 10 and changed_sources == 0 else None
        ),
        "all_reviewed_missing_time": missing_times,
        "outside_or_insufficient": sum(
            e["model_relation"] in {"outside_model", "insufficient_information"}
            for e in included
        ),
        "timed_eligible": len(timed_eligible),
        "placement_cost_scored": cost_scored,
        "median_timed_eligible_seconds": (
            median(timed_eligible) if cost_scored else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = validate(
            args.snapshot.read_bytes(),
            json.loads(args.ledger.read_text(encoding="utf-8")),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, LedgerError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
