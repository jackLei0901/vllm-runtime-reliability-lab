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
V2_GUIDE_VERSION = "2.0.0"
FAULT_DOMAINS = {"leaf", "lifecycle", "unknown"}
DOWNSTREAM = {"observed_normal", "unobserved"}
EXPOSURES = {"none_declared", "aggregate_only", "item_labels_seen", "unknown"}
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
V2_ENTRY_KEYS = ENTRY_KEYS | {
    "guide_version",
    "fault_domain",
    "downstream",
    "downstream_evidence",
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
HUMAN_REVIEW_KEYS = {
    "number",
    "decision",
    "matched_terms",
    "labelled_at",
    "title_sha256",
    "body_sha256",
    "updated_at",
    "exclusion_code",
    "v_label",
    "model_relation",
    "model_parts",
    "model_gap",
    "evidence",
    "guide_version",
    "fault_domain",
    "downstream",
    "downstream_evidence",
    "active_seconds",
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


def _commit(value: object, where: str) -> None:
    if not (
        isinstance(value, str)
        and len(value) == 40
        and all(c in "0123456789abcdef" for c in value)
    ):
        raise LedgerError(f"{where}: expected lowercase Git commit")


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


def _v2_fields(record: dict, where: str) -> None:
    if record.get("guide_version") != V2_GUIDE_VERSION:
        raise LedgerError(f"{where}: wrong v2 guide version")
    domain = record.get("fault_domain")
    if domain not in FAULT_DOMAINS:
        raise LedgerError(f"{where}: invalid fault domain")
    relation = record["model_relation"]
    if relation == "outside_model":
        if domain != "leaf" or record.get("downstream") not in DOWNSTREAM:
            raise LedgerError(f"{where}: outside_model needs leaf and downstream")
        if record["downstream"] == "observed_normal":
            ref = _object(
                record.get("downstream_evidence"),
                {"pointer", "grade"},
                f"{where}.downstream_evidence",
            )
            _text(ref.get("pointer"), f"{where}.downstream_evidence.pointer")
            if ref.get("grade") not in GRADES:
                raise LedgerError(f"{where}: invalid downstream evidence grade")
        elif "downstream_evidence" in record:
            raise LedgerError(f"{where}: unobserved cannot have downstream evidence")
    elif "downstream" in record or "downstream_evidence" in record:
        raise LedgerError(f"{where}: downstream applies only to outside_model")
    if relation == "insufficient_information" and domain != "unknown":
        raise LedgerError(f"{where}: insufficient needs unknown fault domain")


def _labeller(value: object) -> None:
    record = _object(
        value,
        {"kind", "model_id", "visible_instructions_sha256", "prior_v1_exposure"},
        "labeller",
    )
    if record.get("kind") not in {"ai", "human", "ai_with_human_review"}:
        raise LedgerError("labeller: invalid kind")
    _text(record.get("model_id"), "labeller.model_id")
    if record["kind"] == "human" and record["model_id"] != "not_applicable":
        raise LedgerError("labeller: human model_id must be not_applicable")
    _digest(
        record.get("visible_instructions_sha256"),
        "labeller.visible_instructions_sha256",
    )
    if record.get("prior_v1_exposure") not in EXPOSURES:
        raise LedgerError("labeller: invalid prior exposure")


def validate(
    snapshot_bytes: bytes,
    ledger: object,
    guide_bytes: bytes | None = None,
    v1_ledger: object | None = None,
    human_ledger: object | None = None,
) -> dict[str, object]:
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

    if not isinstance(ledger, dict):
        raise LedgerError("ledger: unexpected object shape")
    schema = ledger.get("schema_version")
    if schema not in {"q4-label-ledger-v1", "q4-label-ledger-v2"}:
        raise LedgerError("ledger: schema version mismatch")
    v2 = schema == "q4-label-ledger-v2"
    doc = _object(
        ledger,
        {
            "schema_version",
            "snapshot_sha256",
            "protocol_commit",
            "status",
            "entries",
            "relabels",
            *(
                {
                    "guide_commit",
                    "guide_sha256",
                    "labeller",
                    "v1_prefix_sha256",
                }
                if v2
                else set()
            ),
        },
        "ledger",
    )
    if v2:
        _commit(doc.get("guide_commit"), "ledger.guide_commit")
        _digest(doc.get("guide_sha256"), "ledger.guide_sha256")
        if (
            guide_bytes is None
            or doc["guide_sha256"] != hashlib.sha256(guide_bytes).hexdigest()
        ):
            raise LedgerError("ledger: guide bytes mismatch")
        _labeller(doc.get("labeller"))
        if "relabels" in doc:
            raise LedgerError("ledger: v2 uses a separate human ledger, not relabels")
        _digest(doc.get("v1_prefix_sha256"), "ledger.v1_prefix_sha256")
        if (
            not isinstance(v1_ledger, dict)
            or v1_ledger.get("schema_version") != "q4-label-ledger-v1"
        ):
            raise LedgerError("ledger: v2 requires original v1 ledger")
        validate(snapshot_bytes, v1_ledger)
        original_prefix = v1_ledger["entries"]
        prefix_bytes = json.dumps(
            original_prefix, sort_keys=True, separators=(",", ":")
        ).encode()
        if doc["v1_prefix_sha256"] != hashlib.sha256(prefix_bytes).hexdigest():
            raise LedgerError("ledger: original v1 prefix changed")
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
    if v2:
        for original, candidate in zip(original_prefix, entries, strict=False):
            for field in (
                "number",
                "decision",
                "title_sha256",
                "body_sha256",
                "matched_terms",
                "updated_at",
            ):
                if not isinstance(candidate, dict) or original.get(
                    field
                ) != candidate.get(field):
                    raise LedgerError(f"ledger: v2 changed v1 prefix {field}")
            if original.get("decision") == "exclude" and original.get(
                "exclusion_code"
            ) != candidate.get("exclusion_code"):
                raise LedgerError("ledger: v2 changed v1 exclusion")

    included = []
    missing_times = 0
    guide_issues = 0
    previous_labelled = snapshot_finished
    for index, raw in enumerate(entries):
        entry = _object(raw, V2_ENTRY_KEYS if v2 else ENTRY_KEYS, "entry")
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
        if v2 and entry.get("guide_version") != V2_GUIDE_VERSION:
            raise LedgerError("entry: wrong v2 guide version")
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
                *({"guide_version"} if v2 else set()),
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
        if v2:
            _v2_fields(entry, "entry")
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

    relabels = doc.get("relabels", []) if not v2 else []
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

    if human_ledger is not None and not v2:
        raise LedgerError("human_review: requires v2 ledger")
    human_entries = []
    reviewer = None
    if human_ledger is not None:
        hdoc = _object(
            human_ledger,
            {
                "schema_version",
                "snapshot_sha256",
                "protocol_commit",
                "guide_commit",
                "guide_sha256",
                "v1_prefix_sha256",
                "reviewer",
                "entries",
            },
            "human_ledger",
        )
        if hdoc.get("schema_version") != "q4-human-review-v2":
            raise LedgerError("human_ledger: schema version mismatch")
        for field in (
            "snapshot_sha256",
            "protocol_commit",
            "guide_commit",
            "guide_sha256",
            "v1_prefix_sha256",
        ):
            if hdoc.get(field) != doc.get(field):
                raise LedgerError(f"human_ledger: {field} differs from AI ledger")
        reviewer = _object(
            hdoc.get("reviewer"),
            {"kind", "prior_ai_exposure"},
            "human_ledger.reviewer",
        )
        if (
            reviewer.get("kind") != "human"
            or reviewer.get("prior_ai_exposure") not in EXPOSURES
        ):
            raise LedgerError("human_ledger: invalid reviewer provenance")
        human_entries = hdoc.get("entries")
        if not isinstance(human_entries, list) or len(human_entries) > len(order) - len(
            original_prefix
        ):
            raise LedgerError("human_ledger: invalid candidate prefix")
    human_agreement = 0
    human_parts_agreement = 0
    human_changed_sources = 0
    human_decision_disagreements = 0
    human_eligible = 0
    human_times = []
    previous_human_time = (
        _time(original_prefix[-1]["labelled_at"], "v1.last_labelled_at")
        if v2 and original_prefix
        else snapshot_finished
    )
    ai_by_number = {entry["number"]: entry for entry in entries}
    for index, raw in enumerate(human_entries):
        if human_eligible == 10:
            raise LedgerError("human_review: reviewed beyond tenth eligible")
        review = _object(raw, HUMAN_REVIEW_KEYS, "human_review")
        if review.get("number") != order[len(original_prefix) + index]:
            raise LedgerError("human_review: not post-v1 frozen-order prefix")
        if review.get("matched_terms") != [
            term for term in terms if term in term_matches[review["number"]]
        ]:
            raise LedgerError("human_review: matched terms differ from snapshot")
        reviewed_at = _time(review.get("labelled_at"), "human_review.labelled_at")
        if reviewed_at <= snapshot_finished:
            raise LedgerError("human_review: before snapshot finished")
        if (
            v2
            and original_prefix
            and reviewed_at
            <= _time(original_prefix[-1]["labelled_at"], "v1.last_labelled_at")
        ):
            raise LedgerError("human_review: before v1 prefix ended")
        if reviewed_at < previous_human_time:
            raise LedgerError("human_review: labelled_at decreases along review order")
        previous_human_time = reviewed_at
        if _time(review.get("updated_at"), "human_review.updated_at") > reviewed_at:
            raise LedgerError("human_review: source updated after label")
        _digest(review.get("title_sha256"), "human_review.title_sha256")
        _digest(review.get("body_sha256"), "human_review.body_sha256")
        if review.get("guide_version") != V2_GUIDE_VERSION:
            raise LedgerError("human_review: wrong v2 guide version")
        seconds = review.get("active_seconds")
        if type(seconds) not in {int, float} or not 0 <= seconds <= 86400:
            raise LedgerError("human_review: invalid active_seconds")
        if review.get("decision") == "exclude":
            if review.get("exclusion_code") not in EXCLUSIONS:
                raise LedgerError("human_review: invalid exclusion code")
            if set(review) - {
                "number",
                "decision",
                "matched_terms",
                "title_sha256",
                "body_sha256",
                "updated_at",
                "labelled_at",
                "active_seconds",
                "guide_version",
                "exclusion_code",
            }:
                raise LedgerError("human_review: excluded report has label fields")
        elif review.get("decision") == "include" and "exclusion_code" not in review:
            if review.get("v_label") not in V_LABELS:
                raise LedgerError("human_review: invalid V label")
            _model_evidence(review, "human_review")
            _v2_fields(review, "human_review")
            human_eligible += 1
            human_times.append(seconds)
        else:
            raise LedgerError("human_review: invalid inclusion decision")
        ai_entry = ai_by_number.get(review["number"])
        if ai_entry is None:
            continue
        if reviewed_at >= _time(ai_entry.get("labelled_at"), "entry.labelled_at"):
            raise LedgerError("human_review: human label was not before AI label")
        if any(
            review[field] != ai_entry[field]
            for field in ("title_sha256", "body_sha256", "updated_at")
        ):
            human_changed_sources += 1
        if review["decision"] != ai_entry["decision"] or (
            review["decision"] == "exclude"
            and review["exclusion_code"] != ai_entry["exclusion_code"]
        ):
            human_decision_disagreements += 1
        elif review["decision"] == "include":
            human_agreement += review["model_relation"] == ai_entry["model_relation"]
            human_parts_agreement += set(review["model_parts"]) == set(
                ai_entry["model_parts"]
            )

    timed_eligible = [
        e["active_seconds"] for e in included if e.get("active_seconds") is not None
    ]
    cost_scored = len(included) == 40 and len(timed_eligible) >= 35
    result = {
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
        "placement_cost_scored": cost_scored if not v2 else False,
        "median_timed_eligible_seconds": (
            median(timed_eligible) if cost_scored and not v2 else None
        ),
    }
    if v2:
        human_scored = (
            human_eligible == 10
            and all(entry["number"] in ai_by_number for entry in human_entries)
            and human_changed_sources == 0
            and human_decision_disagreements == 0
            and reviewer is not None
            and reviewer["prior_ai_exposure"] in {"none_declared", "aggregate_only"}
        )
        result.update(
            {
                "guide_version": V2_GUIDE_VERSION,
                "human_reviewed": len(human_entries),
                "human_eligible": human_eligible,
                "human_review_source_changed": human_changed_sources,
                "human_decision_disagreements": human_decision_disagreements,
                "human_relation_agreement": human_agreement if human_scored else None,
                "human_mpart_set_agreement": (
                    human_parts_agreement if human_scored else None
                ),
                "human_review_cost_scored": human_eligible == 10,
                "median_human_review_seconds": (
                    median(human_times) if human_eligible == 10 else None
                ),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--guide", type=Path, help="required for v2 ledger")
    parser.add_argument("--v1-ledger", type=Path, help="required for v2 ledger")
    parser.add_argument(
        "--human-ledger", type=Path, help="optional separate human pass"
    )
    args = parser.parse_args()
    try:
        result = validate(
            args.snapshot.read_bytes(),
            json.loads(args.ledger.read_text(encoding="utf-8")),
            args.guide.read_bytes() if args.guide else None,
            json.loads(args.v1_ledger.read_text(encoding="utf-8"))
            if args.v1_ledger
            else None,
            json.loads(args.human_ledger.read_text(encoding="utf-8"))
            if args.human_ledger
            else None,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, LedgerError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
