"""Mutation controls for the private Q4 sample-record validator."""

from __future__ import annotations

import hashlib
import json
import unittest
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.validate_pain_point_labels import LedgerError, validate

SNAPSHOT = (
    Path(__file__).resolve().parents[1]
    / "data/pain-point-sample/candidate_snapshot_2026-09-28.json"
)
LABELLED_AT = datetime(2026, 10, 5, tzinfo=timezone.utc)
V2_GUIDE = b"synthetic frozen guide v2"


def fixture(size: int = 45) -> tuple[bytes, dict]:
    numbers = list(range(100, 100 + size))
    seed = "synthetic-test:"
    order = sorted(
        numbers,
        key=lambda n: (hashlib.sha256(f"{seed}{n}".encode()).hexdigest(), n),
    )
    snapshot = json.dumps(
        {
            "status": "complete",
            "candidate_numbers": numbers,
            "randomized_numbers": order,
            "random_seed": seed,
            "protocol_commit": "f" * 40,
            "finished_at_utc": "2026-09-28T07:25:30+00:00",
            "terms": ["hang"],
            "search_shards": [
                {
                    "query": (
                        "repo:test is:issue in:title,body hang "
                        "created:2026-01-01..2026-01-31"
                    ),
                    "numbers": numbers,
                }
            ],
        }
    ).encode()
    entries = [
        {
            "number": n,
            "decision": "include",
            "title_sha256": "c" * 64,
            "body_sha256": "a" * 64,
            "matched_terms": ["hang"],
            "updated_at": "2026-09-25T00:00:00+00:00",
            "labelled_at": (LABELLED_AT + timedelta(minutes=i)).isoformat(),
            "active_seconds": 300,
            "v_label": "V1",
            "model_relation": "mapped",
            "model_parts": ["M3"],
            "evidence": {
                "V": {"pointer": "body:L1", "grade": "reporter_narrative"},
                "M3": {"pointer": "body:L2", "grade": "report_log_or_output"},
            },
            "ping_detectable": "unknown",
            "ping_basis": "No ping measurement in this synthetic report",
            "closure_mode": "closed",
            "root_cause_status": "unknown",
        }
        for i, n in enumerate(order[:40])
    ]
    ledger = {
        "schema_version": "q4-label-ledger-v1",
        "snapshot_sha256": hashlib.sha256(snapshot).hexdigest(),
        "protocol_commit": "f" * 40,
        "status": "complete",
        "entries": entries,
    }
    return snapshot, ledger


def v2_fixture() -> tuple[bytes, dict, dict]:
    snapshot, original = fixture()
    ledger = deepcopy(original)
    prefix_bytes = json.dumps(
        original["entries"], sort_keys=True, separators=(",", ":")
    ).encode()
    ledger.update(
        {
            "schema_version": "q4-label-ledger-v2",
            "guide_commit": "e" * 40,
            "guide_sha256": hashlib.sha256(V2_GUIDE).hexdigest(),
            "v1_prefix_sha256": hashlib.sha256(prefix_bytes).hexdigest(),
            "labeller": {
                "kind": "ai",
                "model_id": "not_exposed",
                "visible_instructions_sha256": hashlib.sha256(V2_GUIDE).hexdigest(),
                "prior_v1_exposure": "item_labels_seen",
            },
        }
    )
    for entry in ledger["entries"]:
        entry["guide_version"] = "2.0.0-rc1"
        entry["fault_domain"] = "unknown"
    return snapshot, ledger, original


class PainPointLabelValidatorTests(unittest.TestCase):
    def check_error(self, snapshot: bytes, ledger: dict, message: str) -> None:
        with self.assertRaisesRegex(LedgerError, message):
            validate(snapshot, ledger)

    def test_complete_synthetic_prefix_and_counts(self) -> None:
        snapshot, ledger = fixture()
        result = validate(snapshot, ledger)
        self.assertEqual((result["reviewed"], result["eligible"]), (40, 40))
        self.assertEqual(result["relation_agreement"], None)
        self.assertEqual(result["median_timed_eligible_seconds"], 300)
        self.assertEqual(result["outside_or_insufficient"], 0)

    def test_real_snapshot_accepts_empty_in_progress_ledger(self) -> None:
        snapshot = SNAPSHOT.read_bytes()
        manifest = json.loads(snapshot)
        ledger = {
            "schema_version": "q4-label-ledger-v1",
            "snapshot_sha256": hashlib.sha256(snapshot).hexdigest(),
            "protocol_commit": manifest["protocol_commit"],
            "status": "in_progress",
            "entries": [],
        }
        self.assertEqual(validate(snapshot, ledger)["eligible"], 0)

    def test_reorder_and_duplicate_cannot_change_prefix(self) -> None:
        snapshot, ledger = fixture()
        ledger["entries"][0], ledger["entries"][1] = (
            ledger["entries"][1],
            ledger["entries"][0],
        )
        self.check_error(snapshot, ledger, "review-order prefix")
        ledger["entries"][1] = deepcopy(ledger["entries"][0])
        self.check_error(snapshot, ledger, "review-order prefix")

    def test_exclusion_code_and_label_fields_are_closed(self) -> None:
        snapshot, ledger = fixture()
        first = ledger["entries"][0]
        first["decision"] = "exclude"
        first["exclusion_code"] = "not_registered"
        self.check_error(snapshot, ledger, "exclusion code")
        first["exclusion_code"] = "install_build"
        self.check_error(snapshot, ledger, "excluded report has label fields")

    def test_missing_evidence_and_invalid_model_relation_fail(self) -> None:
        snapshot, ledger = fixture()
        del ledger["entries"][0]["evidence"]["M3"]
        self.check_error(snapshot, ledger, "evidence must cover")
        ledger["entries"][0]["evidence"]["M3"] = {
            "pointer": "body:L2",
            "grade": "report_log_or_output",
        }
        ledger["entries"][0]["model_relation"] = "outside_model"
        self.check_error(snapshot, ledger, "relation cannot have model parts")

    def test_title_digest_and_matched_terms_are_required(self) -> None:
        snapshot, ledger = fixture()
        del ledger["entries"][0]["title_sha256"]
        self.check_error(snapshot, ledger, "entry.title_sha256")
        ledger["entries"][0]["title_sha256"] = "c" * 64
        ledger["entries"][0]["matched_terms"] = ["stuck"]
        self.check_error(snapshot, ledger, "matched terms differ")

    def test_snapshot_shard_membership_is_checked(self) -> None:
        snapshot, ledger = fixture()
        manifest = json.loads(snapshot)
        manifest["search_shards"][0]["numbers"].pop()
        changed = json.dumps(manifest).encode()
        ledger["snapshot_sha256"] = hashlib.sha256(changed).hexdigest()
        self.check_error(changed, ledger, "candidate has no matched term")

    def test_model_gap_needs_a_separate_evidence_pointer(self) -> None:
        snapshot, ledger = fixture()
        first = ledger["entries"][0]
        first["model_relation"] = "model_gap"
        first["model_gap"] = "Synthetic lifecycle boundary not in M1-M6"
        self.check_error(snapshot, ledger, "evidence must cover")
        first["evidence"]["gap"] = {
            "pointer": "body:L3",
            "grade": "reporter_narrative",
        }
        self.assertEqual(validate(snapshot, ledger)["eligible"], 40)

    def test_complete_cannot_stop_early_or_review_past_40(self) -> None:
        snapshot, ledger = fixture()
        ledger["entries"].pop()
        self.check_error(snapshot, ledger, "stopping at eligible item 40")
        snapshot, ledger = fixture()
        order = json.loads(snapshot)["randomized_numbers"]
        extra = deepcopy(ledger["entries"][-1])
        extra["number"] = order[40]
        extra["labelled_at"] = (LABELLED_AT + timedelta(minutes=40)).isoformat()
        ledger["entries"].append(extra)
        self.check_error(snapshot, ledger, "reviewed beyond the 40th eligible")

    def test_label_times_follow_snapshot_and_frozen_order(self) -> None:
        snapshot, ledger = fixture()
        ledger["entries"][0]["labelled_at"] = "2026-09-01T00:00:00+00:00"
        self.check_error(snapshot, ledger, "before snapshot finished")
        snapshot, ledger = fixture()
        ledger["entries"][1]["labelled_at"] = ledger["entries"][0]["labelled_at"]
        self.assertEqual(validate(snapshot, ledger)["eligible"], 40)
        ledger["entries"][1]["labelled_at"] = (
            LABELLED_AT - timedelta(seconds=1)
        ).isoformat()
        self.check_error(snapshot, ledger, "decreases along review order")

    def test_source_versions_cannot_postdate_labels(self) -> None:
        snapshot, ledger = fixture()
        ledger["entries"][0]["updated_at"] = "2026-10-06T00:00:00+00:00"
        self.check_error(snapshot, ledger, "entry: source updated after label")
        snapshot, ledger = fixture()
        ledger["entries"][0]["comments"] = [
            {
                "id": 123,
                "updated_at": "2026-10-06T00:00:00+00:00",
                "body_sha256": "b" * 64,
            }
        ]
        self.check_error(snapshot, ledger, "comment: source updated after label")

    def test_time_limit_is_not_no_sample(self) -> None:
        snapshot, ledger = fixture()
        ledger["entries"] = ledger["entries"][:10]
        ledger["status"] = "no_sample"
        self.check_error(snapshot, ledger, "candidate exhaustion")
        ledger["status"] = "in_progress"
        self.assertEqual(validate(snapshot, ledger)["eligible"], 10)

    def test_relabel_waits_seven_days_after_tenth_initial_label(self) -> None:
        snapshot, ledger = fixture()
        tenth = datetime.fromisoformat(ledger["entries"][9]["labelled_at"])
        ledger["relabels"] = [
            {
                "number": e["number"],
                "labelled_at": (tenth + timedelta(days=7)).isoformat(),
                "title_sha256": e["title_sha256"],
                "body_sha256": e["body_sha256"],
                "updated_at": e["updated_at"],
                "v_label": "V1",
                "model_relation": "mapped",
                "model_parts": ["M3"],
                "evidence": deepcopy(e["evidence"]),
            }
            for e in ledger["entries"][:10]
        ]
        self.assertEqual(validate(snapshot, ledger)["relation_agreement"], 10)
        ledger["relabels"][0]["labelled_at"] = (
            tenth + timedelta(days=7, seconds=-1)
        ).isoformat()
        self.check_error(snapshot, ledger, "seven-day wait")

    def test_relabelled_changed_report_is_not_scored(self) -> None:
        snapshot, ledger = fixture()
        tenth = datetime.fromisoformat(ledger["entries"][9]["labelled_at"])
        ledger["relabels"] = [
            {
                "number": e["number"],
                "labelled_at": (tenth + timedelta(days=7)).isoformat(),
                "title_sha256": e["title_sha256"],
                "body_sha256": e["body_sha256"],
                "updated_at": e["updated_at"],
                "v_label": "V1",
                "model_relation": "mapped",
                "model_parts": ["M3"],
                "evidence": deepcopy(e["evidence"]),
            }
            for e in ledger["entries"][:10]
        ]
        ledger["relabels"][0]["body_sha256"] = "b" * 64
        result = validate(snapshot, ledger)
        self.assertEqual(result["relabel_source_changed"], 1)
        self.assertIsNone(result["relation_agreement"])
        del ledger["relabels"][0]["evidence"]["M3"]
        self.check_error(snapshot, ledger, "evidence must cover")

    def test_relabel_source_cannot_postdate_relabel(self) -> None:
        snapshot, ledger = fixture()
        tenth = datetime.fromisoformat(ledger["entries"][9]["labelled_at"])
        ledger["relabels"] = [
            {
                "number": ledger["entries"][0]["number"],
                "labelled_at": (tenth + timedelta(days=7)).isoformat(),
                "title_sha256": "c" * 64,
                "body_sha256": "a" * 64,
                "updated_at": (tenth + timedelta(days=8)).isoformat(),
                "v_label": "V1",
                "model_relation": "mapped",
                "model_parts": ["M3"],
                "evidence": deepcopy(ledger["entries"][0]["evidence"]),
            }
        ]
        self.check_error(snapshot, ledger, "relabel: source updated after label")

    def test_snapshot_and_protocol_identity_are_bound(self) -> None:
        snapshot, ledger = fixture()
        ledger["snapshot_sha256"] = "0" * 64
        self.check_error(snapshot, ledger, "snapshot bytes mismatch")
        ledger["snapshot_sha256"] = hashlib.sha256(snapshot).hexdigest()
        ledger["protocol_commit"] = "0" * 40
        self.check_error(snapshot, ledger, "protocol commit mismatch")

    def test_comment_identity_is_closed_without_comment_text(self) -> None:
        snapshot, ledger = fixture()
        ledger["entries"][0]["comments"] = [
            {
                "id": 123,
                "updated_at": "2026-09-25T00:00:00+00:00",
                "body_sha256": "b" * 64,
            }
        ]
        self.assertEqual(validate(snapshot, ledger)["eligible"], 40)
        ledger["entries"][0]["comments"][0]["body"] = "raw comment text"
        self.check_error(snapshot, ledger, "unexpected object shape")

    def test_v2_accepts_separate_ledger_without_changing_v1(self) -> None:
        snapshot, ledger, original = v2_fixture()
        result = validate(snapshot, ledger, V2_GUIDE, original)
        self.assertEqual(result["guide_version"], "2.0.0-rc1")
        self.assertEqual(result["eligible"], 40)
        self.assertIsNone(result["human_relation_agreement"])
        self.assertFalse(result["placement_cost_scored"])
        self.assertEqual(validate(*fixture())["eligible"], 40)

    def test_v2_requires_exact_guide_and_labeller_provenance(self) -> None:
        snapshot, ledger, original = v2_fixture()
        with self.assertRaisesRegex(LedgerError, "guide bytes mismatch"):
            validate(snapshot, ledger, v1_ledger=original)
        with self.assertRaisesRegex(LedgerError, "guide bytes mismatch"):
            validate(snapshot, ledger, b"changed", original)
        ledger["labeller"]["prior_v1_exposure"] = "blind"
        with self.assertRaisesRegex(LedgerError, "invalid prior exposure"):
            validate(snapshot, ledger, V2_GUIDE, original)
        ledger["labeller"]["prior_v1_exposure"] = "item_labels_seen"
        ledger["labeller"]["raw_prompt"] = "not allowed"
        with self.assertRaisesRegex(LedgerError, "unexpected object shape"):
            validate(snapshot, ledger, V2_GUIDE, original)

    def test_v2_outside_model_distinguishes_unobserved_from_normal(self) -> None:
        snapshot, ledger, original = v2_fixture()
        first = ledger["entries"][0]
        first["model_relation"] = "outside_model"
        first["model_parts"] = []
        first["evidence"] = {"V": first["evidence"]["V"]}
        first["fault_domain"] = "leaf"
        first["downstream"] = "unobserved"
        self.assertEqual(validate(snapshot, ledger, V2_GUIDE, original)["eligible"], 40)
        first["downstream"] = "observed_normal"
        with self.assertRaisesRegex(LedgerError, "unexpected object shape"):
            validate(snapshot, ledger, V2_GUIDE, original)
        first["downstream_evidence"] = {
            "pointer": "body:L9",
            "grade": "report_log_or_output",
        }
        self.assertEqual(validate(snapshot, ledger, V2_GUIDE, original)["eligible"], 40)
        first["downstream"] = "unobserved"
        with self.assertRaisesRegex(LedgerError, "unobserved cannot"):
            validate(snapshot, ledger, V2_GUIDE, original)
        del first["downstream_evidence"]
        first["fault_domain"] = "unknown"
        with self.assertRaisesRegex(LedgerError, "outside_model needs leaf"):
            validate(snapshot, ledger, V2_GUIDE, original)

    def test_v2_relation_and_exclusion_fields_are_closed(self) -> None:
        snapshot, ledger, original = v2_fixture()
        first = ledger["entries"][0]
        first["model_relation"] = "insufficient_information"
        first["model_parts"] = []
        first["evidence"] = {"V": first["evidence"]["V"]}
        with self.assertRaisesRegex(LedgerError, "insufficient needs unknown"):
            first["fault_domain"] = "leaf"
            validate(snapshot, ledger, V2_GUIDE, original)
        first["fault_domain"] = "unknown"
        self.assertEqual(validate(snapshot, ledger, V2_GUIDE, original)["eligible"], 40)
        first["decision"] = "exclude"
        first["exclusion_code"] = "accuracy_only"
        with self.assertRaisesRegex(LedgerError, "changed v1 prefix decision"):
            validate(snapshot, ledger, V2_GUIDE, original)

    def test_v2_human_agreement_requires_full_source_stable_pass(self) -> None:
        snapshot, ledger, original = v2_fixture()
        ledger["human_reviewer"] = {
            "kind": "human",
            "prior_ai_exposure": "aggregate_only",
        }
        ledger["human_reviews"] = [
            {
                "number": entry["number"],
                "labelled_at": (LABELLED_AT + timedelta(days=1, minutes=i)).isoformat(),
                "title_sha256": entry["title_sha256"],
                "body_sha256": entry["body_sha256"],
                "updated_at": entry["updated_at"],
                "guide_version": "2.0.0-rc1",
                "fault_domain": "unknown",
                "v_label": "V1",
                "model_relation": "mapped",
                "model_parts": ["M3"],
                "evidence": deepcopy(entry["evidence"]),
                "active_seconds": 600,
            }
            for i, entry in enumerate(ledger["entries"][:10])
        ]
        result = validate(snapshot, ledger, V2_GUIDE, original)
        self.assertEqual(result["human_relation_agreement"], 10)
        self.assertEqual(result["human_mpart_set_agreement"], 10)
        self.assertEqual(result["median_human_review_seconds"], 600)
        ledger["human_reviews"][0]["body_sha256"] = "b" * 64
        self.assertIsNone(
            validate(snapshot, ledger, V2_GUIDE, original)["human_relation_agreement"]
        )
        ledger["human_reviews"][0]["body_sha256"] = "a" * 64
        ledger["human_reviewer"]["prior_ai_exposure"] = "item_labels_seen"
        self.assertIsNone(
            validate(snapshot, ledger, V2_GUIDE, original)["human_relation_agreement"]
        )
        ledger["human_reviewer"]["prior_ai_exposure"] = "aggregate_only"
        ledger["human_reviews"][0]["active_seconds"] = -1
        with self.assertRaisesRegex(LedgerError, "invalid active_seconds"):
            validate(snapshot, ledger, V2_GUIDE, original)
        ledger["human_reviews"][0]["active_seconds"] = 600
        ledger["human_reviews"][1]["labelled_at"] = (
            LABELLED_AT + timedelta(days=1, seconds=-1)
        ).isoformat()
        with self.assertRaisesRegex(LedgerError, "decreases along review order"):
            validate(snapshot, ledger, V2_GUIDE, original)

    def test_v2_cannot_change_v1_prefix_or_exclusions(self) -> None:
        snapshot, ledger, original = v2_fixture()
        ledger["entries"][0]["body_sha256"] = "b" * 64
        with self.assertRaisesRegex(LedgerError, "changed v1 prefix body_sha256"):
            validate(snapshot, ledger, V2_GUIDE, original)
        ledger["entries"][0]["body_sha256"] = "a" * 64
        original["entries"][0]["ping_basis"] = "A different but valid basis"
        with self.assertRaisesRegex(LedgerError, "original v1 prefix changed"):
            validate(snapshot, ledger, V2_GUIDE, original)

    def test_v2_exclusion_after_original_prefix_has_no_label_fields(self) -> None:
        snapshot, ledger, original = v2_fixture()
        original["status"] = "in_progress"
        original["entries"] = original["entries"][:1]
        prefix_bytes = json.dumps(
            original["entries"], sort_keys=True, separators=(",", ":")
        ).encode()
        ledger["v1_prefix_sha256"] = hashlib.sha256(prefix_bytes).hexdigest()
        ledger["status"] = "in_progress"
        ledger["entries"] = ledger["entries"][:2]
        second = ledger["entries"][1]
        for field in (
            "v_label",
            "model_relation",
            "model_parts",
            "evidence",
            "fault_domain",
            "ping_detectable",
            "ping_basis",
            "closure_mode",
            "root_cause_status",
        ):
            del second[field]
        second["decision"] = "exclude"
        second["exclusion_code"] = "install_build"
        self.assertEqual(validate(snapshot, ledger, V2_GUIDE, original)["excluded"], 1)
        second["fault_domain"] = "leaf"
        with self.assertRaisesRegex(LedgerError, "excluded report has label fields"):
            validate(snapshot, ledger, V2_GUIDE, original)

    def test_v2_model_gap_needs_located_fault_domain(self) -> None:
        snapshot, ledger, original = v2_fixture()
        first = ledger["entries"][0]
        first["model_relation"] = "model_gap"
        first["model_gap"] = "Cross-process resource lifetime not represented"
        first["model_parts"] = []
        first["evidence"] = {
            "V": first["evidence"]["V"],
            "gap": {"pointer": "body:L3", "grade": "reporter_narrative"},
        }
        with self.assertRaisesRegex(LedgerError, "model_gap needs located"):
            validate(snapshot, ledger, V2_GUIDE, original)
        first["fault_domain"] = "leaf"
        self.assertEqual(validate(snapshot, ledger, V2_GUIDE, original)["eligible"], 40)


if __name__ == "__main__":
    unittest.main()
