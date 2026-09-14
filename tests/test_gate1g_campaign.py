import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "pytorch-unused-grad-dtype"
sys.path.insert(0, str(EXPERIMENT))

import gate1f_campaign as gate1f  # noqa: E402
import gate1g_campaign as gate1g  # noqa: E402
import verify_gate1g  # noqa: E402


class Gate1gCampaignTest(unittest.TestCase):
    def test_parser_handles_adjacent_rank_writes(self) -> None:
        prefix = gate1g.MARKER_PREFIXES["outcome"]
        output = (
            prefix
            + json.dumps({"rank": 0, "outcome": "completed"})
            + prefix
            + json.dumps({"rank": 1, "outcome": "completed"})
        )
        self.assertEqual(
            [0, 1],
            [item["rank"] for item in gate1g.parse_records(output, "outcome")],
        )

    def test_parser_failure_is_retained_as_bounded_code(self) -> None:
        output = gate1g.MARKER_PREFIXES["outcome"] + "{broken"
        records, errors = gate1g.parse_all_records_safe(output)
        self.assertEqual([], records["outcome"])
        self.assertEqual("ValueError", errors["outcome"])

    def test_control_mechanism_requires_both_returns(self) -> None:
        records = {kind: [] for kind in gate1g.MARKER_PREFIXES}
        records["warmup_enter"] = [{"rank": 0}, {"rank": 1}]
        records["warmup_return"] = [{"rank": 0}, {"rank": 1}]
        records["allreduce_enqueued"] = [{"rank": 0}, {"rank": 1}]
        records["allreduce_return"] = [{"rank": 0}, {"rank": 1}]
        records["outcome"] = [
            {"rank": 0, "outcome": "completed"},
            {"rank": 1, "outcome": "completed"},
        ]
        self.assertEqual(
            "completed_symmetric_all_reduce",
            gate1g.classify_mechanism("control", records),
        )
        records["allreduce_return"].pop()
        self.assertEqual(
            "unexpected_pattern", gate1g.classify_mechanism("control", records)
        )

    def test_affected_mechanism_uses_atomic_ready_observation(self) -> None:
        records = {kind: [] for kind in gate1g.MARKER_PREFIXES}
        records["warmup_enter"] = [{"rank": 0}, {"rank": 1}]
        records["warmup_return"] = [{"rank": 0}, {"rank": 1}]
        records["allreduce_enqueued"] = [{"rank": 0}]
        records["ready_observed"] = [{"rank": 1}]
        records["outcome"] = [{"rank": 1, "outcome": "injected_rank_failure"}]
        self.assertEqual(
            "rank1_failure_rank0_all_reduce_wait",
            gate1g.classify_mechanism("affected", records),
        )

    def test_rank0_summary_is_local_and_requires_pending_allreduce(self) -> None:
        artifacts = [
            {
                "rank": 0,
                "pg_config": {"0": {"ranks": [0, 1]}},
                "entries": [
                    {
                        "profiling_name": "nccl:all_reduce",
                        "pg_id": "0",
                        "collective_seq_id": 4,
                        "state": "scheduled",
                    }
                ],
            }
        ]
        summary = gate1g.rank0_pending_allreduce_summary(artifacts)
        self.assertEqual("matched", summary["status"])
        self.assertEqual([0], summary["dump_ranks"])
        self.assertFalse(summary["peer_participation_inferred"])
        self.assertEqual([0, 1], summary["candidates"][0]["group_members"])

        rank0_candidates = list(summary["candidates"])
        artifacts.append(
            {
                "rank": 1,
                "pg_config": {"0": {"ranks": [0, 1]}},
                "entries": [
                    {
                        "profiling_name": "nccl:all_reduce",
                        "pg_id": "0",
                        "collective_seq_id": 99,
                        "state": "scheduled",
                    }
                ],
            }
        )
        summary = gate1g.rank0_pending_allreduce_summary(artifacts)
        self.assertEqual(rank0_candidates, summary["candidates"])

        artifacts[0]["entries"].append(
            {
                "profiling_name": "nccl:all_reduce",
                "pg_id": "0",
                "collective_seq_id": 5,
                "state": "scheduled",
            }
        )
        summary = gate1g.rank0_pending_allreduce_summary(artifacts)
        self.assertEqual("unexpected_pending_all_reduce_count", summary["status"])

    def test_verifier_accepts_exact_synthetic_contract(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_result(root, "control")
            self._write_result(root, "affected")
            verify_gate1g.verify(root)

    def test_verifier_rejects_peer_participation_inference(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_result(root, "control")
            self._write_result(root, "affected")
            path = root / "affected-trial-1.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            record["flight_recorder"]["rank0_pending_all_reduce"][
                "peer_participation_inferred"
            ] = True
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "rank-0-local"):
                verify_gate1g.verify(root)

    @staticmethod
    def _write_result(root: Path, arm: str) -> None:
        affected = arm == "affected"
        control_flags = {
            name: name
            in {
                "shutdown_start",
                "operations_flushed",
                "watchdog_joined_destroying",
                "destroy_complete",
            }
            for name in gate1f.LOG_MESSAGES
        }
        wait_line = verify_gate1g._call_line("wait")
        destroy_line = verify_gate1g._call_line("destroy_process_group")
        record = {
            "schema_version": 5,
            "arm": arm,
            "trial": 1,
            "configuration": {
                "stack_capture_after_seconds": 20,
                "process_group_timeout_seconds": 30,
                "wall_timeout_seconds": 60,
                "work_wait_timeout_seconds": 180,
                "async_error_handling": 3,
            },
            "expected_mechanism": gate1g.EXPECTED_MECHANISM[arm],
            "mechanism_classification": gate1g.EXPECTED_MECHANISM[arm],
            "expected_termination": gate1g.EXPECTED_TERMINATION[arm],
            "termination_classification": gate1g.EXPECTED_TERMINATION[arm],
            "termination_prediction_matched": True,
            "timed_out": affected,
            "natural_return_code": None if affected else 0,
            "reported_return_code": 124 if affected else 0,
            "warmup_enter_records": [{"rank": 0}, {"rank": 1}],
            "warmup_return_records": [{"rank": 0}, {"rank": 1}],
            "allreduce_enqueued_records": (
                [{"rank": 0}] if affected else [{"rank": 0}, {"rank": 1}]
            ),
            "allreduce_return_records": (
                [] if affected else [{"rank": 0}, {"rank": 1}]
            ),
            "ready_observed_records": [{"rank": 1}] if affected else [],
            "rank_outcomes": (
                [{"rank": 1, "outcome": "injected_rank_failure"}]
                if affected
                else [
                    {"rank": 0, "outcome": "completed"},
                    {"rank": 1, "outcome": "completed"},
                ]
            ),
            "teardown_enter_records": (
                [{"rank": 1}] if affected else [{"rank": 0}, {"rank": 1}]
            ),
            "teardown_return_records": ([] if affected else [{"rank": 0}, {"rank": 1}]),
            "rank_ptrace_records": [
                {"rank": rank, "mode": "pr_set_ptracer_parent"} for rank in (0, 1)
            ],
            "marker_parse_errors": {},
            "watchdog_marker_seen": affected,
            "stack_capture": (
                [
                    {
                        "rank": 0,
                        "status": "captured",
                        "raw_persisted": False,
                        "project_frames": [
                            {"function": "run_all_reduce", "line": wait_line}
                        ],
                    },
                    {
                        "rank": 1,
                        "status": "captured",
                        "raw_persisted": False,
                        "project_frames": [{"function": "main", "line": destroy_line}],
                    },
                ]
                if affected
                else []
            ),
            "flight_recorder": (
                {
                    "file_count": 1,
                    "files": [{"rank": 0, "decoded": True}],
                    "rank0_pending_all_reduce": {
                        "status": "matched",
                        "dump_ranks": [0],
                        "peer_participation_inferred": False,
                        "candidates": [
                            {
                                "sequence_number": 4,
                                "group_members": [0, 1],
                                "observed_rank": 0,
                                "operation": "ALLREDUCE",
                                "state": "scheduled",
                            }
                        ],
                    },
                    "raw_persisted": False,
                }
                if affected
                else {
                    "file_count": 0,
                    "files": [],
                    "rank0_pending_all_reduce": {
                        "status": "no_rank0_pending_all_reduce",
                        "dump_ranks": [],
                        "peer_participation_inferred": False,
                        "candidates": [],
                    },
                    "raw_persisted": False,
                }
            ),
            "library_log_flags": (
                {
                    "0": dict(gate1f.EXPECTED_RANK0_FLAGS),
                    "1": dict(gate1f.EXPECTED_RANK1_FLAGS),
                }
                if affected
                else {"0": control_flags, "1": control_flags}
            ),
            "library_log_scan_error": None,
            "preflight": {
                "effective_uid": 1000,
                "ptrace_scope": 1,
                "ptrace_mode": "pr_set_ptracer_parent",
                "py_spy_version_sha256": "a" * 64,
                "nccl_version": [2, 29, 7],
            },
            "output_sha256": "b" * 64,
            "raw_output_persisted": False,
            "runner_error_type": None,
            "reproducer_sha256": hashlib.sha256(
                verify_gate1g.REPRODUCER.read_bytes()
            ).hexdigest(),
            "torch_version": "2.13.0+cu130",
            "torch_git_version": "c" * 40,
            "cuda_runtime_version": "13.0",
            "cuda_visible_devices": 2,
            "gpu_names": ["NVIDIA RTX 4090", "NVIDIA RTX 4090"],
            "rank_identities_recorded": True,
            "no_tracked_orphans": True,
        }
        (root / f"{arm}-trial-1.json").write_text(json.dumps(record), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
