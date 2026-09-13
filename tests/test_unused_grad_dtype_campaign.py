import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "pytorch-unused-grad-dtype"
sys.path.insert(0, str(EXPERIMENT))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class UnusedGradientDtypeCampaignTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.campaign = load_module(EXPERIMENT / "campaign.py", "dtype_campaign")
        cls.verifier = load_module(EXPERIMENT / "verify_results.py", "dtype_verifier")
        cls.probe_campaign = load_module(
            EXPERIMENT / "dtype_probe_campaign.py", "dtype_probe_campaign_test"
        )
        cls.probe_verifier = load_module(
            EXPERIMENT / "verify_dtype_probe.py", "dtype_probe_verifier_test"
        )
        cls.four_gpu_verifier = load_module(
            EXPERIMENT / "verify_four_gpu.py", "four_gpu_verifier_test"
        )
        cls.gate1b_campaign = load_module(
            EXPERIMENT / "gate1b_campaign.py", "gate1b_campaign_test"
        )
        cls.gate1b_verifier = load_module(
            EXPERIMENT / "verify_gate1b.py", "gate1b_verifier_test"
        )
        cls.gate1c_campaign = load_module(
            EXPERIMENT / "gate1c_campaign.py", "gate1c_campaign_test"
        )
        cls.gate1c_verifier = load_module(
            EXPERIMENT / "verify_gate1c.py", "gate1c_verifier_test"
        )
        cls.gate1d_campaign = load_module(
            EXPERIMENT / "gate1d_campaign.py", "gate1d_campaign_test"
        )
        cls.gate1d_verifier = load_module(
            EXPERIMENT / "verify_gate1d.py", "gate1d_verifier_test"
        )

    def test_classifier_requires_exact_assertion_marker(self) -> None:
        self.assertEqual(
            "mixed_gradient_dtype_assertion",
            self.campaign.classify_output(
                "affected",
                1,
                "FSDP reduce-scatter expects uniform gradient dtype but got x",
            ),
        )
        self.assertEqual(
            "unexpected_failure",
            self.campaign.classify_output("affected", 1, "some other error"),
        )

    def test_classifier_requires_success_marker(self) -> None:
        self.assertEqual(
            "completed",
            self.campaign.classify_output(
                "control",
                0,
                'DFX_RESULT={"arm": "control", "completed_steps": 2}',
            ),
        )
        self.assertEqual(
            "unverified_completion",
            self.campaign.classify_output("control", 0, "ordinary output"),
        )

    def test_classifier_rejects_wrong_arm_and_prioritizes_timeout(self) -> None:
        self.assertEqual(
            "unverified_completion",
            self.campaign.classify_output(
                "affected",
                0,
                'DFX_RESULT={"arm": "control", "completed_steps": 2}',
            ),
        )

    def test_accumulated_command_selects_reproducer_and_microbatch_flag(self) -> None:
        command = self.campaign.build_command("affected", 4, "accumulated")
        self.assertIn("accumulation_reproducer.py", command[-5])
        self.assertEqual(["--arm", "affected", "--microbatches", "4"], command[-4:])

    def test_dtype_probe_classifier_requires_observed_uniform_dtypes(self) -> None:
        output = self._probe_output(("torch.bfloat16",), ("torch.bfloat16",))
        self.assertEqual(
            "completed_uniform_bf16",
            self.probe_campaign.classify("unused-parameter", 0, output),
        )
        mixed = self._probe_output(
            ("torch.bfloat16", "torch.float32"),
            ("torch.bfloat16", "torch.float32"),
        )
        self.assertEqual(
            "completed_unexpected_dtype_set",
            self.probe_campaign.classify("unused-parameter", 0, mixed),
        )

    def test_dtype_probe_positive_control_requires_dtype_and_assertion(self) -> None:
        mixed = self._probe_output(
            ("torch.bfloat16", "torch.float32"),
            ("torch.bfloat16", "torch.float32"),
        )
        self.assertEqual(
            "mixed_gradient_dtype_assertion",
            self.probe_campaign.classify(
                "forced-mixed-gradient",
                1,
                mixed + self.probe_campaign.ASSERTION_MARKER,
            ),
        )
        self.assertEqual(
            "unexpected_failure",
            self.probe_campaign.classify(
                "forced-mixed-gradient", 1, self.probe_campaign.ASSERTION_MARKER
            ),
        )

    def test_dtype_probe_timeout_wins_over_assertion(self) -> None:
        mixed = self._probe_output(
            ("torch.bfloat16", "torch.float32"),
            ("torch.bfloat16", "torch.float32"),
        )
        self.assertEqual(
            "timeout",
            self.probe_campaign.classify(
                "forced-mixed-gradient",
                124,
                mixed + self.probe_campaign.ASSERTION_MARKER,
            ),
        )

    def test_dtype_probe_verifier_accepts_frozen_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_probe_matrix(root)
            self.probe_verifier.verify(root, trials=3)

    def test_dtype_probe_verifier_rejects_wrong_observed_dtype(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_probe_matrix(root)
            path = root / "unused-parameter-trial-1.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            record["dtype_records"][0]["grad_dtypes"] = ["torch.float32"]
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "dtype evidence"):
                self.probe_verifier.verify(root, trials=3)

    def test_four_gpu_verifier_accepts_frozen_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_four_gpu_matrix(root)
            self.four_gpu_verifier.verify(root, trials=3)

    def test_four_gpu_verifier_rejects_reused_port(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_four_gpu_matrix(root)
            first = root / "pp1-dp4-divergent-trial-1.json"
            second = root / "pp1-dp4-divergent-trial-2.json"
            first_record = json.loads(first.read_text(encoding="utf-8"))
            second_record = json.loads(second.read_text(encoding="utf-8"))
            second_record["rendezvous_port"] = first_record["rendezvous_port"]
            second.write_text(json.dumps(second_record), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "reused rendezvous port"):
                self.four_gpu_verifier.verify(root, trials=3)
        self.assertEqual(
            "timeout",
            self.campaign.classify_output(
                "affected", 124, self.campaign.ASSERTION_MARKER
            ),
        )

    def test_gate1b_classifier_accepts_control_matrix(self) -> None:
        reduce_records = self._gate1b_reduce_records(
            ("torch.float32",), ("torch.float32",)
        )
        reduce_returns = [{"rank": rank, "call_index": 1} for rank in (0, 1)]
        outcomes = [{"rank": rank, "outcome": "completed"} for rank in (0, 1)]
        self.assertEqual(
            "completed_symmetric_fp32",
            self.gate1b_campaign.classify(
                "control",
                timed_out=False,
                reduce_records=reduce_records,
                reduce_return_records=reduce_returns,
                outcomes=outcomes,
            ),
        )

    def test_gate1b_classifier_accepts_rank_local_assertion_then_timeout(
        self,
    ) -> None:
        reduce_records = self._gate1b_reduce_records(
            ("torch.float32",), ("torch.float32", "torch.bfloat16")
        )
        self.assertEqual(
            "rank1_assertion_rank0_wait",
            self.gate1b_campaign.classify(
                "affected",
                timed_out=True,
                reduce_records=reduce_records,
                reduce_return_records=[],
                outcomes=[{"rank": 1, "outcome": "local_uniformity_assertion"}],
            ),
        )

    def test_gate1b_classifier_rejects_timeout_without_rank_assertion(self) -> None:
        reduce_records = self._gate1b_reduce_records(
            ("torch.float32",), ("torch.float32", "torch.bfloat16")
        )
        self.assertEqual(
            "unexpected_pattern",
            self.gate1b_campaign.classify(
                "affected",
                timed_out=True,
                reduce_records=reduce_records,
                reduce_return_records=[],
                outcomes=[],
            ),
        )

    def test_gate1b_classifier_rejects_duplicate_reduce_record(self) -> None:
        reduce_records = self._gate1b_reduce_records(
            ("torch.float32",), ("torch.float32", "torch.bfloat16")
        )
        reduce_records.append(dict(reduce_records[0]))
        self.assertEqual(
            "unexpected_pattern",
            self.gate1b_campaign.classify(
                "affected",
                timed_out=True,
                reduce_records=reduce_records,
                reduce_return_records=[],
                outcomes=[{"rank": 1, "outcome": "local_uniformity_assertion"}],
            ),
        )

    def test_gate1b_marker_parser_orders_rank_then_call(self) -> None:
        output = "\n".join(
            [
                'DFX_RANK_REDUCE={"rank": 1, "call_index": 1, '
                '"grad_dtypes": ["torch.float32"]}',
                'DFX_RANK_REDUCE={"rank": 0, "call_index": 1, '
                '"grad_dtypes": ["torch.float32"]}',
            ]
        )
        records = self.gate1b_campaign.parse_records(output, "reduce")
        self.assertEqual([0, 1], [record["rank"] for record in records])

    def test_gate1b_verifier_accepts_frozen_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_gate1b_matrix(root)
            self.gate1b_verifier.verify(root, trials=3)

    def test_gate1b_verifier_rejects_capture_without_rank0_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_gate1b_matrix(root)
            path = root / "affected-trial-1.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            record["stack_capture"][0]["project_frames"] = []
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "no reproducer frame"):
                self.gate1b_verifier.verify(root, trials=3)

    def test_gate1b_verifier_rejects_timeout_without_local_assertion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_gate1b_matrix(root)
            path = root / "affected-trial-1.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            record["rank_outcomes"] = []
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "rank outcome matrix"):
                self.gate1b_verifier.verify(root, trials=3)

    def test_gate1c_classifier_accepts_return_then_barrier_wait(self) -> None:
        self.assertEqual(
            "rank1_assertion_rank0_barrier_wait",
            self.gate1c_campaign.classify_mechanism(
                "affected",
                reduce_records=self._gate1b_reduce_records(
                    ("torch.float32",), ("torch.float32", "torch.bfloat16")
                ),
                reduce_return_records=[{"rank": 0, "call_index": 1}],
                barrier_enter_records=[{"rank": 0}],
                barrier_return_records=[],
                outcomes=[{"rank": 1, "outcome": "local_uniformity_assertion"}],
            ),
        )

    def test_gate1c_classifier_rejects_missing_rank0_reduce_return(self) -> None:
        self.assertEqual(
            "unexpected_pattern",
            self.gate1c_campaign.classify_mechanism(
                "affected",
                reduce_records=self._gate1b_reduce_records(
                    ("torch.float32",), ("torch.float32", "torch.bfloat16")
                ),
                reduce_return_records=[],
                barrier_enter_records=[{"rank": 0}],
                barrier_return_records=[],
                outcomes=[{"rank": 1, "outcome": "local_uniformity_assertion"}],
            ),
        )

    def test_gate1c_termination_is_independent_of_mechanism(self) -> None:
        self.assertEqual(
            "wall_bound_rank1_teardown_wait",
            self.gate1c_campaign.classify_termination(
                "affected",
                timed_out=True,
                natural_return_code=None,
                teardown_enter_records=[{"rank": 1}],
                teardown_return_records=[],
                watchdog_marker_seen=True,
            ),
        )

    def test_gate1c_single_dump_is_not_collective_nonparticipation(self) -> None:
        summary = self.gate1c_campaign.strict_pending_reduce_summary(
            [self._gate1c_flight_artifact(0, "scheduled")]
        )
        self.assertEqual("incomplete_dump_set", summary["status"])
        self.assertEqual([1], summary["missing_dump_ranks"])

    def test_gate1c_starts_rank_stack_captures_concurrently(self) -> None:
        rendezvous = threading.Barrier(2, timeout=1)
        original_live = self.gate1c_campaign.identity_is_live
        original_capture = self.gate1c_campaign.capture_stack

        def fake_capture(_py_spy, rank, _pid, _target):
            rendezvous.wait()
            return {"rank": rank, "status": "captured", "raw_persisted": False}

        self.gate1c_campaign.identity_is_live = lambda _identity: True
        self.gate1c_campaign.capture_stack = fake_capture
        try:
            records = self.gate1c_campaign.capture_rank_stacks(
                "py-spy",
                [SimpleNamespace(pid=100), SimpleNamespace(pid=101)],
                Path("reproducer.py"),
            )
        finally:
            self.gate1c_campaign.identity_is_live = original_live
            self.gate1c_campaign.capture_stack = original_capture
        self.assertEqual([0, 1], [item["rank"] for item in records])

    def test_gate1c_strict_join_requires_pending_rank0_reduce(self) -> None:
        matched = self.gate1c_campaign.strict_pending_reduce_summary(
            [
                self._gate1c_flight_artifact(0, "scheduled"),
                self._gate1c_flight_artifact(1, None),
            ]
        )
        self.assertEqual("matched", matched["status"])
        completed = self.gate1c_campaign.strict_pending_reduce_summary(
            [
                self._gate1c_flight_artifact(0, "completed"),
                self._gate1c_flight_artifact(1, None),
            ]
        )
        self.assertEqual("no_pending_single_rank_reduce", completed["status"])

    def test_gate1c_verifier_accepts_frozen_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_gate1c_matrix(root)
            self.gate1c_verifier.verify(root, trials=3)

    def test_gate1c_verifier_rejects_missing_rank_dump(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_gate1c_matrix(root)
            path = root / "affected-trial-1.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            record["flight_recorder"]["files"] = [record["flight_recorder"]["files"][0]]
            record["flight_recorder"]["file_count"] = 1
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "both rank dumps"):
                self.gate1c_verifier.verify(root, trials=3)

    def test_gate1d_verifier_accepts_frozen_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_gate1d_matrix(root)
            self.gate1d_verifier.verify(root, trials=3)

    def test_gate1d_verifier_accepts_termination_prediction_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_gate1d_matrix(root)
            path = root / "affected-trial-1.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            record["termination_classification"] = "torchrun_teardown"
            record["termination_prediction_matched"] = False
            record["timed_out"] = False
            record["natural_return_code"] = 1
            record["reported_return_code"] = 1
            record["watchdog_marker_seen"] = False
            path.write_text(json.dumps(record), encoding="utf-8")
            self.gate1d_verifier.verify(root, trials=3)

    def test_gate1d_verifier_rejects_reproducer_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_gate1d_matrix(root)
            path = root / "affected-trial-1.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            record["reproducer_sha256"] = "0" * 64
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "current source"):
                self.gate1d_verifier.verify(root, trials=3)

    def test_verifier_accepts_complete_reproduced_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_matrix(root, affected="mixed_gradient_dtype_assertion")
            self.verifier.verify(root, trials=3, expected_affected="reproduced")

    def test_verifier_rejects_wrong_affected_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_matrix(root, affected="completed")
            with self.assertRaisesRegex(AssertionError, "predeclared expectation"):
                self.verifier.verify(root, trials=3, expected_affected="reproduced")

    def test_verifier_rejects_lifecycle_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_matrix(root, affected="mixed_gradient_dtype_assertion")
            path = root / "control-trial-1.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            record["no_tracked_orphans"] = False
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "lifecycle contract"):
                self.verifier.verify(root, trials=3, expected_affected="reproduced")

    def test_verifier_rejects_identity_that_disagrees_with_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_matrix(root, affected="mixed_gradient_dtype_assertion")
            path = root / "affected-trial-1.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            record["trial"] = 2
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "does not match filename"):
                self.verifier.verify(root, trials=3, expected_affected="reproduced")

    @staticmethod
    def _write_matrix(root: Path, affected: str) -> None:
        for arm in ("control", "affected"):
            for trial in range(1, 4):
                record = {
                    "arm": arm,
                    "trial": trial,
                    "torch_version": "2.13.0+cu130",
                    "reproducer_sha256": "a" * 64,
                    "cuda_visible_devices": 2,
                    "classification": "completed" if arm == "control" else affected,
                    "raw_output_persisted": False,
                    "rank_identities_recorded": True,
                    "no_tracked_orphans": True,
                }
                (root / f"{arm}-trial-{trial}.json").write_text(
                    json.dumps(record), encoding="utf-8"
                )

    @staticmethod
    def _probe_output(rank0: tuple[str, ...], rank1: tuple[str, ...]) -> str:
        lines = []
        for rank, dtypes in enumerate((rank0, rank1)):
            payload = {
                "rank": rank,
                "grad_count": len(dtypes),
                "grad_dtypes": list(dtypes),
            }
            lines.append("DFX_REDUCE_DTYPES=" + json.dumps(payload))
        return "\n".join(lines) + "\n"

    @staticmethod
    def _gate1b_reduce_records(
        rank0: tuple[str, ...], rank1: tuple[str, ...]
    ) -> list[dict]:
        return [
            {
                "rank": rank,
                "call_index": 1,
                "grad_count": len(dtypes),
                "grad_dtypes": list(dtypes),
            }
            for rank, dtypes in enumerate((rank0, rank1))
        ]

    @classmethod
    def _write_gate1b_matrix(cls, root: Path) -> None:
        for arm in ("control", "affected"):
            for trial in range(1, 4):
                affected = arm == "affected"
                dtypes = (
                    (("torch.float32",), ("torch.float32", "torch.bfloat16"))
                    if affected
                    else (("torch.float32",), ("torch.float32",))
                )
                record = {
                    "schema_version": 1,
                    "arm": arm,
                    "trial": trial,
                    "expected_classification": (
                        "rank1_assertion_rank0_wait"
                        if affected
                        else "completed_symmetric_fp32"
                    ),
                    "classification": (
                        "rank1_assertion_rank0_wait"
                        if affected
                        else "completed_symmetric_fp32"
                    ),
                    "timed_out": affected,
                    "return_code": 124 if affected else 0,
                    "reduce_records": cls._gate1b_reduce_records(*dtypes),
                    "reduce_return_records": (
                        []
                        if affected
                        else [{"rank": rank, "call_index": 1} for rank in (0, 1)]
                    ),
                    "rank_outcomes": (
                        [{"rank": 1, "outcome": "local_uniformity_assertion"}]
                        if affected
                        else [{"rank": rank, "outcome": "completed"} for rank in (0, 1)]
                    ),
                    "watchdog_marker_seen": affected,
                    "stack_capture": (
                        [
                            {
                                "rank": 0,
                                "status": "captured",
                                "raw_persisted": False,
                                "project_frames": [
                                    {"function": "observed", "line": 42}
                                ],
                            }
                        ]
                        if affected
                        else []
                    ),
                    "flight_recorder": {
                        "file_count": 1 if affected else 0,
                        "files": ([{"rank": 0, "decoded": True}] if affected else []),
                        "normalized": (
                            {"secondary_divergences": [{"reason": "missing_member"}]}
                            if affected
                            else None
                        ),
                        "raw_persisted": False,
                    },
                    "output_sha256": "c" * 64,
                    "raw_output_persisted": False,
                    "runner_error_type": None,
                    "reproducer_sha256": "d" * 64,
                    "torch_version": "2.13.0+cu130",
                    "torch_git_version": "e" * 40,
                    "cuda_visible_devices": 2,
                    "gpu_names": ["NVIDIA RTX 4090"] * 2,
                    "rank_identities_recorded": True,
                    "no_tracked_orphans": True,
                }
                (root / f"{arm}-trial-{trial}.json").write_text(
                    json.dumps(record), encoding="utf-8"
                )

    @staticmethod
    def _gate1c_flight_artifact(rank: int, state: str | None) -> dict:
        entries = []
        if state is not None:
            entries.append(
                {
                    "profiling_name": "nccl:_reduce_scatter_base",
                    "pg_id": "0",
                    "collective_seq_id": 8,
                    "record_id": 11,
                    "state": state,
                    "is_p2p": False,
                }
            )
        return {"rank": rank, "pg_config": {"0": {"ranks": [0, 1]}}, "entries": entries}

    @classmethod
    def _write_gate1c_matrix(cls, root: Path) -> None:
        barrier_line = cls.gate1c_verifier._call_line("barrier")
        teardown_line = cls.gate1c_verifier._call_line("destroy_process_group")
        preflight = {
            "effective_uid": 1000,
            "ptrace_scope": 1,
            "observer_authorization": "PR_SET_PTRACER campaign parent",
            "py_spy_version_sha256": "f" * 64,
        }
        for arm in ("control", "affected"):
            for trial in range(1, 4):
                affected = arm == "affected"
                record = {
                    "schema_version": 2,
                    "arm": arm,
                    "trial": trial,
                    "expected_mechanism": (
                        "rank1_assertion_rank0_barrier_wait"
                        if affected
                        else "completed_symmetric_fp32"
                    ),
                    "mechanism_classification": (
                        "rank1_assertion_rank0_barrier_wait"
                        if affected
                        else "completed_symmetric_fp32"
                    ),
                    "expected_termination": (
                        "wall_bound_rank1_teardown_wait"
                        if affected
                        else "normal_completion"
                    ),
                    "termination_classification": (
                        "wall_bound_rank1_teardown_wait"
                        if affected
                        else "normal_completion"
                    ),
                    "timed_out": affected,
                    "natural_return_code": None if affected else 0,
                    "reported_return_code": 124 if affected else 0,
                    "reduce_records": cls._gate1b_reduce_records(
                        ("torch.float32",),
                        (
                            ("torch.float32", "torch.bfloat16")
                            if affected
                            else ("torch.float32",)
                        ),
                    ),
                    "reduce_return_records": (
                        [{"rank": 0, "call_index": 1}]
                        if affected
                        else [{"rank": rank, "call_index": 1} for rank in (0, 1)]
                    ),
                    "barrier_enter_records": (
                        [{"rank": 0}]
                        if affected
                        else [{"rank": rank} for rank in (0, 1)]
                    ),
                    "barrier_return_records": (
                        [] if affected else [{"rank": rank} for rank in (0, 1)]
                    ),
                    "teardown_enter_records": (
                        [{"rank": 1}]
                        if affected
                        else [{"rank": rank} for rank in (0, 1)]
                    ),
                    "teardown_return_records": (
                        [] if affected else [{"rank": rank} for rank in (0, 1)]
                    ),
                    "rank_outcomes": (
                        [{"rank": 1, "outcome": "local_uniformity_assertion"}]
                        if affected
                        else [{"rank": rank, "outcome": "completed"} for rank in (0, 1)]
                    ),
                    "watchdog_marker_seen": affected,
                    "stack_capture": (
                        [
                            {
                                "rank": 0,
                                "status": "captured",
                                "raw_persisted": False,
                                "project_frames": [
                                    {"function": "main", "line": barrier_line}
                                ],
                            },
                            {
                                "rank": 1,
                                "status": "captured",
                                "raw_persisted": False,
                                "project_frames": [
                                    {"function": "main", "line": teardown_line}
                                ],
                            },
                        ]
                        if affected
                        else []
                    ),
                    "flight_recorder": {
                        "file_count": 2 if affected else 0,
                        "files": (
                            [{"rank": rank, "decoded": True} for rank in (0, 1)]
                            if affected
                            else []
                        ),
                        "normalized": {} if affected else None,
                        "strict_pending_reduce": (
                            {
                                "status": "matched",
                                "dump_ranks": [0, 1],
                                "missing_dump_ranks": [],
                                "candidates": [
                                    {
                                        "sequence_number": 8,
                                        "group_members": [0, 1],
                                        "present_ranks": [0],
                                        "operation": "_REDUCE_SCATTER_BASE",
                                        "state_by_rank": [
                                            {"rank": 0, "state": "scheduled"}
                                        ],
                                    }
                                ],
                            }
                            if affected
                            else {
                                "status": "incomplete_dump_set",
                                "dump_ranks": [],
                                "missing_dump_ranks": [0, 1],
                                "candidates": [],
                            }
                        ),
                        "raw_persisted": False,
                    },
                    "preflight": preflight,
                    "output_sha256": "c" * 64,
                    "raw_output_persisted": False,
                    "runner_error_type": None,
                    "reproducer_sha256": "d" * 64,
                    "torch_version": "2.13.0+cu130",
                    "torch_git_version": "e" * 40,
                    "cuda_visible_devices": 2,
                    "gpu_names": ["NVIDIA RTX 4090"] * 2,
                    "rank_identities_recorded": True,
                    "no_tracked_orphans": True,
                }
                (root / f"{arm}-trial-{trial}.json").write_text(
                    json.dumps(record), encoding="utf-8"
                )

    @classmethod
    def _write_gate1d_matrix(cls, root: Path) -> None:
        cls._write_gate1c_matrix(root)
        source_hash = cls.gate1d_verifier.hashlib.sha256(
            cls.gate1d_verifier.REPRODUCER.read_bytes()
        ).hexdigest()
        cls.gate1c_verifier.REPRODUCER = cls.gate1d_verifier.REPRODUCER
        barrier_line = cls.gate1c_verifier._call_line("barrier")
        teardown_line = cls.gate1c_verifier._call_line("destroy_process_group")
        preflight = {
            "effective_uid": 1000,
            "ptrace_scope": 1,
            "ptrace_mode": "pr_set_ptracer_parent",
            "py_spy_version_sha256": "f" * 64,
        }
        for path in root.glob("*.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            record["schema_version"] = 3
            record["configuration"] = {
                "stack_capture_after_seconds": 20,
                "process_group_timeout_seconds": 30,
                "wall_timeout_seconds": 60,
                "async_error_handling": 3,
            }
            record["termination_prediction_matched"] = True
            record["rank_ptrace_records"] = [
                {"rank": rank, "mode": "pr_set_ptracer_parent"} for rank in (0, 1)
            ]
            record["preflight"] = preflight
            record["reproducer_sha256"] = source_hash
            if record["arm"] == "affected":
                record["stack_capture"][0]["project_frames"][0]["line"] = barrier_line
                record["stack_capture"][1]["project_frames"][0]["line"] = teardown_line
            path.write_text(json.dumps(record), encoding="utf-8")

    @classmethod
    def _write_probe_matrix(cls, root: Path) -> None:
        contracts = {
            "unused-parameter": {
                "classification": "completed_uniform_bf16",
                "dtypes": ["torch.bfloat16"],
                "assertion": False,
            },
            "forced-mixed-gradient": {
                "classification": "mixed_gradient_dtype_assertion",
                "dtypes": ["torch.bfloat16", "torch.float32"],
                "assertion": True,
            },
        }
        for case, contract in contracts.items():
            for trial in range(1, 4):
                record = {
                    "case": case,
                    "trial": trial,
                    "expected_classification": contract["classification"],
                    "classification": contract["classification"],
                    "dtype_records": [
                        {
                            "rank": rank,
                            "grad_count": len(contract["dtypes"]),
                            "grad_dtypes": contract["dtypes"],
                        }
                        for rank in (0, 1)
                    ],
                    "assertion_marker_seen": contract["assertion"],
                    "raw_output_persisted": False,
                    "rank_identities_recorded": True,
                    "no_tracked_orphans": True,
                    "reproducer_sha256": "b" * 64,
                    "torch_version": "2.13.0+cu130",
                    "return_code": 1 if contract["assertion"] else 0,
                    "cuda_visible_devices": 2,
                    "gpu_names": ["NVIDIA RTX 4090", "NVIDIA RTX 4090"],
                }
                (root / f"{case}-trial-{trial}.json").write_text(
                    json.dumps(record), encoding="utf-8"
                )

    @classmethod
    def _write_four_gpu_matrix(cls, root: Path) -> None:
        matrix = cls.four_gpu_verifier.MATRIX
        ordinal = 0
        for case, contract in matrix.items():
            for trial in range(1, 4):
                assertion = contract["expected"] == "mixed_gradient_dtype_assertion"
                record = {
                    "case": case,
                    "trial": trial,
                    "prepared_sha256": contract["sha256"],
                    "expected_classification": contract["expected"],
                    "classification": contract["expected"],
                    "return_code": 1 if assertion else 0,
                    "assertion_marker_seen": assertion,
                    "cuda_visible_devices": 4,
                    "gpu_names": ["NVIDIA RTX 4090"] * 4,
                    "raw_output_persisted": False,
                    "rank_identities_recorded": True,
                    "no_tracked_orphans": True,
                    "detail_enabled": False,
                    "training_steps": 200,
                    "torch_version": "2.13.0+cu130",
                    "rendezvous_port": 29600 + ordinal,
                }
                ordinal += 1
                (root / f"{case}-trial-{trial}.json").write_text(
                    json.dumps(record), encoding="utf-8"
                )


if __name__ == "__main__":
    unittest.main()
