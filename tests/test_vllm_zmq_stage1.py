from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import queue
import sys
import tempfile
import threading
import unittest
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "vllm-zmq-event-backpressure"
PLUGIN = (
    EXPERIMENT / "stage1_plugin" / "src" / "dfx_stage1_backpressure" / "__init__.py"
)


def load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Stage1ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_file(
            "stage1_contract_test", EXPERIMENT / "stage1_contract.py"
        )

    def trace(self, offsets: list[float], completed: float | None = 3.0):
        trace = self.contract.ProgressTrace(started_at=100.0)
        for offset in offsets:
            trace.observe(100.0 + offset)
        if completed is not None:
            trace.complete(100.0 + completed)
        return trace

    def test_progress_trace_never_retains_content(self) -> None:
        trace = self.trace([0.2, 0.4, 0.7])
        self.assertEqual(
            set(trace.public_record()),
            {"completed_offset_seconds", "progress_count", "progress_offsets_seconds"},
        )
        self.assertAlmostEqual(trace.maximum_gap(), 0.3)
        self.assertAlmostEqual(trace.maximum_no_progress_gap(), 2.3)

    def test_stall_requires_open_request_and_ten_seconds_without_progress(self) -> None:
        trace = self.trace([0.5], completed=None)
        self.assertFalse(trace.stalled(120.0))
        trace.observe(101.0)
        self.assertFalse(trace.stalled(110.9))
        self.assertTrue(trace.stalled(111.0))
        trace.complete(112.0)
        self.assertFalse(trace.stalled(120.0))

    def test_control_requires_completion_gap_and_zero_drops(self) -> None:
        trace = self.trace([0.1, 0.5, 0.9])
        classify = self.contract.classify_cell
        self.assertEqual(
            classify(
                source_arm="base",
                trigger="control",
                trace=trace,
                stalled_before_release=False,
                stack_match=None,
                recovered_after_release=None,
                accepted_batch_count=1,
                dropped_batch_count=0,
            ),
            "pass",
        )
        self.assertEqual(
            classify(
                source_arm="base",
                trigger="control",
                trace=trace,
                stalled_before_release=False,
                stack_match=None,
                recovered_after_release=None,
                accepted_batch_count=1,
                dropped_batch_count=1,
            ),
            "control_failed",
        )
        self.assertEqual(
            classify(
                source_arm="base",
                trigger="control",
                trace=trace,
                stalled_before_release=False,
                stack_match=None,
                recovered_after_release=None,
                accepted_batch_count=0,
                dropped_batch_count=0,
            ),
            "control_failed",
        )

    def test_base_pause_requires_stack_and_recovery(self) -> None:
        trace = self.trace([0.2, 12.0], completed=13.0)
        classify = self.contract.classify_cell
        common = {
            "source_arm": "base",
            "trigger": "pause",
            "trace": trace,
            "stalled_before_release": True,
            "recovered_after_release": True,
            "accepted_batch_count": 1,
            "dropped_batch_count": 0,
        }
        self.assertEqual(classify(stack_match=True, **common), "pass")
        self.assertEqual(classify(stack_match=False, **common), "mechanism_mismatch")
        common["dropped_batch_count"] = 1
        self.assertEqual(classify(stack_match=True, **common), "mechanism_mismatch")

    def test_fix_pause_requires_progress_and_measured_drop(self) -> None:
        trace = self.trace([0.1, 0.3, 0.6])
        classify = self.contract.classify_cell
        common = {
            "source_arm": "fix",
            "trigger": "pause",
            "trace": trace,
            "stalled_before_release": False,
            "stack_match": None,
            "recovered_after_release": None,
        }
        self.assertEqual(
            classify(accepted_batch_count=1, dropped_batch_count=1, **common),
            "pass",
        )
        self.assertEqual(
            classify(accepted_batch_count=1, dropped_batch_count=0, **common),
            "mechanism_mismatch",
        )

        slow = self.trace([0.1, 11.0], completed=11.1)
        common["trace"] = slow
        self.assertEqual(
            classify(accepted_batch_count=1, dropped_batch_count=1, **common),
            "mechanism_mismatch",
        )

    def test_stack_match_is_ordered_and_fail_closed(self) -> None:
        valid = "\n".join(
            [
                "wait (threading.py:1)",
                "put (queue.py:2)",
                "publish (vllm/distributed/kv_events.py:3)",
            ]
        )
        self.assertTrue(self.contract.stack_matches(valid))
        self.assertFalse(
            self.contract.stack_matches("\n".join(reversed(valid.splitlines())))
        )

    def test_ready_and_queue_count_files_use_closed_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ready = root / "ready.json"
            ready.write_text(
                json.dumps(
                    {
                        "authorization": "yama_absent",
                        "kv_events_sha256": "b" * 64,
                        "mapped_worktree_binaries": {},
                        "pid": 42,
                        "plugin_sha256": "a" * 64,
                        "plugin_version": 1,
                        "state": "paused",
                        "start_time_ticks": 100,
                        "yama_ptrace_scope": None,
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(self.contract.load_ready(ready)["pid"], 42)
            count = root / "queue-counts.json"
            count.write_text(
                '{"accepted_batch_count":1,"dropped_batch_count":2}',
                encoding="utf-8",
            )
            self.assertEqual(
                self.contract.load_queue_counts(count),
                {"accepted_batch_count": 1, "dropped_batch_count": 2},
            )


class Stage1CampaignTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(EXPERIMENT))
        try:
            cls.campaign = load_file(
                "stage1_campaign_test", EXPERIMENT / "stage1_campaign.py"
            )
        finally:
            sys.path.remove(str(EXPERIMENT))

    def test_stream_progress_ignores_usage_only_and_empty_chunks(self) -> None:
        detect = self.campaign._has_stream_progress
        self.assertFalse(detect({"choices": [], "usage": {"completion_tokens": 4}}))
        self.assertFalse(detect({"choices": [{"delta": {"content": ""}}]}))
        self.assertTrue(detect({"choices": [{"delta": {"content": "x"}}]}))
        self.assertTrue(detect({"choices": [{"text": "x"}]}))

    def test_command_file_is_closed_json_array(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "command.json"
            path.write_text('["python","-m","vllm.entrypoints"]', encoding="utf-8")
            self.assertEqual(
                self.campaign.read_command(path),
                ["python", "-m", "vllm.entrypoints"],
            )
            path.write_text('"python -m vllm.entrypoints"', encoding="utf-8")
            with self.assertRaises(ValueError):
                self.campaign.read_command(path)

    def test_server_command_requires_frozen_trigger_settings(self) -> None:
        config = json.dumps(
            {
                "enable_kv_cache_events": True,
                "endpoint": "tcp://127.0.0.1:5557",
                "publisher": "zmq",
                "max_queue_size": 1,
            }
        )
        command = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--enable-prefix-caching",
            "--block-size",
            "16",
            "--async-scheduling",
            "--kv-events-config",
            config,
        ]
        self.campaign.validate_server_command(command)
        with self.assertRaisesRegex(ValueError, "prefix-caching"):
            self.campaign.validate_server_command(
                [item for item in command if item != "--enable-prefix-caching"]
            )
        with self.assertRaisesRegex(ValueError, "negated"):
            self.campaign.validate_server_command(
                [*command, "--no-enable-prefix-caching"]
            )
        with self.assertRaisesRegex(ValueError, "block-size"):
            self.campaign.validate_server_command([*command, "--block-size", "16"])
        with self.assertRaisesRegex(ValueError, "kv-events-config"):
            self.campaign.validate_server_command(
                [*command, "--kv-events-config", config]
            )
        wildcard = json.loads(config)
        wildcard["endpoint"] = "tcp://*:5557"
        wildcard_command = command[:-1] + [json.dumps(wildcard)]
        with self.assertRaisesRegex(ValueError, "loopback"):
            self.campaign.validate_server_command(wildcard_command)

    def test_request_requires_streaming_usage_and_two_output_blocks(self) -> None:
        request = {
            "prompt": "private",
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0,
            "max_tokens": 32,
        }
        self.campaign.validate_request(json.dumps(request).encode())
        request["max_tokens"] = 31
        with self.assertRaisesRegex(ValueError, "32 output tokens"):
            self.campaign.validate_request(json.dumps(request).encode())

    def test_observer_public_trace_excludes_stream_content(self) -> None:
        observer = self.campaign.StreamObserver()
        observer.observe()
        observer.complete()
        self.assertEqual(
            set(observer.public_trace()),
            {"completed_offset_seconds", "progress_count", "progress_offsets_seconds"},
        )

    def test_recovery_requires_new_progress_and_completion(self) -> None:
        observer = self.campaign.StreamObserver()
        observer.observe()
        release_offset, count = observer.release_snapshot()
        self.assertGreaterEqual(release_offset, 0)
        observer.complete()
        self.assertFalse(observer.recovered_after(count))

        observer = self.campaign.StreamObserver()
        observer.observe()
        _, count = observer.release_snapshot()
        observer.observe()
        observer.complete()
        self.assertTrue(observer.recovered_after(count))

    def test_cleanup_signals_group_even_when_leader_already_exited(self) -> None:
        process = Mock(pid=123)
        process.wait.return_value = 0
        with (
            patch.object(self.campaign.os, "killpg", create=True) as killpg,
            patch.object(
                self.campaign, "wait_for_process_group_exit", return_value=True
            ),
            patch.object(self.campaign, "identity_is_live", return_value=False),
        ):
            result = self.campaign.stop_process_group(process, (456, 789))
        killpg.assert_called_once_with(123, self.campaign.signal.SIGTERM)
        self.assertTrue(result["process_group_gone"])
        self.assertTrue(result["engine_core_gone"])

    def test_classification_priority_preserves_primary_failure(self) -> None:
        prioritize = self.campaign.prioritize_classification
        common = {
            "hook_integrity_failed": False,
            "release_observed": False,
            "request_sized": False,
        }
        self.assertEqual(
            prioritize("pass", stream_error_kind="transport_error", **common),
            "stream_failed",
        )
        self.assertEqual(
            prioritize("pass", stream_error_kind=None, **common),
            "hook_release_missing",
        )
        common["release_observed"] = True
        self.assertEqual(
            prioritize("pass", stream_error_kind=None, **common),
            "trigger_not_reached",
        )


class Stage1VerifierTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.verifier = load_file(
            "stage1_verifier_test", EXPERIMENT / "verify_stage1_results.py"
        )

    def record(self, index: int):
        source_arm, trigger = self.verifier.CELLS[index]
        pause_base = index == 3
        pause_fix = index == 4
        offsets = [0.1, 0.2, 10.3] if pause_base else [0.1, 0.2, 0.3]
        return {
            "accepted_batch_count": 1,
            "build_identity_sha256": "9" * 64,
            "campaign_error_kind": None,
            "cell_index": index,
            "classification": "pass",
            "completion_tokens": 32,
            "dropped_batch_count": 1 if pause_fix else 0,
            "engine_core_bound": True,
            "engine_core_kv_events_sha256": "e" * 64,
            "engine_core_mapped_worktree_binaries": {"vllm/_C.abi3.so": "8" * 64},
            "environment": {
                "cuda": "13.0",
                "cuda_available": True,
                "gpu_capability": [8, 9],
                "gpu_count": 1,
                "gpu_name": "test-gpu",
                "import_matches_tree": True,
                "kv_events_relative_file": "vllm/distributed/kv_events.py",
                "kv_events_sha256": "e" * 64,
                "python": "3.12.0",
                "torch": "2.13.0+cu130",
                "tree_kv_events_sha256": "e" * 64,
                "vllm": "test",
                "vllm_relative_file": "vllm/__init__.py",
            },
            "health_during_stall": "2xx" if pause_base else None,
            "hook_ready": True,
            "hook_error_kind": None,
            "hook_instance_count": 1,
            "implementation_sha256": {
                name: self.verifier.sha256(path)
                for name, path in self.verifier.HASHED_IMPLEMENTATION.items()
            },
            "observer_authorization": "pr_set_ptracer_observer",
            "private_server_log_sha256": "a" * 64,
            "progress": {
                "completed_offset_seconds": offsets[-1] + 0.1,
                "progress_count": len(offsets),
                "progress_offsets_seconds": offsets,
            },
            "progress_count_at_release": 2 if pause_base else 0,
            "prompt_tokens": 16,
            "recovered_after_release": True if pause_base else None,
            "release_observed": True,
            "release_offset_seconds": 10.2
            if pause_base
            else (0.5 if pause_fix else None),
            "request_sha256": "b" * 64,
            "schema_version": 1,
            "server_command_sha256": "c" * 64,
            "cleanup": {
                "engine_core_gone": True,
                "process_group_gone": True,
                "termination": "terminated",
            },
            "source": {
                "base_commit": self.verifier.BASE_COMMIT,
                "fix_head": self.verifier.FIX_HEAD,
                "fix_patch_sha256": self.verifier.FIX_PATCH_SHA256,
                "tree": (
                    self.verifier.BASE_TREE
                    if source_arm == "base"
                    else self.verifier.FIX_TREE
                ),
            },
            "source_arm": source_arm,
            "stack": (
                {"available": True, "match": True, "raw_sha256": "d" * 64}
                if pause_base
                else {"available": None, "match": None, "raw_sha256": None}
            ),
            "stalled_before_release": pause_base,
            "stream_error_kind": None,
            "trigger": trigger,
            "yama_ptrace_scope": 1,
        }

    def test_verifier_accepts_the_exact_four_cells(self) -> None:
        for index in self.verifier.CELLS:
            self.verifier.verify_cell(self.record(index), index)

    def test_verifier_rejects_missing_stack_producer(self) -> None:
        record = deepcopy(self.record(3))
        record["stack"] = {
            "available": False,
            "match": None,
            "raw_sha256": None,
        }
        with self.assertRaisesRegex(AssertionError, "stack unavailable"):
            self.verifier.verify_cell(record, 3)

    def test_verifier_rejects_control_that_never_reached_queue(self) -> None:
        record = self.record(1)
        record["accepted_batch_count"] = 0
        with self.assertRaisesRegex(AssertionError, "trigger path"):
            self.verifier.verify_cell(record, 1)

    def test_verifier_rejects_engine_core_import_shadowing(self) -> None:
        record = self.record(1)
        record["engine_core_kv_events_sha256"] = "f" * 64
        with self.assertRaisesRegex(AssertionError, "EngineCore kv_events"):
            self.verifier.verify_cell(record, 1)

    def test_bounded_campaign_error_does_not_retain_message(self) -> None:
        self.assertEqual(
            Stage1CampaignTest.campaign.bounded_error_kind(RuntimeError("secret")),
            "contract_error",
        )


class Stage1aVerifierTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(EXPERIMENT))
        try:
            cls.verifier = load_file(
                "stage1a_verifier_test", EXPERIMENT / "verify_stage1a.py"
            )
        finally:
            sys.path.remove(str(EXPERIMENT))

    def record(self, source_arm: str):
        base = source_arm == "base"
        counts_before = {
            "accepted_batch_count": 1,
            "dropped_batch_count": 0 if base else 1,
        }
        counts_after = (
            {"accepted_batch_count": 2, "dropped_batch_count": 0}
            if base
            else counts_before.copy()
        )
        plugin = (
            self.verifier.PLUGIN_ROOT
            / "src"
            / "dfx_stage1_backpressure"
            / "__init__.py"
        )
        return {
            "cleanup": {
                "engine_core_gone": True,
                "process_group_gone": True,
                "termination": "exited",
            },
            "counts_after_release": counts_after,
            "counts_before_release": counts_before,
            "engine_core_kv_events_sha256": "a" * 64,
            "engine_core_mapped_worktree_binaries": {},
            "environment": {
                "python": "3.12",
                "torch": "2.13.0+cu130",
                "cuda": None,
                "cuda_available": False,
                "gpu_count": 0,
                "import_matches_tree": True,
                "kv_events_sha256": "a" * 64,
                "tree_kv_events_sha256": "a" * 64,
            },
            "implementation_sha256": {
                "preflight": self.verifier.sha256_file(
                    EXPERIMENT / "stage1a_cpu_preflight.py"
                ),
                "campaign": self.verifier.sha256_file(
                    EXPERIMENT / "stage1_campaign.py"
                ),
                "contract": self.verifier.sha256_file(
                    EXPERIMENT / "stage1_contract.py"
                ),
                "plugin": self.verifier.sha256_file(plugin),
            },
            "ready_authorization": "pr_set_ptracer_observer",
            "schema_version": 1,
            "source_arm": source_arm,
            "source_tree": (
                self.verifier.BASE_TREE if base else self.verifier.FIX_TREE
            ),
            "stack": (
                {"available": True, "match": True, "raw_sha256": "b" * 64}
                if base
                else {"available": None, "match": None, "raw_sha256": None}
            ),
            "stack_attempt_count": 1 if base else 0,
            "subject_return_code": 0,
            "verdict": "PASS",
            "yama_ptrace_scope": 1,
        }

    def test_verifier_accepts_both_real_publisher_arms(self) -> None:
        self.verifier.verify_record(self.record("base"), "base")
        self.verifier.verify_record(self.record("fix"), "fix")

    def test_verifier_rejects_permissive_yama(self) -> None:
        record = self.record("base")
        record["yama_ptrace_scope"] = 0
        with self.assertRaisesRegex(AssertionError, "not restrictive"):
            self.verifier.verify_record(record, "base")

    def test_verifier_rejects_engine_core_import_shadowing(self) -> None:
        record = self.record("base")
        record["engine_core_kv_events_sha256"] = "f" * 64
        with self.assertRaisesRegex(AssertionError, "EngineCore kv_events"):
            self.verifier.verify_record(record, "base")


class Stage1BuildIdentityVerifierTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.verifier = load_file(
            "stage1_build_verifier_test",
            EXPERIMENT / "verify_stage1_build_identity.py",
        )
        cls.generator = load_file(
            "stage1_build_generator_test",
            EXPERIMENT / "stage1_build_identity.py",
        )
        cls.result_dir = (
            ROOT / "results" / ("vllm-zmq-backpressure-stage1-build-20260915")
        )

    def record(self, arm: str):
        path = self.result_dir / f"stage1-build-{arm}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_verifier_accepts_both_exact_build_identities(self) -> None:
        records = {arm: self.record(arm) for arm in ("base", "fix")}
        for arm, record in records.items():
            self.verifier.verify(record, arm)
        self.verifier.verify_pair(records)

    def test_verifier_rejects_a_different_wheel(self) -> None:
        record = self.record("base")
        record["wheel_sha256"] = "f" * 64
        with self.assertRaisesRegex(AssertionError, "wheel hash"):
            self.verifier.verify(record, "base")

    def test_verifier_rejects_a_cross_arm_dependency_change(self) -> None:
        records = {arm: self.record(arm) for arm in ("base", "fix")}
        records["fix"]["distributions"]["aiohttp"]["version"] = "unexpected"
        with self.assertRaisesRegex(AssertionError, "dependency manifest"):
            self.verifier.verify_pair(records)

    def test_record_verification_detects_installed_file_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "example.py"
            package.write_bytes(b"original\n")
            metadata = root / "example-1.0.dist-info"
            metadata.mkdir()
            digest = base64.urlsafe_b64encode(
                hashlib.sha256(package.read_bytes()).digest()
            ).rstrip(b"=")
            (metadata / "RECORD").write_text(
                f"example.py,sha256={digest.decode()},9\n"
                "example-1.0.dist-info/RECORD,,\n",
                encoding="utf-8",
            )
            verified, owned = self.generator.verify_record_files(metadata)
            self.assertEqual(verified, 1)
            self.assertIn("example.py", owned)
            package.write_bytes(b"tampered\n")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                self.generator.verify_record_files(metadata)


class Stage1PluginTest(unittest.TestCase):
    def setUp(self) -> None:
        self.plugin = load_file("stage1_plugin_test", PLUGIN)

    def install_fake_vllm(self):
        vllm = ModuleType("vllm")
        distributed = ModuleType("vllm.distributed")
        kv_events = ModuleType("vllm.distributed.kv_events")

        class Publisher:
            _dfx_stage1_wrapped = False

            def __init__(self) -> None:
                self._event_queue = queue.Queue(maxsize=1)
                self.production_entered = threading.Event()

            def _publisher_thread(self) -> None:
                self.production_entered.set()

        kv_events.ZmqEventPublisher = Publisher
        return Publisher, {
            "vllm": vllm,
            "vllm.distributed": distributed,
            "vllm.distributed.kv_events": kv_events,
        }

    def test_disabled_plugin_does_not_import_vllm(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.plugin.register()

    def test_file_release_and_drop_counter_are_cross_process_safe(self) -> None:
        publisher_class, modules = self.install_fake_vllm()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kv_events_path = root / "kv_events.py"
            kv_events_path.write_text("# test module\n", encoding="utf-8")
            vllm_init = root / "vllm" / "__init__.py"
            vllm_init.parent.mkdir()
            vllm_init.write_text("# test package\n", encoding="utf-8")
            modules["vllm"].__file__ = str(vllm_init)
            modules["vllm.distributed.kv_events"].__file__ = str(kv_events_path)
            environment = {
                self.plugin.ENABLE_ENV: "1",
                self.plugin.CONTROL_DIR_ENV: str(root),
            }
            with (
                patch.dict(sys.modules, modules),
                patch.dict("os.environ", environment, clear=True),
                patch.object(self.plugin, "YAMA_SCOPE", root / "no-yama"),
                patch.object(
                    self.plugin, "_process_start_time_ticks", return_value=100
                ),
            ):
                self.plugin.register()
                publisher = publisher_class()
                worker = threading.Thread(target=publisher._publisher_thread)
                worker.start()
                ready = root / f"ready-{os.getpid()}.json"
                for _ in range(100):
                    if ready.exists():
                        break
                    threading.Event().wait(0.01)
                self.assertTrue(ready.exists())
                ready_record = json.loads(ready.read_text())
                self.assertEqual(
                    ready_record["plugin_sha256"], self.plugin._plugin_sha256()
                )
                self.assertEqual(
                    ready_record["kv_events_sha256"],
                    self.plugin.hashlib.sha256(kv_events_path.read_bytes()).hexdigest(),
                )
                self.assertIsNone(ready_record["yama_ptrace_scope"])
                self.assertFalse(publisher.production_entered.is_set())

                publisher._event_queue.put("first")
                with self.assertRaises(queue.Full):
                    publisher._event_queue.put_nowait("second")
                counts = json.loads(
                    (root / f"queue-counts-{os.getpid()}.json").read_text()
                )
                self.assertEqual(counts["accepted_batch_count"], 1)
                self.assertEqual(counts["dropped_batch_count"], 1)

                (root / "release").touch()
                worker.join(timeout=2)
                self.assertFalse(worker.is_alive())
                self.assertTrue(publisher.production_entered.is_set())
                self.assertTrue((root / f"released-{os.getpid()}.json").exists())

    def test_register_is_idempotent(self) -> None:
        publisher_class, modules = self.install_fake_vllm()
        with tempfile.TemporaryDirectory() as temporary:
            environment = {
                self.plugin.ENABLE_ENV: "1",
                self.plugin.CONTROL_DIR_ENV: temporary,
            }
            with (
                patch.dict(sys.modules, modules),
                patch.dict("os.environ", environment, clear=True),
            ):
                self.plugin.register()
                first = publisher_class._publisher_thread
                self.plugin.register()
                self.assertIs(publisher_class._publisher_thread, first)


if __name__ == "__main__":
    unittest.main()
