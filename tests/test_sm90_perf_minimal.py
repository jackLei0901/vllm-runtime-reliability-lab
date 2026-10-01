"""CPU-only successor protocol tests; no GPU results are implied."""

import copy
import importlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "experiments/kernel-operand-contracts/sm90-block-fp8-perf-minimal"
NAMES = (
    "protocol",
    "runtime",
    "build_perf",
    "build_pair",
    "correctness",
    "timing",
    "decision",
    "apparatus",
    "replay_check",
    "trace_tools",
    "serve_trace",
    "model_preflight",
    "freeze_packet",
    "witness_review",
)


class TestSM90PerfMinimal(unittest.TestCase):
    def setUp(self):
        self.module_patch = patch.dict(sys.modules)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        self.path_patch = patch.object(sys, "path", [str(PACKET), *sys.path])
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)
        for name in NAMES:
            sys.modules.pop(name, None)
        self.mods = {name: importlib.import_module(name) for name in NAMES}
        self.protocol = self.mods["protocol"]
        self.decision = self.mods["decision"]

    def records(self, improved_m=7, shape_count=3):
        rows = []
        changed = self.protocol.CHANGED_M[:improved_m]
        for shape_index, shape in enumerate(self.protocol.SHAPES):
            for m in self.protocol.M_VALUES:
                common = {
                    "shape": list(shape),
                    "left_m": m,
                    "right_m": m,
                    "left_cache": "rotating",
                    "right_cache": "rotating",
                }
                rows.append({**common, "kind": "AA", "status": "calibrated"})
                improves = m in changed and shape_index < shape_count
                rows.append(
                    {
                        **common,
                        "kind": "variant",
                        "status": "positive" if improves else "below_positive_gate",
                        "median_difference_us": 2.0 if improves else 0.0,
                        "bound_us": 1.0,
                    }
                )
        return rows

    def test_seven_m_pass_six_fail(self):
        self.assertEqual(
            self.decision.timing_outcome(self.records())["status"], "worth_proposing"
        )
        self.assertEqual(
            self.decision.timing_outcome(self.records(6))["status"],
            "not_worth_proposing",
        )

    def test_three_shapes_pass_two_fail(self):
        self.assertEqual(
            self.decision.timing_outcome(self.records(shape_count=2))["status"],
            "not_worth_proposing",
        )

    def test_counts_by_m_not_by_shape(self):
        rows = self.records(9, 4)
        for row in rows:
            if row["kind"] == "variant" and row["left_m"] in self.protocol.CHANGED_M:
                omitted = self.protocol.CHANGED_M.index(row["left_m"]) % 3
                if tuple(row["shape"]) == self.protocol.SHAPES[omitted]:
                    row.update(status="below_positive_gate", median_difference_us=0.0)
        result = self.decision.timing_outcome(rows)
        self.assertEqual(result["improved_m_count"], 9)
        self.assertEqual(result["status"], "worth_proposing")
        by_shape = [
            sum(
                r["kind"] == "variant"
                and tuple(r["shape"]) == shape
                and r["status"] == "positive"
                for r in rows
            )
            for shape in self.protocol.SHAPES
        ]
        self.assertEqual(sum(count >= 7 for count in by_shape), 1)

    def test_missing_duplicate_extra_and_unscored_fail_closed(self):
        originals = self.records()
        variants = [
            originals[:-1],
            originals[:-1] + [copy.deepcopy(originals[0])],
            originals + [copy.deepcopy(originals[0])],
        ]
        for kind in ("AA", "variant"):
            rows = copy.deepcopy(originals)
            next(r for r in rows if r["kind"] == kind)["status"] = "unscored"
            variants.append(rows)
        for rows in variants:
            with self.subTest(rows=len(rows)):
                self.assertEqual(
                    self.decision.timing_outcome(rows)["status"],
                    "insufficient_evidence",
                )

    def test_controls_both_directions_override_regression(self):
        for sign in (-1, 1):
            rows = self.records()
            control = next(
                r for r in rows if r["kind"] == "variant" and r["left_m"] == 1
            )
            control.update(
                median_difference_us=sign * 2.0,
                status="positive" if sign == 1 else "negative",
            )
            changed = next(
                r for r in rows if r["kind"] == "variant" and r["left_m"] == 4
            )
            changed.update(median_difference_us=-2.0, status="negative")
            self.assertEqual(
                self.decision.timing_outcome(rows)["status"], "insufficient_evidence"
            )

    def test_changed_regression_rejects(self):
        rows = self.records()
        row = next(r for r in rows if r["kind"] == "variant" and r["left_m"] == 4)
        row.update(median_difference_us=-2.0, status="negative")
        self.assertEqual(
            self.decision.timing_outcome(rows)["status"], "not_worth_proposing"
        )

    def test_bound_equality_is_not_control_failure(self):
        rows = self.records()
        row = next(r for r in rows if r["kind"] == "variant" and r["left_m"] == 1)
        row["median_difference_us"] = -1.0
        self.assertEqual(
            self.decision.timing_outcome(rows)["status"], "worth_proposing"
        )

    def test_nonfinite_and_invalid_bounds_fail_closed(self):
        for field, value in (
            ("bound_us", 0.0),
            ("bound_us", float("nan")),
            ("median_difference_us", float("inf")),
        ):
            rows = self.records()
            next(r for r in rows if r["kind"] == "variant")[field] = value
            self.assertEqual(
                self.decision.timing_outcome(rows)["status"], "insufficient_evidence"
            )

    def correctness_rows(self):
        return [
            {"arm": arm, "case": case, "status": "passed"}
            for arm in ("base", "variant")
            for case in self.mods["correctness"].cases()
        ]

    def test_correctness_only_variant_numerical_failure_rejects(self):
        rows = self.correctness_rows()
        self.assertEqual(len(rows), 176)
        self.assertEqual(self.decision.correctness_outcome(rows), "passed")
        rows[-1]["status"] = "failed"
        self.assertEqual(self.decision.correctness_outcome(rows), "not_worth_proposing")
        rows[-1]["status"] = "unscored"
        self.assertEqual(self.decision.correctness_outcome(rows), "unscored")
        rows[-1]["status"] = "passed"
        rows[0]["status"] = "failed"
        self.assertEqual(self.decision.correctness_outcome(rows), "unscored")

    def test_duplicate_correctness_cannot_replace_case(self):
        rows = self.correctness_rows()
        rows[-1] = copy.deepcopy(rows[0])
        self.assertEqual(self.decision.correctness_outcome(rows), "unscored")

    def test_single_plan_matches_b_and_ab_plans_stay_present(self):
        timing = self.mods["timing"]
        self.assertEqual(timing.plan("S"), timing.plan("B"))
        self.assertEqual(tuple(map(len, timing.plan("S"))), (80, 80))
        self.assertEqual(tuple(map(len, timing.plan("A"))), (92, 16))

    def test_s_cli_refuses_bad_freeze_without_a_or_workload(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            command = [
                sys.executable,
                str(PACKET / "timing.py"),
                "--session",
                "S",
                "--base",
                "absent",
                "--variant",
                "absent",
                "--dispatch",
                "absent",
                "--correctness",
                "absent",
                "--out",
                str(out),
                "--freeze-commit",
                "bad",
            ]
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=10
            )
            self.assertNotEqual(completed.returncode, 0)
            record = json.loads((out / "raw.json").read_text())
            self.assertEqual(record["status"], "unscored")
            self.assertIn("public freeze", record["failure"]["message"])
            self.assertNotIn("gpu", record)

    def test_blocked_process_classifier(self):
        fn = self.mods["apparatus"].blocked_process
        for command in (
            ["/usr/bin/nvcc"],
            ["python", "-m", "vllm.entrypoints.openai.api_server"],
            ["/usr/bin/nsys", "profile"],
            ["python", "/packet/build_perf.py", "--worker"],
        ):
            self.assertTrue(fn(command))
        self.assertFalse(fn(["python", "timing.py"]))

    def test_idle_snapshot_other_gpu_pid_fails(self):
        app = self.mods["apparatus"]
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                app.subprocess, "check_output", side_effect=["999\n", "GPU-test, 0\n"]
            ),
        ):
            with self.assertRaisesRegex(ValueError, "not idle"):
                app.idle_receipt(Path(tmp), own_pid=123)

    def test_idle_snapshot_empty_context_required(self):
        app = self.mods["apparatus"]
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(
                app.subprocess, "check_output", side_effect=["", "GPU-test, 128\n"]
            ),
        ):
            self.assertEqual(
                app.idle_receipt(Path(tmp), own_pid=123)["status"], "idle_snapshot"
            )

    def test_idle_snapshot_host_pid_self_unknown_and_hidden_memory(self):
        app = self.mods["apparatus"]
        for pids, memory in (
            ("123\n", "0"),
            ("999\n", "0"),
            ("[N/A]\n", "0"),
            ("", "129"),
        ):
            with (
                self.subTest(pids=pids, memory=memory),
                tempfile.TemporaryDirectory() as tmp,
                patch.object(
                    app.subprocess,
                    "check_output",
                    side_effect=[pids, f"GPU-test, {memory}\n"],
                ),
            ):
                with self.assertRaises(ValueError):
                    app.idle_receipt(Path(tmp), own_pid=123)

    def test_memory_headroom_boundaries_and_unknown(self):
        app = self.mods["apparatus"]
        with tempfile.TemporaryDirectory() as tmp:
            proc = Path(tmp) / "proc"
            cg = Path(tmp) / "cg"
            proc.mkdir()
            cg.mkdir()
            (proc / "meminfo").write_text(
                "MemTotal: 134217728 kB\nMemAvailable: 125829120 kB\n"
            )
            self.assertFalse(app.memory_headroom(proc, cg)["e4_eligible"])
            for limit, used, eligible in (
                (96, 32, True),
                (95, 0, False),
                (96, 33, False),
                (120, 10, True),
            ):
                (cg / "memory.max").write_text(str(limit * app.GIB))
                (cg / "memory.current").write_text(str(used * app.GIB))
                self.assertEqual(app.memory_headroom(proc, cg)["e4_eligible"], eligible)

    def test_idle_only_cli_checks_idle_without_build_memory_gate(self):
        app = self.mods["apparatus"]
        for failure in (False, True):
            with (
                tempfile.TemporaryDirectory() as tmp,
                patch.object(
                    sys,
                    "argv",
                    [
                        "apparatus.py",
                        "--idle-only",
                        "--out",
                        str(Path(tmp) / "out"),
                        "--freeze-commit",
                        "a" * 40,
                    ],
                ),
                patch.object(self.mods["runtime"], "public_freeze", return_value={}),
                patch.object(app, "memory_headroom") as memory,
                patch.object(
                    app,
                    "idle_receipt",
                    return_value={"status": "idle_snapshot"},
                    side_effect=ValueError("GPU not idle") if failure else None,
                ),
            ):
                if failure:
                    with self.assertRaises(SystemExit):
                        app.main()
                else:
                    app.main()
                memory.assert_not_called()
                record = json.loads((Path(tmp) / "out/identity.json").read_text())
                self.assertEqual(
                    record["status"], "unscored" if failure else "idle_passed"
                )
                self.assertNotIn("memory", record)

    def test_runbook_idle_guards_and_deferred_replay_order(self):
        text = (PACKET / "RUNBOOK.zh-CN.md").read_text(encoding="utf-8")
        e2 = text.index("### 3.2 E2")
        e1 = text.index("### 3.3 E1")
        e3 = text.index("### 3.4 E3")
        seal = text.index("### 3.5")
        self.assertLess(e2, text.index('--idle-only --out "$WORK/pre-e2-idle"'))
        self.assertLess(text.index('--idle-only --out "$WORK/pre-e2-idle"'), e1)
        self.assertLess(e1, text.index('--idle-only --out "$WORK/pre-e1-idle"'))
        self.assertLess(text.index('--idle-only --out "$WORK/pre-e1-idle"'), e3)
        self.assertGreater(text.index('"$P/replay_check.py"'), seal)

    def test_build_shared_deadline_limits_second_timeout(self):
        now = datetime(2026, 10, 1, tzinfo=timezone.utc)
        fn = self.mods["build_perf"].remaining_timeout
        self.assertEqual(
            fn(1200, (now + timedelta(seconds=2100)).isoformat(), now), 1200
        )
        self.assertEqual(fn(1200, (now + timedelta(seconds=900)).isoformat(), now), 900)
        with self.assertRaises(TimeoutError):
            fn(1200, now.isoformat(), now)
        with self.assertRaises(ValueError):
            fn(1200, "2026-10-01T00:00:00", now)

    def test_build_pair_order_and_failure_no_second_arm(self):
        mod = self.mods["build_pair"]
        for codes, arms in (([0, 0], ["base", "variant"]), ([1], ["base"])):
            with (
                tempfile.TemporaryDirectory() as tmp,
                patch.object(
                    sys,
                    "argv",
                    [
                        "build_pair.py",
                        "--base-src",
                        "base",
                        "--variant-src",
                        "variant",
                        "--cutlass-src",
                        "cutlass",
                        "--cmake-reference",
                        "reference",
                        "--out",
                        str(Path(tmp) / "result"),
                        "--freeze-commit",
                        "a" * 40,
                    ],
                ),
                patch.object(mod.runtime, "public_freeze", return_value={}),
                patch.object(mod.subprocess, "call", side_effect=codes) as call,
            ):
                if codes[0]:
                    with self.assertRaises(SystemExit):
                        mod.main()
                else:
                    mod.main()
                commands = [c.args[0] for c in call.call_args_list]
                self.assertEqual([c[c.index("--arm") + 1] for c in commands], arms)
                deadlines = [c[c.index("--deadline-utc") + 1] for c in commands]
                self.assertEqual(len(set(deadlines)), 1)

    def test_projection_boundary(self):
        fn = self.mods["timing"].projection_fits
        self.assertTrue(fn(3, 3, 160, 157))
        self.assertFalse(fn(3, 3, 160, 156))

    def test_idle_call_precedes_gpu_initialization(self):
        # Inspect call order without importing torch or requiring Linux/GPU.
        import ast

        tree = ast.parse((PACKET / "timing.py").read_text())
        calls = {
            node.func.attr: node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("idle_receipt", "gpu")
        }
        self.assertLess(calls["idle_receipt"], calls["gpu"])

    def test_collector_resets_v1_and_sets_localhost_controls(self):
        with patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "0"}):
            env = self.mods["serve_trace"].launch_env()
        self.assertNotIn("VLLM_USE_V2_MODEL_RUNNER", env)
        self.assertEqual(env["VLLM_SERVER_DEV_MODE"], "1")
        self.assertIn("127.0.0.1", env["NO_PROXY"])

    def loads(self):
        return [
            {
                "concurrency": level,
                "start_utc_ns": 10**18 + i * 10_000,
                "end_utc_ns": 10**18 + i * 10_000 + 9000,
                "start_monotonic_ns": i * 10_000,
                "end_monotonic_ns": i * 10_000 + 9000,
                "usage": [{"completion_tokens": 64}] * level,
            }
            for i, level in enumerate((1, 16, 63, 64))
        ]

    def database(
        self,
        path,
        wrong_count=False,
        wrong_class=False,
        cross_pid=False,
        ambiguous=False,
    ):
        db = sqlite3.connect(path)
        db.executescript(
            "CREATE TABLE StringIds(id INTEGER,value TEXT); "
            "CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,"
            "globalTid INTEGER,text TEXT); "
            "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,"
            "globalTid INTEGER,correlationId INTEGER,nameId INTEGER,"
            "returnValue INTEGER); "
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,"
            "globalPid INTEGER,correlationId INTEGER,demangledName INTEGER,"
            "graphNodeId INTEGER);"
        )
        db.executemany(
            "INSERT INTO StringIds VALUES(?,?)",
            [
                (1, "cudaGraphLaunch_v10000"),
                (2, "cutlass_3x_gemm_fp8_blockwise_Pingpong"),
                (3, "cutlass_3x_gemm_fp8_blockwise_Cooperative"),
            ],
        )
        pid = 4 << 24
        for index, load in enumerate(self.loads()):
            for step in range(10):
                start = load["start_utc_ns"] + 100 + step * 500
                corr = index * 100 + step
                db.execute(
                    "INSERT INTO NVTX_EVENTS VALUES(?,?,?,?)",
                    (
                        start,
                        start + 100,
                        pid + 3,
                        "execute_64_context_0(...)_generation_64(...)",
                    ),
                )
                if ambiguous and index == step == 0:
                    db.execute(
                        "INSERT INTO NVTX_EVENTS VALUES(?,?,?,?)",
                        (
                            start,
                            start + 100,
                            pid + 3,
                            "execute_64_context_0(...)_generation_64(...)",
                        ),
                    )
                db.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?,?,?)",
                    (start + 10, start + 20, pid + 3, corr, 1, 0),
                )
                count = 143 if wrong_count and index == step == 0 else 144
                for node in range(count):
                    kernel_pid = (
                        pid + (1 << 24) if cross_pid and index == step == 0 else pid
                    )
                    kind = 3 if index else 2
                    if wrong_class and index == step == 0:
                        kind = 3
                    # GPU events deliberately lie AFTER the CPU NVTX range.
                    db.execute(
                        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?)",
                        (start + 150, start + 160, kernel_pid, corr, kind, node),
                    )
        db.commit()
        db.close()

    def config(self):
        return {
            "compilation_config": {
                "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 64],
                "cudagraph_mode": "FULL_AND_PIECEWISE",
            }
        }

    def test_replay_correlates_after_cpu_range_and_never_claims_exact_m(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.sqlite"
            self.database(path)
            result = self.mods["replay_check"].witness(
                path, self.loads(), self.config()
            )
            self.assertEqual(result["status"], "witnessed")
            self.assertEqual(result["exact_m"], "unmeasured")

    def test_replay_mutations_leave_level_unwitnessed(self):
        for mutation in ("wrong_count", "wrong_class", "cross_pid", "ambiguous"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "trace.sqlite"
                self.database(path, **{mutation: True})
                result = self.mods["replay_check"].witness(
                    path, self.loads(), self.config()
                )
                self.assertEqual(result["levels"][1]["status"], "unwitnessed")

    def test_windows_missing_duplicate_overlap_and_clock_fail_closed(self):
        variants = [self.loads()[:-1], self.loads() + [self.loads()[0]]]
        rows = self.loads()
        rows[1]["start_utc_ns"] = rows[0]["start_utc_ns"]
        variants.append(rows)
        rows = self.loads()
        rows[0]["end_utc_ns"] += 100_000_000
        variants.append(rows)
        for rows in variants:
            with self.assertRaises(ValueError):
                self.mods["replay_check"].batch_windows(rows)

    def test_manifest_canonical_bytes(self):
        freezer = self.mods["freeze_packet"]
        record = {"files_sha256": {"input": "digest"}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest"
            freezer.write_manifest(path, record)
            freezer.check_manifest(path, record)
            self.assertNotIn(b"\r\n", path.read_bytes())

    def freeze_fixture(self, tmp):
        runtime = self.mods["runtime"]
        path = Path(tmp) / "manifest.json"
        raw = b'{"files_sha256": {"input": "digest"}}\n'
        path.write_bytes(raw)
        for context in (
            patch.object(runtime, "MANIFEST", path),
            patch.object(
                self.mods["freeze_packet"], "entries", return_value={"input": "digest"}
            ),
        ):
            context.start()
            self.addCleanup(context.stop)
        return runtime, path, raw

    def test_public_freeze_fetch_once_then_offline_with_network_down(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime, _, raw = self.freeze_fixture(tmp)
            with patch.object(runtime.urllib.request, "urlopen") as network:
                network.return_value.__enter__.return_value.read.return_value = raw
                receipt = runtime.verify_public_freeze("a" * 40)
                network.assert_called_once()
            path = Path(tmp) / "receipt.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with (
                patch.dict(os.environ, {runtime.FREEZE_RECEIPT_ENV: str(path)}),
                patch.object(
                    runtime.urllib.request,
                    "urlopen",
                    side_effect=OSError("network down"),
                ) as network,
            ):
                for _ in range(10):
                    self.assertEqual(
                        runtime.public_freeze("a" * 40)["commit"], "a" * 40
                    )
                network.assert_not_called()

    def test_offline_freeze_receipt_mutations_and_missing_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime, manifest, raw = self.freeze_fixture(tmp)
            with patch.object(runtime.urllib.request, "urlopen") as network:
                network.return_value.__enter__.return_value.read.return_value = raw
                good = runtime.verify_public_freeze("a" * 40)
            path = Path(tmp) / "receipt.json"
            with patch.object(
                runtime.urllib.request,
                "urlopen",
                side_effect=AssertionError("must stay offline"),
            ) as network:
                for key, value in (
                    ("status", "unscored"),
                    ("schema_version", 2),
                    ("commit", "b" * 40),
                    ("manifest_sha256", "0" * 64),
                    ("url", "https://example.com"),
                    ("verified_at_unix_ns", 0),
                ):
                    bad = {**good, key: value}
                    path.write_text(json.dumps(bad), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        runtime.public_freeze("a" * 40, path)
                with patch.dict(os.environ, {runtime.FREEZE_RECEIPT_ENV: ""}):
                    with self.assertRaises(ValueError):
                        runtime.public_freeze("a" * 40)
                path.write_text(json.dumps(good), encoding="utf-8")
                manifest.write_bytes(raw + b" ")
                with self.assertRaises(ValueError):
                    runtime.public_freeze("a" * 40, path)
                network.assert_not_called()

    def test_freeze_preflight_network_failure_preserved_not_reusable(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime, _, _ = self.freeze_fixture(tmp)
            path = Path(tmp) / "receipt.json"
            with (
                patch.object(
                    sys,
                    "argv",
                    ["runtime.py", "--freeze-commit", "a" * 40, "--out", str(path)],
                ),
                patch.object(
                    runtime.urllib.request,
                    "urlopen",
                    side_effect=TimeoutError("offline"),
                ),
            ):
                with self.assertRaises(SystemExit):
                    runtime.freeze_preflight()
                self.assertEqual(json.loads(path.read_text())["status"], "unscored")
                with self.assertRaises(ValueError):
                    runtime.public_freeze("a" * 40, path)
                with self.assertRaises(ValueError):
                    runtime.freeze_preflight()

    def test_freeze_preflight_success_receipt_and_child_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime, _, raw = self.freeze_fixture(tmp)
            path = Path(tmp) / "public_freeze.json"
            with (
                patch.object(
                    sys,
                    "argv",
                    ["runtime.py", "--freeze-commit", "a" * 40, "--out", str(path)],
                ),
                patch.object(runtime.urllib.request, "urlopen") as network,
            ):
                network.return_value.__enter__.return_value.read.return_value = raw
                runtime.freeze_preflight()
                network.assert_called_once()
            with (
                patch.dict(os.environ, {runtime.FREEZE_RECEIPT_ENV: str(path)}),
                patch.object(
                    runtime.urllib.request,
                    "urlopen",
                    side_effect=AssertionError("offline"),
                ),
            ):
                self.assertEqual(runtime.public_freeze("a" * 40)["commit"], "a" * 40)
                self.assertEqual(
                    self.mods["serve_trace"].launch_env()[runtime.FREEZE_RECEIPT_ENV],
                    str(path),
                )
                with patch.object(
                    self.mods["freeze_packet"], "entries", return_value={}
                ):
                    with self.assertRaises(ValueError):
                        runtime.public_freeze("a" * 40)


if __name__ == "__main__":
    unittest.main()
