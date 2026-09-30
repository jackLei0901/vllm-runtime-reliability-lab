"""CPU apparatus checks, not measurements or evidence of GPU compatibility."""

import importlib
import io
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import zipfile
from contextlib import closing, redirect_stderr
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "experiments/kernel-operand-contracts/sm90-block-fp8-perf"
NAMES = (
    "runtime",
    "protocol",
    "build_perf",
    "correctness",
    "timing",
    "trace_tools",
    "model_preflight",
    "serve_trace",
    "witness_review",
    "freeze_packet",
    "summarize",
    "wheel_preflight",
    "serving_preflight",
)
with patch.dict(sys.modules), patch.object(sys, "path", [str(PACKET), *sys.path]):
    for name in NAMES:
        sys.modules.pop(name, None)
    modules = {name: importlib.import_module(name) for name in NAMES}

runtime, protocol = modules["runtime"], modules["protocol"]
correctness, timing = modules["correctness"], modules["timing"]
trace, witness = modules["trace_tools"], modules["witness_review"]


class TestSM90PerfTools(unittest.TestCase):
    def test_manifest_uses_exact_lf_bytes_and_refuses_overwrite(self):
        freezer = modules["freeze_packet"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            record = {"files_sha256": {"a": "digest"}, "status": "test"}
            freezer.write_manifest(path, record)
            self.assertEqual(path.read_bytes(), freezer.manifest_bytes(record))
            self.assertNotIn(b"\r\n", path.read_bytes())
            freezer.check_manifest(path, record)
            with self.assertRaises(FileExistsError):
                freezer.write_manifest(path, record)
            with path.open("wb") as file:
                file.write(freezer.manifest_bytes(record).replace(b"\n", b"\r\n"))
            self.assertEqual(json.loads(path.read_bytes()), record)
            with self.assertRaisesRegex(ValueError, "UTF-8/LF"):
                freezer.check_manifest(path, record)

    def test_installation_preflight_refuses_invalid_freeze_before_loading_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "new"
            argv = [sys.executable, str(PACKET / "serving_preflight.py")]
            for name in (
                "vllm-src",
                "index",
                "wheel",
                "install-log",
                "install-command-file",
            ):
                argv.extend(["--" + name, str(Path(directory) / "absent")])
            argv.extend(["--out", str(out), "--freeze-commit", "not-a-commit"])
            result = subprocess.run(argv, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            receipt = json.loads((out / "installation.json").read_text())
            self.assertEqual(receipt["status"], "unscored")
            self.assertIn("full public freeze commit", receipt["failure"]["message"])
            self.assertNotIn("environment", receipt)
            self.assertFalse((out / "serving_build.json").exists())

    def test_installation_preflight_rejects_empty_install_command(self):
        checker = modules["serving_preflight"]
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "log"
            command = Path(directory) / "command"
            log.write_text("installer output")
            command.write_text("")
            with patch.object(
                checker.wheel_preflight, "inspect_archive", return_value={}
            ):
                with self.assertRaisesRegex(ValueError, "nonempty"):
                    checker.prepare_provenance(
                        Path(directory), Path("index"), Path("wheel"), log, command, {}
                    )

    def test_preflight_checks_dependencies_without_initializing_cuda(self):
        import types

        torch = types.SimpleNamespace(
            __version__="2.13.0+cu130", version=types.SimpleNamespace(cuda="13.0")
        )
        checker = modules["serving_preflight"]
        with (
            patch.dict(sys.modules, {"torch": torch}),
            patch.object(
                checker.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(
                    [], 0, "No broken requirements found.", ""
                ),
            ),
            patch.object(checker.importlib.metadata, "distributions", return_value=[]),
        ):
            self.assertEqual(checker.check_dependencies()["torch_cuda"], "13.0")

    def test_preflight_rejects_broken_dependencies(self):
        import types

        torch = types.SimpleNamespace(
            __version__="2.13.0+cu130", version=types.SimpleNamespace(cuda="13.0")
        )
        checker = modules["serving_preflight"]
        with (
            patch.dict(sys.modules, {"torch": torch}),
            patch.object(
                checker.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(
                    [], 1, "conflicting requirements", ""
                ),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "pip check"):
                checker.check_dependencies()

    def test_preflight_rejects_wrong_torch_before_dependency_process(self):
        import types

        torch = types.SimpleNamespace(
            __version__="2.12.1+cu130", version=types.SimpleNamespace(cuda="13.0")
        )
        checker = modules["serving_preflight"]
        with (
            patch.dict(sys.modules, {"torch": torch}),
            patch.object(checker.subprocess, "run") as run,
        ):
            with self.assertRaisesRegex(ValueError, "exact Torch"):
                checker.check_dependencies()
            run.assert_not_called()

    def test_all_help_commands_work_without_cuda_or_nvml(self):
        for name in NAMES:
            if name in ("runtime", "protocol"):
                continue
            with self.subTest(name=name):
                completed = subprocess.run(
                    [sys.executable, str(PACKET / f"{name}.py"), "--help"],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_plan_has_no_cuda_execution(self):
        for session, expected in (("A", 42336), ("B", 62720)):
            result = subprocess.run(
                [
                    sys.executable,
                    str(PACKET / "timing.py"),
                    "--session",
                    session,
                    "--plan",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(json.loads(result.stdout)["replays"], expected)
        result = subprocess.run(
            [sys.executable, str(PACKET / "correctness.py"), "--plan"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(
            json.loads(result.stdout), {"cases_per_arm": 88, "arm_cases": 176}
        )

    def test_correctness_matrix_is_unique_and_complete(self):
        cases = correctness.cases()
        self.assertEqual(len(cases), len(set(cases)))
        self.assertEqual(sum(c[2:] == ("bf16", "eager") for c in cases), 64)
        self.assertEqual(sum(c[2:] == ("bf16", "graph") for c in cases), 8)
        self.assertEqual(sum(c[2] == "fp16" for c in cases), 16)

    def test_plan_calibrates_before_comparisons(self):
        a, ap = timing.plan("A")
        b, bp = timing.plan("B")
        self.assertEqual((len(a), len(ap), len(b), len(bp)), (92, 16, 80, 80))
        self.assertEqual([p[0] for p in ap[:4]], ["boundary"] * 4)
        self.assertTrue(all(p[0] == "variant" for p in bp))

    def test_clock_gate_rejects_bad_samples(self):
        sample = {"sm_mhz": 1000, "reasons": 4, "limit_mw": 250000}
        self.assertTrue(runtime.clock_valid([sample], 1000, 250000))
        for mutation in ({"reasons": 8}, {"sm_mhz": 899}, {"limit_mw": 240000}):
            self.assertFalse(
                runtime.clock_valid([{**sample, **mutation}], 1000, 250000)
            )
        self.assertFalse(runtime.clock_valid([], 1000, 250000))

    def test_interval_selection_excludes_idle_samples(self):
        samples = [{"time": i} for i in range(10)]
        selected = timing.interval_samples(samples, list(range(10)), [(2, 3), (6, 7)])
        self.assertEqual([s["time"] for s in selected], [2, 3, 6, 7])

    def test_scoring_preserves_rejected_blocks(self):
        rows = [
            {"values": [[12.0] * 18, [10.0] * 18], "intervals": [[0, 1]]}
            for _ in range(7)
        ]
        samples = [{"time": 0.5, "sm_mhz": 1000, "reasons": 0, "limit_mw": 250000}]
        diff, refs = timing.score_blocks(rows, samples, [0.5], 1000, 250000)
        self.assertEqual(diff, [2.0] * 7)
        self.assertEqual(refs, [10.0] * 7)
        diff, refs = timing.score_blocks(rows, [], [], 1000, 250000)
        self.assertEqual(diff, [None] * 7)
        self.assertEqual(refs, [])

    def test_score_requires_valid_calibration(self):
        def record(kind, right):
            return {
                "kind": kind,
                "shape": [4096, 4096],
                "left_m": 64,
                "right_m": 64,
                "left_cache": "rotating",
                "right_cache": "rotating",
                "blocks": [
                    {"values": [[10.0] * 18, [right] * 18], "intervals": [[0, 1]]}
                    for _ in range(7)
                ],
            }

        samples = [{"time": 0.5, "sm_mhz": 1000, "reasons": 4, "limit_mw": 250000}]
        rows = [record("AA", 10), record("variant", 8)]
        timing.score(rows, samples)
        self.assertEqual(rows[1]["status"], "positive")
        rows = [record("variant", 8)]
        timing.score(rows, samples)
        self.assertEqual(rows[0]["status"], "unscored")

    def test_union_does_not_double_count_overlapping_kernels(self):
        self.assertEqual(witness.union_duration([(1, 5), (3, 7), (8, 10)]), 8)
        with self.assertRaises(ValueError):
            witness.union_duration([(5, 1)])

    def test_capture_shapes_need_matching_capture_range(self):
        events = [
            {"name": "capture_64_FULL", "pid": 1, "tid": 2, "ts": 0, "dur": 20},
            {
                "name": "_C::cutlass_scaled_mm",
                "pid": 1,
                "tid": 2,
                "ts": 1,
                "dur": 2,
                "args": {"Input Dims": [[64, 4096], [64, 4096], [4096, 4096]]},
            },
        ]
        witness.validate_capture(events, 1, 64, 4096, 4096)
        with self.assertRaises(ValueError):
            witness.validate_capture(events, 1, 63, 4096, 4096)
        events[0]["name"] = "warmup"
        with self.assertRaises(ValueError):
            witness.validate_capture(events, 1, 64, 4096, 4096)

    def test_witness_cannot_be_names_or_concurrency_only(self):
        for reviewer, basis in (
            ("AI", "capture_op_metadata_and_reviewed_replay_topology"),
            ("human", "kernel_name"),
        ):
            with self.assertRaises(ValueError):
                witness.inspect_review(
                    {"status": "collected_review_pending"},
                    {"reviewer": reviewer, "mapping_basis": basis},
                )

    def test_dispatch_needs_three_distinct_path_witnesses(self):
        rows = [{"arm": "base", "name": "Cooperative"} for _ in range(3)] + [
            {"arm": "variant", "name": "Pingpong"} for _ in range(3)
        ]
        trace.validate_dispatch(rows, ["base", "variant"])
        with self.assertRaises(ValueError):
            trace.validate_dispatch(rows[:-1], ["base", "variant"])
        rows[-1]["name"] = "Cooperative"
        with self.assertRaises(ValueError):
            trace.validate_dispatch(rows, ["base", "variant"])

    def test_sqlite_parser_attributes_real_api_rows_not_names_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.sqlite"
            with closing(sqlite3.connect(path)) as db, db:
                db.executescript(
                    "CREATE TABLE StringIds(id INTEGER,value TEXT);"
                    "CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,"
                    "globalTid INTEGER,text TEXT,textId INTEGER);"
                    "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(globalPid INTEGER,"
                    "demangledName INTEGER,correlationId INTEGER,gridX INTEGER,"
                    "gridY INTEGER,gridZ INTEGER,blockX INTEGER,blockY INTEGER,"
                    "blockZ INTEGER); CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER("
                    "start INTEGER,end INTEGER,globalTid INTEGER,"
                    "correlationId INTEGER);"
                )
                db.execute(
                    "INSERT INTO StringIds VALUES(1,"
                    "'cutlass_3x_gemm_fp8_blockwise<Cooperative>')"
                )
                for i in range(3):
                    db.execute(
                        "INSERT INTO NVTX_EVENTS "
                        "VALUES(?,?,42,'labperf:base:m64',NULL)",
                        (i * 100, i * 100 + 90),
                    )
                    db.execute(
                        "INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER VALUES(?,?,42,?)",
                        (i * 100 + 10, i * 100 + 20, i),
                    )
                    db.execute(
                        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL "
                        "VALUES(1,1,?,32,1,1,128,1,1)",
                        (i,),
                    )
            rows = trace.dispatch_rows(path)
            self.assertEqual(len(rows), 3)
            trace.validate_dispatch(rows, ["base"])
            with closing(sqlite3.connect(path)) as db, db:
                db.execute("DELETE FROM NVTX_EVENTS")
            self.assertEqual(trace.dispatch_rows(path), [])

    def test_receipt_pair_only_allows_dispatch_difference(self):
        base = {
            "namespace": "base",
            "cutlass": {},
            "cmake_reference": {},
            "harness_sha256": {},
            "public_freeze": {},
            "source": {"csrc_sha256": {"x": "1", modules["build_perf"].DISPATCH: "a"}},
            "binary": {"torch": "2.13.0"},
        }
        variant = {
            **base,
            "namespace": "variant",
            "source": {"csrc_sha256": {"x": "1", modules["build_perf"].DISPATCH: "b"}},
        }
        runtime.verify_pair(base, variant)
        variant["source"]["csrc_sha256"]["x"] = "changed"
        with self.assertRaises(ValueError):
            runtime.verify_pair(base, variant)

    def test_reused_output_is_refused(self):
        with tempfile.TemporaryDirectory() as path, self.assertRaises(FileExistsError):
            runtime.fresh(path)

    def test_collector_keeps_native_graphs_and_separates_cupti_passes(self):
        collect = modules["serve_trace"]
        nsys = collect.command("python", "model", "/private/work", "nsys")
        shapes = collect.command("python", "model", "/private/work", "shapes")
        self.assertEqual(nsys[0], "nsys")
        self.assertEqual(shapes[0], "python")
        self.assertIn("--cuda-graph-trace=node", nsys)
        for command in (nsys, shapes):
            self.assertNotIn("--enforce-eager", command)
            self.assertEqual(
                command[command.index("--performance-mode") + 1], "balanced"
            )
        settings = json.loads(shapes[shapes.index("--profiler-config") + 1])
        self.assertTrue(settings["capture_torch_profiler"])
        self.assertTrue(settings["torch_profiler_record_shapes"])

    def test_manifest_has_all_tools_and_is_not_public_freeze(self):
        entries = modules["freeze_packet"].entries()
        self.assertTrue(any(p.endswith("timing.py") for p in entries))
        self.assertTrue(any(p.endswith("witness_review.py") for p in entries))
        self.assertFalse(any(p.endswith("FREEZE_MANIFEST.json") for p in entries))

    def test_public_summary_removes_private_identity_and_handles_preflight_failure(
        self,
    ):
        summary = modules["summarize"]
        data = {
            "status": "unscored",
            "gpu": {"name": "H800", "uuid": "private-uuid"},
            "private_path": "secret",
            "failure": "private stack and path",
        }
        result = summary.summary(data)
        self.assertEqual(result["gpu"], {"name": "H800"})
        self.assertNotIn("secret", json.dumps(result))
        self.assertNotIn("private", json.dumps(result))
        self.assertEqual(summary.summary({"status": "unscored"})["gpu"], {})

    def test_dispatch_patch_has_only_one_predicate_change(self):
        patch = (PACKET / "dispatch_variant.patch").read_text()
        self.assertEqual(
            [
                s
                for s in patch.splitlines()
                if s.startswith("+") and not s.startswith("+++")
            ],
            ["+  bool swap_ab = a.size(0) <= 64 || (a.size(0) % 4) != 0;"],
        )

    def test_cross_session_uuid_change_requires_all_other_identity_fields(self):
        fields = (
            "name",
            "sm_count",
            "l2_bytes",
            "torch",
            "torch_cuda",
            "cuda_driver_version",
            "nvcc_version",
            "nvidia_driver",
        )
        left = {key: "same" for key in fields} | {"uuid": "card-A"}
        right = left | {"uuid": "card-B"}
        self.assertTrue(runtime.compatible_gpu(left, right))
        for key in fields:
            with self.subTest(field=key), self.assertRaises(ValueError):
                runtime.compatible_gpu(left, right | {key: "different"})
        with self.assertRaises(ValueError):
            runtime.compatible_gpu(left, {"uuid": "card-B"})

    def test_parent_wheel_provenance_verifies_metadata_and_actual_extension_bytes(self):
        collector = modules["serve_trace"]
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "parent.whl"
            log = Path(directory) / "install.log"
            log.write_text("recorded install")
            index = Path(directory) / "metadata.json"
            index.write_text(
                json.dumps(
                    [
                        {
                            "package_name": "vllm",
                            "version": "0.0+g4f14516",
                            "platform_tag": "manylinux_2_28_x86_64",
                            "python_tag": "cp38",
                            "abi_tag": "abi3",
                            "variant": None,
                            "filename": "test.whl",
                            "path": f"../../../{collector.PARENT}/test.whl",
                        }
                    ]
                )
            )
            binary = b"synthetic test bytes, not a working ELF"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("vllm/_C.so", binary)
                archive.writestr(
                    "vllm.dist-info/METADATA",
                    "Name: vllm\nVersion: 0.0+g4f14516\n"
                    "Requires-Dist: torch (==2.13.0)\n",
                )
            installed = {
                "extensions_sha256": {
                    "_C.so": __import__("hashlib").sha256(binary).hexdigest()
                }
            }
            provenance = {
                "kind": "precompiled_parent",
                "source_pin": modules["build_perf"].PIN,
                "extensions_sha256": installed["extensions_sha256"],
                "build_command": "recorded explicit parent install",
                "build_log": str(log),
                "build_log_sha256": runtime.sha(log),
                "wheel_path": str(wheel),
                "wheel_sha256": runtime.sha(wheel),
                "wheel_commit": collector.PARENT,
                "wheel_url": f"https://wheels.vllm.ai/{collector.PARENT}/test.whl",
                "wheel_index_url": f"https://wheels.vllm.ai/{collector.PARENT}/cu130/vllm/metadata.json",
                "wheel_index_path": str(index),
                "wheel_index_sha256": runtime.sha(index),
            }
            with (
                patch.object(
                    modules["build_perf"],
                    "git",
                    side_effect=[
                        collector.PARENT,
                        modules["build_perf"].ROOT + "scaled_mm_entry.cu",
                    ],
                ),
                # Python 3.10 has no file_digest; provenance must not depend on it.
                patch.object(collector.hashlib, "file_digest", None, create=True),
            ):
                self.assertEqual(
                    collector.verify_provenance(provenance, installed, Path(directory)),
                    "precompiled_parent",
                )
            with self.assertRaises(ValueError):
                collector.verify_provenance(
                    provenance | {"wheel_commit": "b" * 40}, installed, Path(directory)
                )
            mismatched = {"extensions_sha256": {"_C.so": "wrong"}}
            with self.assertRaises(ValueError):
                collector.verify_provenance(provenance, mismatched, Path(directory))
            with self.assertRaises(ValueError):
                collector.verify_provenance(
                    provenance
                    | {"wheel_url": "https://wheels.vllm.ai/latest/test.whl"},
                    installed,
                    Path(directory),
                )
            with (
                patch.object(
                    modules["build_perf"],
                    "git",
                    side_effect=[
                        collector.PARENT,
                        modules["build_perf"].ROOT + "scaled_mm_entry.cu",
                    ],
                ),
                self.assertRaises(ValueError),
            ):
                collector.verify_provenance(
                    provenance | {"extensions_sha256": mismatched["extensions_sha256"]},
                    mismatched,
                    Path(directory),
                )

    def test_collector_plan_is_one_selected_configuration(self):
        result = subprocess.run(
            [
                sys.executable,
                str(PACKET / "serve_trace.py"),
                "--model-dir",
                "PRIVATE_MODEL",
                "--vllm-src",
                "PRIVATE_SRC",
                "--configuration",
                "forced",
                "--mode",
                "shapes",
                "--plan",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(json.loads(result.stdout)["configurations"], ["forced"])

    def test_archive_preflight_checks_torch_and_platform_without_gpu(self):
        helper = modules["wheel_preflight"]
        collector = modules["serve_trace"]
        with tempfile.TemporaryDirectory() as directory:
            index, wheel = (
                Path(directory) / "metadata.json",
                Path(directory) / "test.whl",
            )
            row = {
                "package_name": "vllm",
                "version": "0.0+g4f14516",
                "platform_tag": "manylinux_2_28_x86_64",
                "python_tag": "cp38",
                "abi_tag": "abi3",
                "filename": "test.whl",
                "path": f"../../../{collector.PARENT}/test.whl",
            }
            index.write_text(json.dumps([row]))

            def archive(torch_version, tag):
                with zipfile.ZipFile(wheel, "w") as file:
                    file.writestr(
                        "vllm.dist-info/METADATA",
                        "Name: vllm\nVersion: 0.0+g4f14516\n"
                        f"Requires-Dist: torch=={torch_version}\n",
                    )
                    file.writestr(
                        "vllm.dist-info/WHEEL", f"Wheel-Version: 1.0\nTag: {tag}\n"
                    )
                    for name in ("vllm/_C_stable_libtorch.abi3.so",):
                        file.writestr(name, b"synthetic test only")

            archive("2.13.0", "cp38-abi3-manylinux_2_28_x86_64")
            result = helper.inspect_archive(index, wheel)
            self.assertEqual(
                result["wheel_url"],
                f"https://wheels.vllm.ai/{collector.PARENT}/test.whl",
            )
            self.assertIn("runtime unverified", result["limits"])
            archive("2.13.0", "cp38-abi3-linux_x86_64")
            generic = helper.inspect_archive(index, wheel)
            self.assertTrue(generic["generic_linux_internal_tag"])
            self.assertEqual(generic["index_platform_tag"], "manylinux_2_28_x86_64")
            for version, tag in (
                ("2.12.0", "cp38-abi3-manylinux_2_28_x86_64"),
                ("2.13.0", "cp38-abi3-manylinux_2_28_aarch64"),
            ):
                archive(version, tag)
                with (
                    self.subTest(version=version, tag=tag),
                    self.assertRaises(ValueError),
                ):
                    helper.inspect_archive(index, wheel)

    def test_wheel_index_rejects_duplicate_or_wrong_parent(self):
        collector = modules["serve_trace"]
        row = {
            "package_name": "vllm",
            "version": "0.0+g4f14516",
            "platform_tag": "manylinux_2_28_x86_64",
            "python_tag": "cp38",
            "abi_tag": "abi3",
            "filename": "test.whl",
            "path": f"../../../{collector.PARENT}/test.whl",
        }
        with self.assertRaises(ValueError):
            collector.select_wheel_index([row, row])
        with self.assertRaises(ValueError):
            collector.select_wheel_index(
                [row | {"path": "https://wheels.vllm.ai/latest/test.whl"}]
            )

    def test_archive_download_uses_small_reads_and_reports_without_gpu(self):
        helper = modules["wheel_preflight"]

        class Response(io.BytesIO):
            headers = {"Content-Length": "3"}

        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "test.whl"
            with (
                patch.object(
                    helper.urllib.request, "urlopen", return_value=Response(b"abc")
                ),
                redirect_stderr(output),
            ):
                helper.fetch("https://example.invalid/test", target, 8)
            self.assertEqual(target.read_bytes(), b"abc")
            self.assertIn("Response received", output.getvalue())
            self.assertIn("Downloaded 3 bytes", output.getvalue())
            with (
                patch.object(
                    helper.urllib.request, "urlopen", return_value=Response(b"abc")
                ),
                redirect_stderr(output),
                self.assertRaises(FileExistsError),
            ):
                helper.fetch("https://example.invalid/test", target, 8)

    def test_public_freeze_requires_matching_public_bytes(self):
        raw = json.dumps({"files_sha256": {}}).encode()
        freeze = modules["freeze_packet"]
        with (
            patch.dict(sys.modules, {"freeze_packet": freeze}),
            patch.object(freeze, "entries", return_value={}),
            patch.object(runtime, "Path") as path,
            patch.object(runtime, "sha", return_value="digest"),
            patch.object(runtime.urllib.request, "urlopen") as fetch,
        ):
            path.return_value.with_name.return_value.read_bytes.return_value = raw
            fetch.return_value = io.BytesIO(raw)
            self.assertEqual(
                runtime.public_freeze("a" * 40)["manifest_sha256"], "digest"
            )
            fetch.return_value = io.BytesIO(b"different")
            with self.assertRaises(ValueError):
                runtime.public_freeze("a" * 40)
            fetch.reset_mock()
            with self.assertRaises(ValueError):
                runtime.public_freeze("short")
            fetch.assert_not_called()

    def test_cpu_tensor_layouts_and_weights_are_m_independent(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional CPU Torch unavailable")
        first = runtime.tensors(torch, 63, 256, 256, torch.bfloat16, device="cpu")
        second = runtime.tensors(torch, 64, 256, 256, torch.bfloat16, device="cpu")
        self.assertTrue(torch.equal(first[2].float(), second[2].float()))
        self.assertTrue(torch.equal(first[4], second[4]))
        self.assertEqual(first[3].stride(), (1, 63))
        self.assertEqual(first[4].stride(), (1, 2))


if __name__ == "__main__":
    unittest.main()
