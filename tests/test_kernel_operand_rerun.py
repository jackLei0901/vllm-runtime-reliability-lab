"""CPU tests for the B2/B0 single-rerun boundary and receipt ordering."""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

DIR = Path(__file__).resolve().parents[1] / "experiments" / "kernel-operand-contracts"
SPEC = importlib.util.spec_from_file_location("select_rerun", DIR / "select_rerun.py")
assert SPEC is not None and SPEC.loader is not None
sel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sel)

CASES = [f"t.py::c{i}" for i in range(6)]


def scored(case: str, usable: bool = True) -> list[dict]:
    return [
        {"case": case, "event": "start"},
        {"case": case, "event": "result", "cuda_usable_after": usable},
        {"case": case, "event": "end"},
    ]


def skipped(case: str) -> list[dict]:
    return [{"case": case, "event": "skipped_context_unusable"}]


class EligibleTests(unittest.TestCase):
    def test_complete_run_selects_nothing(self) -> None:
        receipts = [r for case in CASES for r in scored(case)]
        self.assertEqual(sel.eligible(CASES, receipts), [])

    def test_broken_context_reruns_only_the_skipped_cases(self) -> None:
        receipts = scored(CASES[0]) + scored(CASES[1], usable=False)
        for case in CASES[2:]:
            receipts += skipped(case)
        self.assertEqual(sel.eligible(CASES, receipts), CASES[2:])

    def test_hard_abort_reruns_only_unstarted_later_cases(self) -> None:
        receipts = scored(CASES[0]) + [{"case": CASES[1], "event": "start"}]
        self.assertEqual(sel.eligible(CASES, receipts), CASES[2:])

    def test_abort_during_setup_is_identified_and_not_rerun(self) -> None:
        # start is written before setup, so a setup-stage abort looks the same.
        receipts = [{"case": CASES[0], "event": "start"}]
        self.assertEqual(sel.eligible(CASES, receipts), CASES[1:])

    def test_abort_on_last_case_selects_nothing(self) -> None:
        receipts = [r for case in CASES[:-1] for r in scored(case)]
        receipts.append({"case": CASES[-1], "event": "start"})
        self.assertEqual(sel.eligible(CASES, receipts), [])

    def test_in_process_setup_failure_is_not_rerun(self) -> None:
        receipts = scored(CASES[0])
        receipts += [
            {"case": CASES[1], "event": "start"},
            {"case": CASES[1], "event": "end"},
        ]
        for case in CASES[2:]:
            receipts += scored(case)
        self.assertEqual(sel.eligible(CASES, receipts), [])
        self.assertEqual(sel.classify(CASES, receipts)[CASES[1]], "failed_no_result")


class FailClosedTests(unittest.TestCase):
    def assert_no_rerun(self, receipts: list[dict], collected=CASES) -> None:
        with self.assertRaises(sel.NoRerun):
            sel.eligible(collected, receipts)

    def test_unstarted_cases_without_an_abort(self) -> None:
        self.assert_no_rerun(scored(CASES[0]))

    def test_unstarted_case_before_the_abort(self) -> None:
        receipts = scored(CASES[0]) + [{"case": CASES[2], "event": "start"}]
        self.assert_no_rerun(receipts)

    def test_two_aborted_cases(self) -> None:
        receipts = [
            {"case": CASES[0], "event": "start"},
            {"case": CASES[1], "event": "start"},
        ]
        self.assert_no_rerun(receipts)

    def test_aborted_case_that_was_not_the_last_to_start(self) -> None:
        receipts = [{"case": CASES[0], "event": "start"}] + scored(CASES[1])
        self.assert_no_rerun(receipts)

    def test_skip_without_a_broken_context(self) -> None:
        self.assert_no_rerun(scored(CASES[0]) + skipped(CASES[1]))

    def test_skip_before_the_broken_context(self) -> None:
        receipts = skipped(CASES[0]) + scored(CASES[1], usable=False)
        self.assert_no_rerun(receipts)

    def test_receipt_for_an_uncollected_case(self) -> None:
        self.assert_no_rerun(scored("t.py::other"))

    def test_repeated_result(self) -> None:
        receipts = scored(CASES[0])
        receipts.append({"case": CASES[0], "event": "result"})
        self.assert_no_rerun(receipts)

    def test_result_without_start(self) -> None:
        self.assert_no_rerun([{"case": CASES[0], "event": "result"}])

    def test_unknown_event(self) -> None:
        self.assert_no_rerun([{"case": CASES[0], "event": "started"}])

    def test_unreadable_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.jsonl"
            path.write_text("{not json}\n")
            with self.assertRaises(sel.NoRerun):
                sel.read_receipts(path)


class ParsingTests(unittest.TestCase):
    def test_collect_output_ignores_summary_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "collected.txt"
            path.write_text(
                "t.py::c0[256-512-1024]\nt.py::c1\n\n2 tests collected in 0.1s\n"
            )
            self.assertEqual(
                sel.read_collected(path), ["t.py::c0[256-512-1024]", "t.py::c1"]
            )


class CommandLineTests(unittest.TestCase):
    def run_cli(self, receipts: list[dict]) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as tmp:
            collected = Path(tmp) / "collected.txt"
            collected.write_text("\n".join(CASES) + "\n")
            receipt_file = Path(tmp) / "receipts.jsonl"
            receipt_file.write_text("".join(json.dumps(r) + "\n" for r in receipts))
            return subprocess.run(
                [
                    sys.executable,
                    str(DIR / "select_rerun.py"),
                    str(collected),
                    str(receipt_file),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

    def test_refusal_exits_2_with_no_node_ids(self) -> None:
        done = self.run_cli(scored(CASES[0]))
        self.assertEqual(done.returncode, 2)
        self.assertEqual(done.stdout, "")
        self.assertIn("no rerun", done.stderr)

    def test_nothing_to_rerun_prints_nothing(self) -> None:
        done = self.run_cli([r for case in CASES for r in scored(case)])
        self.assertEqual(done.returncode, 0)
        self.assertEqual(done.stdout, "")


class ReceiptOrderingTests(unittest.TestCase):
    """The GPU test must write ``start`` before setup and ``end`` at teardown."""

    @classmethod
    def setUpClass(cls) -> None:
        source = (DIR / "test_blockwise_dtype_and_stride_sm90.py").read_text("utf-8")
        cls.functions = {
            node.name: node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef)
        }

    def events_in(self, name: str) -> list[str]:
        found = []
        for node in ast.walk(self.functions[name]):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values, strict=True):
                    if (
                        isinstance(key, ast.Constant)
                        and key.value == "event"
                        and isinstance(value, ast.Constant)
                    ):
                        found.append(value.value)
        return found

    def test_guard_writes_start_before_yield_and_end_after(self) -> None:
        guard = self.functions["_guard"]
        body = [ast.unparse(stmt) for stmt in guard.body]
        start = next(i for i, s in enumerate(body) if "'event': 'start'" in s)
        yielded = next(i for i, s in enumerate(body) if s == "yield")
        end = next(i for i, s in enumerate(body) if "'event': 'end'" in s)
        self.assertLess(start, yielded)
        self.assertLess(yielded, end)

    def test_score_writes_only_the_result(self) -> None:
        self.assertEqual(self.events_in("score"), ["result"])

    def test_every_case_calls_score(self) -> None:
        tests = [name for name in self.functions if name.startswith("test_")]
        self.assertEqual(len(tests), 6)
        for name in tests:
            calls = [
                node.func.id
                for node in ast.walk(self.functions[name])
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            ]
            self.assertIn("score", calls, name)


if __name__ == "__main__":
    unittest.main()
