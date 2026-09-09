import ast
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENT_PATTERN = re.compile(r"\b(?:FR|SR)-\d{3}\b")


def discovered_test_ids() -> set[str]:
    test_ids: set[str] = set()
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        module = path.stem
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if child.name.startswith("test_"):
                        test_ids.add(f"{module}.{node.name}.{child.name}")
    return test_ids


class TestPlanTraceabilityTest(unittest.TestCase):
    def test_every_requirement_maps_to_an_existing_test(self) -> None:
        requirements_text = (ROOT / "REQUIREMENTS.md").read_text(encoding="utf-8")
        requirement_ids = set(REQUIREMENT_PATTERN.findall(requirements_text))
        cases = json.loads((ROOT / "tests" / "cases.json").read_text(encoding="utf-8"))

        self.assertEqual(set(cases), requirement_ids)
        known_tests = discovered_test_ids()
        for requirement_id, test_ids in cases.items():
            self.assertTrue(test_ids, requirement_id)
            for test_id in test_ids:
                self.assertIn(test_id, known_tests, f"{requirement_id}: {test_id}")


if __name__ == "__main__":
    unittest.main()
