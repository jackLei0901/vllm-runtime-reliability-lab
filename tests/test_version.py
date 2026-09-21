import re
import unittest
from pathlib import Path

import dfxlab

ROOT = Path(__file__).resolve().parents[1]


class VersionTest(unittest.TestCase):
    def test_package_and_project_versions_match(self) -> None:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)
        self.assertIsNotNone(match)
        self.assertEqual(dfxlab.__version__, match.group(1))


if __name__ == "__main__":
    unittest.main()
