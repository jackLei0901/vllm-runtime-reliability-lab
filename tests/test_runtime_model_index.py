"""CPU-only consistency checks for the Lab's documentation index."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "docs" / "model" / "RUNTIME_MODEL.md"
EXPERIMENTS = ROOT / "experiments"
REQUIRED = {"model_cells", "status", "upstream_exit", "last_scored"}
STATUSES = {"active", "paused", "closed", "archived"}
ARCHIVED_STUBS = {"oom-boundary", "preemption", "soak"}


def parse_status_header(readme: Path) -> dict[str, str]:
    text = readme.read_text(encoding="utf-8")
    match = re.match(r"\A---\r?\n(.*?)\r?\n---\r?\n", text, re.DOTALL)
    if match is None:
        raise ValueError(f"missing status header: {readme}")
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, separator, value = line.partition(":")
        if not separator or key not in REQUIRED or key in fields or not value.strip():
            raise ValueError(f"invalid status header line: {readme}: {line}")
        fields[key] = value.strip()
    if set(fields) != REQUIRED:
        raise ValueError(f"incomplete status header: {readme}")
    return fields


def parse_model_cells(value: str, valid_ids: set[str]) -> list[str]:
    if not re.fullmatch(r"\[(?:M\d+(?:, M\d+)*)?\]", value):
        raise ValueError(f"invalid model_cells syntax: {value}")
    ids = [] if value == "[]" else value[1:-1].split(", ")
    if len(ids) != len(set(ids)) or any(
        identifier not in valid_ids for identifier in ids
    ):
        raise ValueError(f"unknown or duplicate model ID: {value}")
    return ids


class RuntimeModelIndexTests(unittest.TestCase):
    def test_every_experiment_has_valid_model_status(self) -> None:
        valid_ids = set(
            re.findall(r"^## (M\d+) —", MODEL.read_text(encoding="utf-8"), re.MULTILINE)
        )
        self.assertEqual(valid_ids, {f"M{i}" for i in range(1, 7)})
        directories = [path for path in EXPERIMENTS.iterdir() if path.is_dir()]
        self.assertTrue(directories)
        for directory in directories:
            with self.subTest(experiment=directory.name):
                fields = parse_status_header(directory / "README.md")
                parse_model_cells(fields["model_cells"], valid_ids)
                self.assertIn(fields["status"], STATUSES)

    def test_archived_stubs_do_not_claim_a_score_or_model_cell(self) -> None:
        for name in ARCHIVED_STUBS:
            with self.subTest(experiment=name):
                fields = parse_status_header(EXPERIMENTS / name / "README.md")
                self.assertEqual(fields["status"], "archived")
                self.assertEqual(fields["model_cells"], "[]")
                self.assertEqual(fields["last_scored"], "never")

    def test_unknown_model_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown"):
            parse_model_cells("[M7]", {"M1", "M2"})

    def test_bilingual_index_links_resolve(self) -> None:
        for index in (
            ROOT / "docs" / "INDEX.md",
            ROOT / "docs" / "INDEX.zh-CN.md",
        ):
            with self.subTest(index=index.name):
                text = index.read_text(encoding="utf-8")
                for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
                    if "://" in target or target.startswith("#"):
                        continue
                    self.assertTrue(
                        (index.parent / target.split("#", 1)[0]).exists(), target
                    )


if __name__ == "__main__":
    unittest.main()
