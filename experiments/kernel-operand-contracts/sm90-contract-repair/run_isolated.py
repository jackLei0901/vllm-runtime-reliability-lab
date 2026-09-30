"""Collect the complete matrix, then run each case once in a fresh process."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

EXPECTED_CASES = 30
EXPECTED_BASE_SKIPS = 3
PACKET = Path(__file__).resolve().parent
TEST = PACKET / "test_repair_sm90.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_receipt(path: Path, arm: str) -> dict:
    receipt = json.loads(path.read_text())
    if receipt["status"] != "built_not_validated":
        raise ValueError("build is not eligible")
    if receipt["source"]["arm"] != arm:
        raise ValueError("arm mismatch")
    if receipt["source"]["head"] != "7b054aca96cea8be1369d651c3434ad140580b92":
        raise ValueError("source pin mismatch")
    for name, expected in receipt["harness_sha256"].items():
        if sha256(PACKET / name) != expected:
            raise ValueError(f"harness changed: {name}")
    if not {"test_repair_sm90.py", "run_isolated.py", "pytest.ini"}.issubset(
        receipt["harness_sha256"]
    ):
        raise ValueError("receipt does not bind tests and isolation runner")
    name = receipt["binary"]["binary_file"]
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("binary_file must be a basename")
    if sha256(path.resolve().parent / name) != receipt["binary"]["binary_sha256"]:
        raise ValueError("binary changed")
    return receipt


def parse_collection(text: str) -> list[str]:
    nodes = [
        line.strip() for line in text.splitlines() if "test_repair_sm90.py::" in line
    ]
    if len(nodes) != EXPECTED_CASES or len(set(nodes)) != len(nodes):
        raise ValueError("collection count or uniqueness mismatch")
    return nodes


def classify_xml(path: Path, returncode: int, node: str, arm: str) -> str:
    if not path.is_file():
        return "unscored"
    cases = list(ET.parse(path).getroot().iter("testcase"))
    if len(cases) != 1:
        return "unscored"
    case = cases[0]
    if case.find("error") is not None:
        return "unscored"
    properties = {p.attrib["name"]: p.attrib["value"] for p in case.iter("property")}
    if properties.get("cuda_context_usable_after") != "True":
        return "unscored"
    if case.find("failure") is not None:
        return "prediction_missed"
    if case.find("skipped") is not None:
        if (
            arm == "base"
            and "::test_fix_rejects_during_capture[" in node
            and case.find("skipped").attrib.get("message")
            == "base unsafe inputs are measured eagerly, not captured"
        ):
            return "planned_skip"
        return "unscored"
    if returncode != 0:
        return "unscored"
    return "prediction_matched"


def pytest_command(packet: Path = PACKET) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:randomly",
        "-c",
        str(packet.resolve() / "pytest.ini"),
        f"--rootdir={packet.resolve()}",
    ]


def run_once(
    command: list[str], env: dict, log: Path, timeout: float, cwd: Path = PACKET
) -> int:
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=cwd,
        )
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if sys.platform == "linux":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()
            return -1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["base", "fix"], required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=int, default=420)
    args = parser.parse_args()
    if not 1 <= args.budget_seconds <= 600:
        parser.error("arm test budget must be 1..600 seconds")
    verify_receipt(args.receipt, args.arm)
    args.out.mkdir(parents=True, exist_ok=False)
    arm_started = time.monotonic()
    deadline = arm_started + args.budget_seconds
    env = dict(
        os.environ,
        KOC_ARM=args.arm,
        KOC_BUILD_RECEIPT=str(args.receipt.resolve()),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTEST_ADDOPTS="",
    )
    env.pop("PYTEST_PLUGINS", None)
    command = pytest_command()
    events = args.out / "cases.jsonl"
    result = {
        "arm": args.arm,
        "status": "unscored",
        "cases": [],
        "test_sha256": sha256(TEST),
        "runner_sha256": sha256(Path(__file__)),
        "receipt_sha256": sha256(args.receipt),
    }
    try:
        collect_log = args.out / "collection.log"
        code = run_once(
            command + ["--collect-only", str(TEST)],
            env,
            collect_log,
            min(60, max(1, deadline - time.monotonic())),
        )
        if code != 0:
            raise ValueError("collection failed")
        nodes = parse_collection(collect_log.read_text())
        collection_elapsed = time.monotonic() - arm_started
        result["collection_elapsed_seconds"] = collection_elapsed
        result["collection"] = nodes
        for index, node in enumerate(nodes):
            entry = {"index": index, "node": node, "status": "unscored"}
            remaining = deadline - time.monotonic()
            if remaining > 0:
                verify_receipt(args.receipt, args.arm)
                xml = args.out / f"case-{index:02d}.xml"
                start = {"event": "start", "index": index, "node": node}
                with events.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(start) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                case_started = time.monotonic()
                code = run_once(
                    command
                    + [
                        "-o",
                        "junit_family=xunit1",
                        node,
                        f"--junitxml={xml.resolve()}",
                    ],
                    env,
                    args.out / f"case-{index:02d}.log",
                    min(45, remaining),
                )
                entry["elapsed_seconds"] = time.monotonic() - case_started
                try:
                    entry["status"] = classify_xml(xml, code, node, args.arm)
                except ET.ParseError:
                    entry["status"] = "unscored"
                entry["returncode"] = code
            else:
                entry["reason"] = "budget_exhausted_not_run"
            with events.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"event": "end", **entry}) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            result["cases"].append(entry)
            if index == 2:
                durations = [item.get("elapsed_seconds") for item in result["cases"]]
                if all(value is not None for value in durations):
                    elapsed = sum(durations)
                    result["first_three_elapsed_seconds"] = elapsed
                    result["projected_arm_seconds"] = (
                        collection_elapsed + elapsed * EXPECTED_CASES / 3
                    )
        statuses = [entry["status"] for entry in result["cases"]]
        expected_skips = EXPECTED_BASE_SKIPS if args.arm == "base" else 0
        if statuses.count("planned_skip") != expected_skips:
            result["status"] = "unscored"
        elif "unscored" in statuses:
            result["status"] = "unscored"
        elif "prediction_missed" in statuses:
            result["status"] = "prediction_missed"
        else:
            result["status"] = "all_predictions_matched"
    except ValueError as exc:
        result["failure"] = str(exc)
    finally:
        (args.out / "run_receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "arm": args.arm,
                "status": result["status"],
                "recorded_cases": len(result["cases"]),
            }
        )
    )
    if result["status"] != "all_predictions_matched":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
