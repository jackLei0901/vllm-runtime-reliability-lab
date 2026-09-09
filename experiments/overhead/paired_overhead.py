from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALLOWED_PLAN_KEYS = {
    "schema_version",
    "experiment_id",
    "environment_class",
    "sequence",
    "variables",
    "commands",
    "ready_url",
    "startup_timeout_seconds",
    "workload_timeout_seconds",
    "shutdown_timeout_seconds",
    "recorder_settle_seconds",
    "metadata",
}
REQUIRED_COMMANDS = {"server", "workload", "recorder"}


class PlanError(ValueError):
    pass


class TrialAbort(RuntimeError):
    pass


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def plan_digest(plan: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(plan).encode()).hexdigest()


def load_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise PlanError("plan must be a JSON object")
    unknown = set(plan) - ALLOWED_PLAN_KEYS
    if unknown:
        raise PlanError(f"unknown plan fields: {sorted(unknown)}")
    required = {
        "schema_version",
        "experiment_id",
        "environment_class",
        "sequence",
        "variables",
        "commands",
        "ready_url",
    }
    missing = required - set(plan)
    if missing:
        raise PlanError(f"missing plan fields: {sorted(missing)}")
    if plan["schema_version"] != 1:
        raise PlanError("only schema_version 1 is supported")
    sequence = plan["sequence"]
    if not isinstance(sequence, list) or len(sequence) < 4:
        raise PlanError("sequence must contain at least four arms")
    expected = ["disabled" if i % 2 == 0 else "enabled" for i in range(len(sequence))]
    if sequence != expected:
        raise PlanError("sequence must alternate disabled, enabled and start disabled")
    commands = plan["commands"]
    if not isinstance(commands, dict) or set(commands) != REQUIRED_COMMANDS:
        raise PlanError("commands must contain exactly server, workload and recorder")
    for name, command in commands.items():
        if not isinstance(command, list) or not command:
            raise PlanError(f"commands.{name} must be a non-empty list")
        if not all(isinstance(token, str) and token for token in command):
            raise PlanError(f"commands.{name} tokens must be non-empty strings")
    if not isinstance(plan["variables"], dict):
        raise PlanError("variables must be an object")
    return plan


def resolve_tokens(tokens: list[str], variables: dict[str, object]) -> list[str]:
    values = {key: str(value) for key, value in variables.items()}
    try:
        return [token.format_map(values) for token in tokens]
    except KeyError as exc:
        raise PlanError(f"undefined command variable: {exc.args[0]}") from exc


def wait_until_ready(
    url: str, process: subprocess.Popen[bytes], timeout: float
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=0.5) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(0.1)
    return False


def start_process(
    command: list[str], stdout: Path, stderr: Path
) -> subprocess.Popen[bytes]:
    creationflags = 0
    start_new_session = os.name != "nt"
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    with stdout.open("wb") as stdout_handle, stderr.open("wb") as stderr_handle:
        return subprocess.Popen(
            command,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=start_new_session,
            creationflags=creationflags,
        )


def process_group_alive(process: subprocess.Popen[bytes]) -> bool:
    if os.name == "nt":
        return process.poll() is None
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    return True


def stop_process(
    process: subprocess.Popen[bytes], timeout: float, *, graceful_console: bool = False
) -> tuple[int | None, str]:
    if process.poll() is not None:
        return process.returncode, "already_exited"
    method = "sigint"
    if os.name == "nt":
        if graceful_console:
            method = "ctrl_break"
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            method = "terminate"
            process.terminate()
    else:
        os.killpg(process.pid, signal.SIGINT)
    try:
        return process.wait(timeout=timeout), method
    except subprocess.TimeoutExpired:
        method += "_then_kill"
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
        return process.wait(timeout=timeout), method


@dataclass
class LinuxProcessUsage:
    pid: int
    sample_interval: float = 0.05

    def __post_init__(self) -> None:
        self.samples = 0
        self.peak_rss_bytes: int | None = None
        self.first_cpu_ticks: int | None = None
        self.last_cpu_ticks: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        if sys.platform.startswith("linux"):
            self._thread.start()

    def stop(self) -> dict[str, int | float | None]:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1)
        ticks = None
        if self.first_cpu_ticks is not None and self.last_cpu_ticks is not None:
            ticks = self.last_cpu_ticks - self.first_cpu_ticks
        clock_ticks = (
            os.sysconf("SC_CLK_TCK") if sys.platform.startswith("linux") else 1
        )
        return {
            "samples": self.samples,
            "peak_rss_bytes": self.peak_rss_bytes,
            "cpu_seconds": None if ticks is None else ticks / clock_ticks,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                stat_fields = Path(f"/proc/{self.pid}/stat").read_text().split()
                status = Path(f"/proc/{self.pid}/status").read_text().splitlines()
                cpu_ticks = int(stat_fields[13]) + int(stat_fields[14])
                rss_kib = next(
                    int(line.split()[1]) for line in status if line.startswith("VmRSS:")
                )
            except (FileNotFoundError, PermissionError, StopIteration, ValueError):
                break
            self.samples += 1
            self.first_cpu_ticks = (
                cpu_ticks if self.first_cpu_ticks is None else self.first_cpu_ticks
            )
            self.last_cpu_ticks = cpu_ticks
            rss_bytes = rss_kib * 1024
            self.peak_rss_bytes = max(self.peak_rss_bytes or 0, rss_bytes)
            self._stop.wait(self.sample_interval)


def parse_workload(stdout: str) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("workload produced no JSON")
    payload = json.loads(lines[-1])
    if not isinstance(payload, dict):
        raise ValueError("workload JSON must be an object")
    if not isinstance(payload.get("workload_signature"), str):
        raise ValueError("workload JSON needs workload_signature")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("workload JSON needs a metrics object")
    for key, value in metrics.items():
        if not isinstance(key, str) or not isinstance(value, (int, float)):
            raise ValueError("workload metrics must be numeric")
    return payload


def run_trial(
    plan: dict[str, Any], arm: str, index: int, output_dir: Path
) -> dict[str, Any]:
    trial_dir = output_dir / f"trial-{index:02d}-{arm}"
    trial_dir.mkdir(parents=True, exist_ok=False)
    variables = dict(plan["variables"])
    variables.update(
        {
            "python": sys.executable,
            "trial_dir": str(trial_dir.resolve()),
            "arm": arm,
            "trial_index": index,
        }
    )
    timeout_start = float(plan.get("startup_timeout_seconds", 120))
    timeout_workload = float(plan.get("workload_timeout_seconds", 600))
    timeout_shutdown = float(plan.get("shutdown_timeout_seconds", 30))
    settle = float(plan.get("recorder_settle_seconds", 0.25))

    server_command = resolve_tokens(plan["commands"]["server"], variables)
    server = start_process(
        server_command, trial_dir / "server.stdout.log", trial_dir / "server.stderr.log"
    )
    variables["server_pid"] = server.pid
    recorder: subprocess.Popen[bytes] | None = None
    usage: LinuxProcessUsage | None = None
    result: dict[str, Any] = {
        "index": index,
        "arm": arm,
        "server_pid": server.pid,
        "server_ready": False,
        "workload_returncode": None,
        "recorder_returncode": None,
        "server_returncode": None,
        "workload": None,
        "errors": [],
    }
    try:
        ready_url = str(plan["ready_url"]).format_map(
            {key: str(value) for key, value in variables.items()}
        )
        result["server_ready"] = wait_until_ready(ready_url, server, timeout_start)
        if not result["server_ready"]:
            result["errors"].append("server_not_ready")
            raise TrialAbort
        if arm == "enabled":
            recorder_command = resolve_tokens(plan["commands"]["recorder"], variables)
            recorder = start_process(
                recorder_command,
                trial_dir / "recorder.stdout.log",
                trial_dir / "recorder.stderr.log",
            )
            usage = LinuxProcessUsage(recorder.pid)
            usage.start()
            time.sleep(settle)
            if recorder.poll() is not None:
                result["errors"].append("recorder_exited_before_workload")

        workload_command = resolve_tokens(plan["commands"]["workload"], variables)
        completed = subprocess.run(
            workload_command,
            capture_output=True,
            timeout=timeout_workload,
            check=False,
            text=True,
            encoding="utf-8",
        )
        (trial_dir / "workload.stdout.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        (trial_dir / "workload.stderr.log").write_text(
            completed.stderr, encoding="utf-8"
        )
        result["workload_returncode"] = completed.returncode
        if completed.returncode != 0:
            result["errors"].append("workload_failed")
        else:
            try:
                result["workload"] = parse_workload(completed.stdout)
            except (ValueError, json.JSONDecodeError) as exc:
                result["errors"].append(f"invalid_workload_output:{exc}")
    except TrialAbort:
        pass
    except subprocess.TimeoutExpired:
        result["errors"].append("workload_timeout")
    finally:
        if recorder is not None:
            result["recorder_running_before_stop"] = recorder.poll() is None
            if not result["recorder_running_before_stop"]:
                result["errors"].append("recorder_exited_during_trial")
            result["recorder_returncode"], result["recorder_stop_method"] = (
                stop_process(recorder, timeout_shutdown, graceful_console=True)
            )
            summary_path = trial_dir / "artifacts" / "run-summary.private.json"
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                result["collector_timing"] = summary.get("collector_timing")
            else:
                result["errors"].append("recorder_summary_missing")
            result["recorder_usage"] = usage.stop() if usage is not None else None
            result["recorder_group_alive_after_stop"] = process_group_alive(recorder)
            if result["recorder_group_alive_after_stop"]:
                result["errors"].append("recorder_group_still_alive")
        result["server_running_before_stop"] = server.poll() is None
        if result["server_ready"] and not result["server_running_before_stop"]:
            result["errors"].append("server_exited_during_trial")
        result["server_returncode"], result["server_stop_method"] = stop_process(
            server, timeout_shutdown
        )
        result["server_group_alive_after_stop"] = process_group_alive(server)
        if result["server_group_alive_after_stop"]:
            result["errors"].append("server_group_still_alive")
    result["valid"] = not result["errors"]
    return result


def validate_pairing(trials: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    valid = [trial for trial in trials if trial.get("valid")]
    signatures = {
        trial["workload"]["workload_signature"]
        for trial in valid
        if trial.get("workload")
    }
    if len(valid) != len(trials):
        errors.append("one_or_more_trials_invalid")
    if len(signatures) != 1:
        errors.append("workload_signatures_do_not_match")
    arms = [trial["arm"] for trial in valid]
    if arms.count("disabled") != arms.count("enabled"):
        errors.append("valid_arm_counts_do_not_match")
    return errors


def aggregate_metrics(trials: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, dict[str, list[float]]] = {"disabled": {}, "enabled": {}}
    for trial in trials:
        if not trial.get("valid") or not trial.get("workload"):
            continue
        for key, value in trial["workload"]["metrics"].items():
            by_arm[trial["arm"]].setdefault(key, []).append(float(value))
    summary: dict[str, Any] = {}
    for arm, metrics in by_arm.items():
        summary[arm] = {
            key: {"values": values, "median": statistics.median(values)}
            for key, values in metrics.items()
        }
    return summary


def compare_paired_metrics(trials: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons: dict[str, list[float]] = {}
    for disabled, enabled in zip(trials[::2], trials[1::2], strict=True):
        if not disabled.get("valid") or not enabled.get("valid"):
            continue
        disabled_metrics = disabled["workload"]["metrics"]
        enabled_metrics = enabled["workload"]["metrics"]
        for key in disabled_metrics.keys() & enabled_metrics.keys():
            baseline = float(disabled_metrics[key])
            if baseline == 0:
                continue
            delta_pct = (float(enabled_metrics[key]) - baseline) / baseline * 100
            comparisons.setdefault(key, []).append(delta_pct)
    return {
        key: {
            "relative_delta_pct_by_pair": values,
            "median_relative_delta_pct": statistics.median(values),
            "pair_count": len(values),
        }
        for key, values in sorted(comparisons.items())
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        f"# Paired overhead run: {result['experiment_id']}",
        "",
        f"- Environment class: `{result['environment_class']}`",
        f"- Plan SHA-256: `{result['plan_sha256']}`",
        f"- Pairing valid: `{result['pairing_valid']}`",
        "- This report describes harness output; it does not by itself establish "
        "production overhead.",
        "- Return codes from harness-initiated termination are platform-dependent; "
        "validity requires processes to be alive before stop and absent afterward.",
        "",
        "| Trial | Arm | Ready | Workload rc | Recorder rc | Server rc | Valid |",
        "| ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for trial in result["trials"]:
        lines.append(
            "| {index} | {arm} | {server_ready} | {workload_returncode} | "
            "{recorder_returncode} | {server_returncode} | {valid} |".format(**trial)
        )
    if result["pairing_errors"]:
        lines.extend(["", "Pairing errors:"])
        lines.extend(f"- `{error}`" for error in result["pairing_errors"])
    if result["paired_comparisons"]:
        lines.extend(
            [
                "",
                "| Metric | Pairs | Median enabled-vs-disabled delta |",
                "| --- | ---: | ---: |",
            ]
        )
        for metric, comparison in result["paired_comparisons"].items():
            lines.append(
                f"| `{metric}` | {comparison['pair_count']} | "
                f"{comparison['median_relative_delta_pct']:+.3f}% |"
            )
        lines.extend(
            [
                "",
                "Signed deltas are descriptive: lower is better for latency, while "
                "higher is better for throughput.",
            ]
        )
    lines.extend(
        [
            "",
            "Numeric workload metrics are retained in `paired-overhead.private.json`. ",
            "Review that private file before publishing derived statistics.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    args.output.mkdir(parents=True, exist_ok=False)
    trials = [
        run_trial(plan, arm, index, args.output)
        for index, arm in enumerate(plan["sequence"], start=1)
    ]
    pairing_errors = validate_pairing(trials)
    result = {
        "schema_version": 1,
        "experiment_id": plan["experiment_id"],
        "environment_class": plan["environment_class"],
        "plan_sha256": plan_digest(plan),
        "metadata": plan.get("metadata", {}),
        "trials": trials,
        "pairing_valid": not pairing_errors,
        "pairing_errors": pairing_errors,
        "aggregates": aggregate_metrics(trials),
        "paired_comparisons": compare_paired_metrics(trials),
    }
    (args.output / "paired-overhead.private.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output / "REPORT.md").write_text(render_markdown(result), encoding="utf-8")
    return 0 if result["pairing_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
