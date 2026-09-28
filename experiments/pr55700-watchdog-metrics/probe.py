"""PR #55700: does the watchdog timeout counter move during a continuing hold?

Frozen record: docs/reviews/PR55700_FIRST_CANDIDATE_2026-09-28.zh-CN.md, with
addendum A1 (docs/reviews/PR55700_FIRST_CANDIDATE_ADDENDUM_A1_2026-09-28.zh-CN.md).
Source pin: PR head ``b274bf04dd4c6d54807a136babce5b5d17dd74be``.

The probe starts a real ``vllm serve`` (CPU backend) with the same interpreter
whose installation it verified. The only substitution is ``--worker-cls
hold_worker.HoldingCPUWorker``: the EngineCore busy loop, executor, watchdog,
scheduler stats, output transport and frontend Prometheus logger are unchanged
PR code. It holds one ``execute_model`` call, waits for a ``feed timeout``
witness, scrapes ``/metrics`` during the hold, releases it, and scrapes again.

``--tp 1`` (uniproc) scores the EngineCore question. ``--tp 2`` (multiproc)
holds the output rank (worker question) and then a non-output rank (secondary
rank question). Every apparatus failure is ``unscored``: unverified identity
(checked before launch), a failed control, a hold that was not entered or not
released, a failed request, an absent required series, or no post-release
export. A metric series that is absent is kept distinct from one at zero.

Linux only. ``result.json`` holds counts, relative times, configuration and
outcomes only; ``server.log`` and dump files stay private. The functions above
the runtime section import only the standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

PR_HEAD = "b274bf04dd4c6d54807a136babce5b5d17dd74be"
# Every runtime file the PR changes; tests are excluded.
IDENTITY_FILES = (
    "vllm/config/__init__.py",
    "vllm/config/vllm.py",
    "vllm/config/watchdog_config.py",
    "vllm/distributed/device_communicators/shm_broadcast.py",
    "vllm/engine/arg_utils.py",
    "vllm/utils/safe_fs.py",
    "vllm/utils/watch_dog.py",
    "vllm/v1/core/sched/scheduler.py",
    "vllm/v1/engine/core.py",
    "vllm/v1/executor/multiproc_executor.py",
    "vllm/v1/metrics/loggers.py",
    "vllm/v1/metrics/stats.py",
    "vllm/v1/outputs.py",
)
# From source at PR_HEAD: an idle EngineCore feeds every 5 s (input queue
# ``get(timeout=5)``) and an idle shm reader every 5 s
# (``SHM_READER_RECHECK_INTERVAL_MS``). A shorter timeout fires while idle.
IDLE_FEED_S = 5.0
FEED_TIMEOUT = "due to feed timeout"
METRIC_RE = re.compile(
    r"^vllm:watchdog_timeouts(?:_total)?\{(?P<labels>[^}]*)\}\s+(?P<value>\S+)$"
)
LABEL_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')
MIN_SAMPLES = 3
# Only files this probe and hold_worker.py create are ever removed.
MARKERS = ("hold_rank*", "held_rank*", "released_rank*", "release")


def parse_timeouts(text: str) -> dict[str, float]:
    """Sum the timeout counter by ``watchdog`` label; absent series are absent."""
    totals: dict[str, float] = {}
    for line in text.splitlines():
        match = METRIC_RE.match(line.strip())
        if match is None:
            continue
        name = dict(LABEL_RE.findall(match["labels"])).get("watchdog")
        if name is not None:
            totals[name] = totals.get(name, 0.0) + float(match["value"])
    return totals


def unscored(reason: str) -> dict[str, str]:
    return {"outcome": "unscored", "reason": reason}


def evaluate_control(
    warm: dict[str, float] | None,
    idle: dict[str, float] | None,
    requests_ok: bool,
    new_witness: bool,
    required: list[str],
) -> dict:
    """Case (c): series exist after warm-up and nothing changes while idle."""
    missing = [n for n in required if warm is None or n not in warm]
    changed = (
        warm is None
        or idle is None
        or any(warm.get(n) != idle.get(n) for n in set(warm) | set(idle))
    )
    ok = requests_ok and not new_witness and not missing and not changed
    return {
        "requests_ok": requests_ok,
        "new_witness": new_witness,
        "missing_series": missing,
        "changed_while_idle": changed,
        "ok": ok,
    }


def apparatus_failure(episode: dict) -> str | None:
    checks = (
        ("hold_entered", "hold_not_entered"),
        ("request_ok", "held_request_failed"),
        ("released", "release_not_observed"),
        ("post_release_requests_ok", "post_release_requests_failed"),
    )
    for key, reason in checks:
        if not episode.get(key):
            return reason
    if episode.get("baseline") is None:
        return "baseline_scrape_failed"
    if episode.get("post_release") is None:
        return "post_release_scrape_failed"
    return None


def post_witness_scrapes(episode: dict, witness_s: float) -> list[dict] | None:
    """Every scheduled scrape from the witness onward, or None if any failed."""
    scrapes = [s for t, s in episode["samples"] if t >= witness_s]
    return None if any(s is None for s in scrapes) else scrapes


def export_observed(episode: dict, name: str) -> bool:
    before = episode["baseline"].get(name) if episode.get("baseline") else None
    after = episode["post_release"].get(name) if episode.get("post_release") else None
    return before is not None and after is not None and after > before


def score_continuing(episode: dict, name: str) -> dict[str, str]:
    if reason := apparatus_failure(episode):
        return unscored(reason)
    witness_s = episode["witness_s"].get(name)
    if witness_s is None:
        return unscored("no_witness")
    baseline = episode["baseline"].get(name)
    if baseline is None:
        return unscored("series_absent_at_baseline")
    scrapes = post_witness_scrapes(episode, witness_s)
    if scrapes is None:
        return unscored("scrape_failed_during_hold")
    if len(scrapes) < MIN_SAMPLES:
        return unscored("too_few_samples_after_witness")
    if any(name not in s for s in scrapes):
        return unscored("series_absent_during_hold")
    if any(s[name] > baseline for s in scrapes):
        return {"outcome": "refuted", "reason": "counter_rose_during_hold"}
    if not export_observed(episode, name):
        return unscored("export_control_failed")
    return {"outcome": "supported", "reason": "counter_at_baseline_during_hold"}


def score_rank(episode: dict, name: str, export_control_ok: bool) -> dict:
    """Does a held non-output rank's count ever appear in /metrics?"""
    if reason := apparatus_failure(episode):
        return unscored(reason)
    witness_s = episode["witness_s"].get(name)
    if witness_s is None:
        return unscored("no_witness")
    scrapes = post_witness_scrapes(episode, witness_s)
    if scrapes is None:
        return unscored("scrape_failed_during_hold")
    if len(scrapes) < MIN_SAMPLES:
        return unscored("too_few_samples_after_witness")
    scrapes = [*scrapes, episode["post_release"]]
    baseline = episode["baseline"].get(name)
    values = [s[name] for s in scrapes if name in s]
    if any(value > (baseline or 0.0) for value in values):
        return {"outcome": "refuted", "reason": "non_output_rank_exported"}
    if baseline is not None or values:
        return unscored("series_present_without_rise")
    if not export_control_ok:
        return unscored("export_control_failed")
    return {"outcome": "supported", "reason": "non_output_rank_series_absent"}


def score(tp: int, episodes: list[dict], control_ok: bool) -> dict[str, dict]:
    first = episodes[0]
    outcomes = {"engine_core_continuing_hold": score_continuing(first, "engine_0")}
    if tp > 1:
        outcomes["worker_continuing_hold"] = score_continuing(first, "worker_0")
        outcomes["non_output_rank_exported"] = score_rank(
            episodes[1], "worker_1", export_observed(first, "worker_0")
        )
    if not control_ok:
        outcomes = {key: unscored("control_failed") for key in outcomes}
    return outcomes


def outcome_keys(tp: int) -> list[str]:
    keys = ["engine_core_continuing_hold"]
    if tp > 1:
        keys += ["worker_continuing_hold", "non_output_rank_exported"]
    return keys


# ---- runtime section (Linux host with the pinned vLLM installed) ----


def file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def check_identity(vllm_src: Path) -> dict:
    """Verify the vLLM that ``sys.executable`` imports; the server uses it too."""

    def run(*command: str) -> tuple[bool, str]:
        """Return (command succeeded, stdout); failure never looks like output."""
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return False, ""
        return done.returncode == 0, done.stdout.strip()

    head_ok, head = run("git", "-C", str(vllm_src), "rev-parse", "HEAD")
    status_ok, dirty = run("git", "-C", str(vllm_src), "status", "--porcelain", "-uno")
    code = "import os, vllm; print(os.path.dirname(vllm.__file__))"
    import_ok, installed = run(sys.executable, "-c", code)
    root = Path(installed).parent if import_ok and installed else None
    head = head if head_ok else ""
    clean = status_ok and dirty == ""
    mismatched = [
        rel
        for rel in IDENTITY_FILES
        if root is None
        or file_sha256(vllm_src / rel) is None
        or file_sha256(root / rel) != file_sha256(vllm_src / rel)
    ]
    here = Path(__file__).resolve().parent
    return {
        "source_head": head,
        "source_head_matches_pin": head == PR_HEAD,
        "source_status_ok": status_ok,
        "source_tracked_clean": clean,
        "vllm_importable": root is not None,
        "mismatched_files": mismatched,
        "apparatus_sha256": {
            name: file_sha256(here / name) for name in ("probe.py", "hold_worker.py")
        },
        "verified": head == PR_HEAD and clean and not mismatched,
    }


def http(url: str, body: dict | None = None, timeout: float = 5.0) -> str | None:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode()
    except Exception:
        return None


class Server:
    def __init__(self, args: argparse.Namespace, work: Path, control: Path) -> None:
        self.base = f"http://127.0.0.1:{args.port}"
        self.model = args.model
        self.dump_dir = work / "dumps"
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        watchdog = {
            "timeout": args.watchdog_timeout,
            "check_interval": args.check_interval,
            "dump_dir": str(self.dump_dir),
        }
        command = [
            sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", args.model,
            "--port", str(args.port),
            "--tensor-parallel-size", str(args.tp),
            "--distributed-executor-backend", "uni" if args.tp == 1 else "mp",
            "--max-model-len", "512",
            "--max-num-seqs", "4",
            "--gpu-memory-utilization", "0.5",
            "--worker-cls", "hold_worker.HoldingCPUWorker",
            "--watchdog-config", json.dumps(watchdog),
        ]  # fmt: skip
        here = str(Path(__file__).resolve().parent)
        env = dict(os.environ, PR55700_CONTROL_DIR=str(control))
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (here, env.get("PYTHONPATH", "")) if p
        )
        self.log = (work / "server.log").open("w")
        self.proc = subprocess.Popen(
            command, env=env, stdout=self.log, stderr=subprocess.STDOUT
        )

    def wait_healthy(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and self.proc.poll() is None:
            if http(self.base + "/health") is not None:
                return True
            time.sleep(2)
        return False

    def scrape(self) -> dict[str, float] | None:
        text = http(self.base + "/metrics")
        return None if text is None else parse_timeouts(text)

    def complete(self, timeout: float = 60.0) -> bool:
        body = {"model": self.model, "prompt": "Hello", "max_tokens": 8}
        return http(self.base + "/v1/completions", body, timeout) is not None

    def witness_count(self, name: str) -> int:
        return sum(
            path.read_text(errors="replace").count(FEED_TIMEOUT)
            for path in self.dump_dir.glob(f"VLLM_STACK_DUMP_for_{name}_*.log")
        )

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()


def run_control(server: Server, names: list[str], required: list[str], idle_s: float):
    witnesses = {name: server.witness_count(name) for name in names}
    requests_ok = all(server.complete() for _ in range(3))
    time.sleep(2.0)
    warm = server.scrape()
    time.sleep(idle_s)
    idle = server.scrape()
    new_witness = any(server.witness_count(n) != witnesses[n] for n in names)
    return evaluate_control(warm, idle, requests_ok, new_witness, required)


def run_episode(
    server: Server, control: Path, rank: int, names: list[str], hold_s: float
) -> dict:
    """Cases (a) and (b): hold ``rank``, sample during the hold, release."""
    for pattern in MARKERS:
        for marker in control.glob(pattern):
            marker.unlink()
    baseline = server.scrape()
    witness_before = {name: server.witness_count(name) for name in names}
    (control / f"hold_rank{rank}").write_text("1\n")
    result: dict[str, bool] = {}
    request = threading.Thread(
        target=lambda: result.update(ok=server.complete(hold_s + 120.0))
    )
    request.start()
    held = control / f"held_rank{rank}"
    deadline = time.monotonic() + 60.0
    while not held.exists() and time.monotonic() < deadline:
        time.sleep(0.2)
    entered = held.exists()
    start = time.monotonic()
    witness_s: dict[str, float | None] = dict.fromkeys(names)
    samples: list[list] = []
    while entered and time.monotonic() - start < hold_s:
        now = round(time.monotonic() - start, 1)
        for name in names:
            if witness_s[name] is None and (
                server.witness_count(name) > witness_before[name]
            ):
                witness_s[name] = now
        samples.append([now, server.scrape()])
        time.sleep(1.0)
    (control / "release").write_text("1\n")
    request.join(hold_s + 180.0)
    released = (control / f"released_rank{rank}").exists()
    post_ok = all(server.complete() for _ in range(2))
    time.sleep(2.0)
    return {
        "held_rank": rank,
        "hold_entered": entered,
        "request_ok": result.get("ok", False),
        "released": released,
        "post_release_requests_ok": post_ok,
        "witness_s": witness_s,
        "baseline": baseline,
        "samples": samples,
        "post_release": server.scrape(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tp", type=int, choices=(1, 2), required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--vllm-src", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8055)
    parser.add_argument("--watchdog-timeout", type=int, default=15)
    parser.add_argument("--check-interval", type=int, default=1)
    parser.add_argument("--hold-s", type=float, default=45.0)
    args = parser.parse_args()
    rpc_timeout = float(os.environ.get("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "300"))
    if args.watchdog_timeout <= IDLE_FEED_S + 2 * args.check_interval:
        parser.error("watchdog timeout must exceed the 5 s idle feed period")
    if args.hold_s <= args.watchdog_timeout + 5 * args.check_interval:
        parser.error("hold must outlast the watchdog timeout with margin")
    if args.tp > 1 and args.hold_s >= rpc_timeout - 30:
        parser.error("hold must stay well below the execute_model RPC deadline")

    work = args.work_dir.resolve()
    if work.exists() and (not work.is_dir() or any(work.iterdir())):
        parser.error("--work-dir must be new or empty; earlier evidence is kept")
    control = work / "control"
    control.mkdir(parents=True)
    record: dict = {
        "pr_head": PR_HEAD,
        "tp": args.tp,
        "model": args.model,
        "watchdog": {"timeout": args.watchdog_timeout, "check": args.check_interval},
        "hold_s": args.hold_s,
        "rpc_timeout_s": rpc_timeout,
        "identity": check_identity(args.vllm_src.resolve()),
    }

    def finish(outcomes: dict[str, dict]) -> int:
        record["outcomes"] = outcomes
        (work / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(outcomes, indent=2))
        return 0 if record.get("episodes") else 2

    if not record["identity"]["verified"]:
        return finish(
            dict.fromkeys(outcome_keys(args.tp), unscored("identity_unverified"))
        )

    names = ["engine_0"] if args.tp == 1 else ["engine_0", "worker_0", "worker_1"]
    required = ["engine_0"] if args.tp == 1 else ["engine_0", "worker_0"]
    server = Server(args, work, control)
    try:
        if not server.wait_healthy(900.0):
            return finish(
                dict.fromkeys(outcome_keys(args.tp), unscored("server_not_healthy"))
            )
        idle_s = args.watchdog_timeout + 3 * args.check_interval + 6
        record["control"] = run_control(server, names, required, idle_s)
        ranks = [0] if args.tp == 1 else [0, 1]
        record["episodes"] = [
            run_episode(server, control, rank, names, args.hold_s) for rank in ranks
        ]
        return finish(score(args.tp, record["episodes"], record["control"]["ok"]))
    finally:
        server.stop()


if __name__ == "__main__":
    raise SystemExit(main())
