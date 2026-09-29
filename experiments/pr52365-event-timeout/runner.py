"""PR #52365: is the default 60 s event-wait bound fatal for a step that completes?

Frozen record: docs/reviews/PR52365_SECOND_CANDIDATE_2026-09-29.zh-CN.md and
addendum A1 (docs/reviews/PR52365_SECOND_CANDIDATE_ADDENDUM_A1_2026-09-29.zh-CN.md).
Pins: base ``157bcb7c489689dd34cf28d9c9970a465d326a03``, PR head
``d996d76ec68e9f6a348b7b085ae1da61cb6095be``.

``sweep`` (base only) sends one-token requests and records whole-request wall
time per prompt length, screening for a request >= 70 s and a short control
<= 50 s. This does not establish an event-wait duration. ``ab`` then runs
the same two requests on three fresh
servers: base, PR head with default settings, and PR head with
``VLLM_ENGINE_ITERATION_TIMEOUT_S=0``. A private import-time wrapper measures
the unchanged wait function in the PR arms. vLLM source is never modified.

Every server is launched by the interpreter whose installation was verified;
identity failure writes an ``unscored`` receipt and starts nothing. The work
directory must be new or empty. ``result.json`` holds settings, aggregate wait
latencies, booleans and outcomes; server logs and per-call traces stay private.
Functions above the runtime section import only the standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

BASE_PIN = "157bcb7c489689dd34cf28d9c9970a465d326a03"
PR_PIN = "d996d76ec68e9f6a348b7b085ae1da61cb6095be"
# Runtime files the PR changes; event_utils.py is absent at the base.
PR_FILES = (
    "vllm/v1/worker/gpu/async_utils.py",
    "vllm/v1/worker/gpu/event_utils.py",
    "vllm/v1/worker/gpu_model_runner.py",
)
LONG_MIN_S = 70.0  # HTTP-time screening only, not proof of an event wait
SHORT_MAX_S = 50.0
WAIT_LONG_MIN_S = 65.0  # measured, completing event wait on the disabled arm
TIMEOUT_TEXT = "waiting for"
TIMEOUT_PREFIX = "Timed out after"
FATAL_TEXT = "EngineCore encountered a fatal error"


def unscored(reason: str) -> dict[str, str]:
    return {"outcome": "unscored", "reason": reason}


def select_lengths(sweep: list[dict]) -> dict:
    """Use one predeclared ladder step beyond the first >=70 s screen."""
    ok = [r for r in sweep if r["ok"] and r["seconds"] is not None]
    shorts = [r for r in ok if r["seconds"] <= SHORT_MAX_S]
    first_long_index = next(
        (i for i, r in enumerate(sweep) if r["ok"] and r["seconds"] >= LONG_MIN_S),
        None,
    )
    next_step = (
        sweep[first_long_index + 1]
        if first_long_index is not None and first_long_index + 1 < len(sweep)
        else None
    )
    selected = (
        next_step
        if next_step is not None
        and next_step["ok"]
        and next_step["seconds"] >= LONG_MIN_S
        else None
    )
    return {
        "screen": sweep[first_long_index]["tokens"]
        if first_long_index is not None
        else None,
        "long": selected["tokens"] if selected is not None else None,
        "short": max((r["tokens"] for r in shorts), default=None),
        "max_seconds": max((r["seconds"] for r in ok), default=None),
        "outcome": "candidate_found"
        if selected is not None and shorts
        else "no_candidate",
    }


def score_ab(identity_ok: bool, arms: dict[str, dict]) -> dict[str, str]:
    """Score the default-bound claim from the three arms (see addendum A1)."""
    if not identity_ok:
        return unscored("identity_unverified")
    names = ("base", "pr_default", "pr_disabled")
    if any(name not in arms or not arms[name].get("healthy") for name in names):
        return unscored("server_not_healthy")
    if not all(arms[name]["short_ok"] for name in names):
        return unscored("short_control_failed")
    base, pr, disabled = arms["base"], arms["pr_default"], arms["pr_disabled"]
    if any(arms[name]["short_s"] > SHORT_MAX_S for name in names):
        return unscored("short_control_not_short")
    if not base["long_ok"] or base["long_s"] < LONG_MIN_S:
        return unscored("base_long_step_not_established")
    if not disabled["long_ok"]:
        return unscored("pr_disabled_long_failed")
    if any(
        not arms[name][phase + "_wait"]["valid"]
        for name in ("pr_default", "pr_disabled")
        for phase in ("short", "long")
    ):
        return unscored("event_trace_invalid")
    if any(
        arms[name][phase + "_wait"]["count"] > 0
        and arms[name][phase + "_wait"]["sources"] != ["v1"]
        for name in ("pr_default", "pr_disabled")
        for phase in ("short", "long")
    ):
        return unscored("unexpected_runner_path")
    if disabled["long_wait"]["returned_max_s"] < WAIT_LONG_MIN_S:
        return unscored("long_event_wait_not_established")
    if any(
        arms[name]["short_wait"]["count"] == 0
        or arms[name]["short_wait"]["timeout_count"]
        for name in ("pr_default", "pr_disabled")
    ):
        return unscored("short_event_control_unverified")
    if pr["long_ok"]:
        if pr["long_wait"]["count"] == 0:
            return unscored("default_long_wait_unobserved")
        return {"outcome": "not_reproduced", "reason": "default_arm_completed"}
    if not pr["timeout_logged"] or not pr["long_wait"]["timeout_count"]:
        return unscored("failure_not_attributed_to_event_timeout")
    return {
        "outcome": "supported",
        "reason": "default_bound_rejected_completing_request",
    }


def wait_summary(path: Path, start_line: int) -> tuple[dict, int]:
    """Read a complete-line, per-call trace; missing/malformed data fail closed."""
    if not path.exists():
        return {
            "valid": False,
            "count": 0,
            "returned_max_s": 0.0,
            "timeout_count": 0,
            "sources": [],
        }, 0
    data = path.read_text()
    if not data.endswith("\n"):
        return {
            "valid": False,
            "count": 0,
            "returned_max_s": 0.0,
            "timeout_count": 0,
            "sources": [],
        }, 0
    lines = data.splitlines()
    waits = []
    try:
        if not any(json.loads(line).get("kind") == "installed" for line in lines):
            raise ValueError("trace hook not installed")
        for line in lines[start_line:]:
            event = json.loads(line)
            if event.get("kind") == "wait":
                seconds = event["seconds"]
                if (
                    not isinstance(seconds, (int, float))
                    or not math.isfinite(seconds)
                    or seconds < 0
                ):
                    raise ValueError("invalid duration")
                if event.get("status") not in ("returned", "timeout", "error"):
                    raise ValueError("invalid status")
                if event.get("source") not in ("v1", "v2", "other"):
                    raise ValueError("invalid caller source")
                waits.append(event)
            elif event.get("kind") != "installed":
                raise ValueError("invalid trace event")
    except (ValueError, KeyError, TypeError):
        return {
            "valid": False,
            "count": 0,
            "returned_max_s": 0.0,
            "timeout_count": 0,
            "sources": [],
        }, len(lines)
    return {
        "valid": True,
        "count": len(waits),
        "returned_max_s": max(
            (x["seconds"] for x in waits if x["status"] == "returned"), default=0.0
        ),
        "timeout_count": sum(x["status"] == "timeout" for x in waits),
        "sources": sorted({x["source"] for x in waits}),
    }, len(lines)


# ---- runtime section (Linux GPU host; nothing below runs in the tests) ----


def file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def run(
    *command: str, env: dict | None = None, cwd: Path | None = None
) -> tuple[bool, str]:
    try:
        done = subprocess.run(
            command, capture_output=True, text=True, timeout=120, env=env, cwd=cwd
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, ""
    return done.returncode == 0, done.stdout.strip()


def check_identity(python: str, src: Path, pin: str) -> dict:
    """Verify the vLLM that ``python`` imports; the server uses that interpreter."""
    head_ok, head = run("git", "-C", str(src), "rev-parse", "HEAD")
    status_ok, dirty = run("git", "-C", str(src), "status", "--porcelain", "-uno")
    clean_env = dict(os.environ)
    hook_dir = Path(__file__).resolve().parent
    python_paths = clean_env.get("PYTHONPATH", "").split(os.pathsep)
    clean_env["PYTHONPATH"] = os.pathsep.join(
        path for path in python_paths if path and Path(path).resolve() != hook_dir
    )
    site_code = (
        "import importlib.util as u; "
        "s=u.find_spec('sitecustomize'); print(s.origin if s else '')"
    )
    site_ok, site_origin = run(python, "-c", site_code, env=clean_env, cwd=src.parent)
    code = "import os, vllm; print(os.path.dirname(vllm.__file__))"
    import_ok, installed = (
        run(python, "-c", code, env=clean_env, cwd=src.parent)
        if site_ok and not site_origin
        else (False, "")
    )
    root = Path(installed).parent if import_ok and installed else None
    mismatched = [
        rel
        for rel in PR_FILES
        if root is None or file_sha256(root / rel) != file_sha256(src / rel)
    ]
    base_event_file = "vllm/v1/worker/gpu/event_utils.py"
    if pin == BASE_PIN and (
        (src / base_event_file).exists()
        or (root is not None and (root / base_event_file).exists())
    ):
        mismatched.append(base_event_file + "_unexpected_at_base")
    head = head if head_ok else ""
    clean = status_ok and dirty == ""
    return {
        "pin": pin,
        "head_matches_pin": head == pin,
        "source_status_ok": status_ok,
        "source_tracked_clean": clean,
        "sitecustomize_check_ok": site_ok,
        "existing_sitecustomize": bool(site_origin),
        "vllm_importable": root is not None,
        "mismatched_files": mismatched,
        "verified": head == pin
        and clean
        and site_ok
        and not site_origin
        and not mismatched,
    }


def http(url: str, body: dict | None = None, timeout: float = 10.0) -> str | None:
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
    def __init__(
        self, args, python: str, log: Path, env_extra: dict, trace: Path | None = None
    ) -> None:
        self.base = f"http://127.0.0.1:{args.port}"
        self.model = args.model
        self.trace = trace
        self.trace_line = 0
        command = [
            python, "-m", "vllm.entrypoints.cli.main", "serve", args.model,
            "--port", str(args.port),
            "--tensor-parallel-size", "1",
            "--distributed-executor-backend", "uni",
            "--async-scheduling",
            "--no-enable-prefix-caching",
            "--max-num-seqs", "1",
            "--max-model-len", str(args.max_model_len),
            "--max-num-batched-tokens", str(args.max_model_len),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--revision", args.model_revision,
        ]  # fmt: skip
        env = dict(os.environ, **env_extra)
        env.pop("LLR_WAIT_TRACE_PATH", None)
        env["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        if trace is not None:
            env["LLR_WAIT_TRACE_PATH"] = str(trace)
            env["PYTHONPATH"] = os.pathsep.join(
                filter(
                    None, (str(Path(__file__).resolve().parent), env.get("PYTHONPATH"))
                )
            )
        self.log = log.open("w")
        self.proc = subprocess.Popen(
            command,
            env=env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def wait_healthy(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and self.proc.poll() is None:
            if http(self.base + "/health") is not None:
                return True
            time.sleep(3)
        return False

    def one_token(self, tokens: int, token_id: int, timeout_s: float):
        """Time the request and attribute new event-wait records to it."""
        body = {
            "model": self.model,
            "prompt": [token_id] * tokens,
            "max_tokens": 1,
            "temperature": 0.0,
        }
        start = time.monotonic()
        ok = http(self.base + "/v1/completions", body, timeout_s) is not None
        seconds = round(time.monotonic() - start, 2)
        if self.trace is None:
            return ok, seconds, None
        summary, self.trace_line = wait_summary(self.trace, self.trace_line)
        return ok, seconds, summary

    def stop(self) -> None:
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(90)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.proc.wait(10)
        self.log.close()


def fresh_work_dir(parser: argparse.ArgumentParser, path: Path) -> Path:
    work = path.resolve()
    if work.exists() and (not work.is_dir() or any(work.iterdir())):
        parser.error("--work-dir must be new or empty; earlier evidence is kept")
    work.mkdir(parents=True)
    return work


def finish(work: Path, record: dict) -> int:
    (work / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record.get("outcome"), indent=2))
    return 0 if record.get("outcome", {}).get("outcome") != "unscored" else 2


def cmd_sweep(args, parser) -> int:
    work = fresh_work_dir(parser, args.work_dir)
    record: dict = {
        "stage": "sweep",
        "runner_sha256": file_sha256(Path(__file__)),
        "model": args.model,
        "model_revision": args.model_revision,
        "forced_runner": "VLLM_USE_V2_MODEL_RUNNER=0",
        "max_model_len": args.max_model_len,
        "identity": check_identity(args.python, args.vllm_src, BASE_PIN),
    }
    if not record["identity"]["verified"]:
        record["outcome"] = unscored("identity_unverified")
        return finish(work, record)
    try:
        server = Server(args, args.python, work / "server.log", {})
        try:
            if (
                not server.wait_healthy(1800.0)
                or not server.one_token(256, 100, 300)[0]
            ):
                record["outcome"] = unscored("server_not_healthy")
                return finish(work, record)
            sweep = []
            crossed_screen = False
            for index, tokens in enumerate(sorted(args.lengths)):
                ok, seconds, _ = server.one_token(
                    tokens, 1000 + index, args.request_timeout
                )
                sweep.append(
                    {"tokens": tokens, "ok": ok, "seconds": seconds if ok else None}
                )
                if not ok or crossed_screen:
                    break
                if seconds >= LONG_MIN_S:
                    crossed_screen = True
            record["sweep"] = sweep
            record["outcome"] = select_lengths(sweep)
            return finish(work, record)
        finally:
            server.stop()
    except (OSError, subprocess.TimeoutExpired) as exc:
        record["outcome"] = unscored("apparatus_error")
        record["error_type"] = type(exc).__name__
        return finish(work, record)


def run_arm(args, name: str, python: str, env: dict, work: Path) -> dict:
    trace = None if name == "base" else work / f"{name}.waits.jsonl"
    server = Server(args, python, work / f"{name}.log", env, trace)
    arm: dict = {"env": env, "healthy": False}
    long_log_offset = 0
    try:
        arm["healthy"] = (
            server.wait_healthy(1800.0) and server.one_token(256, 100, 300)[0]
        )
        if arm["healthy"]:
            arm["short_ok"], arm["short_s"], arm["short_wait"] = server.one_token(
                args.short, 2000, args.request_timeout
            )
            server.log.flush()
            long_log_offset = (work / f"{name}.log").stat().st_size
            arm["long_ok"], arm["long_s"], arm["long_wait"] = server.one_token(
                args.long, 3000, args.request_timeout
            )
            time.sleep(10.0)
            arm["health_after_long_ok"] = http(server.base + "/health") is not None
            arm["server_exit_before_stop"] = server.proc.poll()
    finally:
        server.stop()
    log_path = work / f"{name}.log"
    text = log_path.read_text(errors="replace")
    with log_path.open("rb") as stream:
        stream.seek(long_log_offset)
        long_text = stream.read().decode(errors="replace")
    arm["timeout_logged"] = any(
        TIMEOUT_PREFIX in line and TIMEOUT_TEXT in line
        for line in long_text.splitlines()
    )
    arm["fatal_logged"] = FATAL_TEXT in text
    if trace is not None:
        arm["trace_sha256"] = file_sha256(trace)
    return arm


def cmd_ab(args, parser) -> int:
    work = fresh_work_dir(parser, args.work_dir)
    identity = {
        "base": check_identity(args.base_python, args.base_src, BASE_PIN),
        "pr": check_identity(args.pr_python, args.pr_src, PR_PIN),
    }
    record: dict = {
        "stage": "ab",
        "runner_sha256": file_sha256(Path(__file__)),
        "model": args.model,
        "model_revision": args.model_revision,
        "forced_runner": "VLLM_USE_V2_MODEL_RUNNER=0",
        "max_model_len": args.max_model_len,
        "short_tokens": args.short,
        "long_tokens": args.long,
        "identity": identity,
        "trace_hook_sha256": file_sha256(Path(__file__).with_name("sitecustomize.py")),
    }
    identity_ok = all(v["verified"] for v in identity.values())
    if not identity_ok or record["trace_hook_sha256"] is None:
        record["outcome"] = unscored("identity_unverified")
        return finish(work, record)
    record["arms"] = {}
    try:
        for name, python, env in (
            ("base", args.base_python, {}),
            ("pr_default", args.pr_python, {}),
            ("pr_disabled", args.pr_python, {"VLLM_ENGINE_ITERATION_TIMEOUT_S": "0"}),
        ):
            record["arms"][name] = run_arm(args, name, python, env, work)
    except (OSError, subprocess.TimeoutExpired) as exc:
        record["outcome"] = unscored("apparatus_error")
        record["error_type"] = type(exc).__name__
        return finish(work, record)
    record["outcome"] = score_ab(identity_ok, record["arms"])
    return finish(work, record)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", required=True)
    common.add_argument("--model-revision", required=True)
    common.add_argument("--max-model-len", type=int, required=True)
    common.add_argument("--work-dir", type=Path, required=True)
    common.add_argument("--port", type=int, default=8065)
    common.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    common.add_argument("--request-timeout", type=float, default=600.0)
    sweep = sub.add_parser("sweep", parents=[common])
    sweep.add_argument("--python", required=True)
    sweep.add_argument("--vllm-src", type=Path, required=True)
    sweep.add_argument("--lengths", type=int, nargs="+", required=True)
    ab = sub.add_parser("ab", parents=[common])
    ab.add_argument("--base-python", required=True)
    ab.add_argument("--base-src", type=Path, required=True)
    ab.add_argument("--pr-python", required=True)
    ab.add_argument("--pr-src", type=Path, required=True)
    ab.add_argument("--short", type=int, required=True)
    ab.add_argument("--long", type=int, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.model_revision):
        parser.error("--model-revision must be a 40-hex-character commit SHA")
    if args.max_model_len <= 1:
        parser.error("--max-model-len must exceed one token")
    if args.max_model_len < 257:
        parser.error("--max-model-len must fit the 256-token warm-up plus output")
    if os.environ.get("VLLM_ENGINE_ITERATION_TIMEOUT_S") is not None:
        parser.error("unset VLLM_ENGINE_ITERATION_TIMEOUT_S; arms set it themselves")
    if args.command == "sweep" and any(length <= 0 for length in args.lengths):
        parser.error("all prompt lengths must be positive")
    if args.command == "sweep" and (
        len(args.lengths) < 2 or args.lengths != sorted(set(args.lengths))
    ):
        parser.error("--lengths needs at least two distinct increasing values")
    if args.command == "ab" and (args.short <= 0 or args.long <= args.short):
        parser.error("require 0 < --short < --long")
    if args.command == "ab" and args.long + 1 > args.max_model_len:
        parser.error("--long plus one output token must fit --max-model-len")
    if args.command == "sweep" and max(args.lengths) + 1 > args.max_model_len:
        parser.error(
            "all prompt lengths plus one output token must fit --max-model-len"
        )
    return cmd_sweep(args, parser) if args.command == "sweep" else cmd_ab(args, parser)


if __name__ == "__main__":
    raise SystemExit(main())
