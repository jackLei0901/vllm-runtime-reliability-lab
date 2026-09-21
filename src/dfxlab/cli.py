from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

from dfxlab.bundle import BundleError
from dfxlab.collect_bundle import collect_bundle
from dfxlab.collectors import environment_snapshot, runtime_allowlist
from dfxlab.external_writer import IncidentWriter
from dfxlab.faults import inject_signal
from dfxlab.recorder import IncidentRecorder
from dfxlab.replay import ReplayError, replay
from dfxlab.report import summarize_file
from dfxlab.schema import atomic_write_private_json
from dfxlab.verify_bundle import verify_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vllm-dfx")
    subparsers = parser.add_subparsers(dest="command", required=True)

    env_parser = subparsers.add_parser("snapshot-env", help="capture environment")
    env_parser.add_argument("--output", type=Path)

    record_parser = subparsers.add_parser(
        "record", help="record bounded runtime history"
    )
    record_parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    record_parser.add_argument("--pid", type=int)
    record_parser.add_argument("--output", type=Path, required=True)
    record_parser.add_argument(
        "--sample-interval",
        "--interval",
        dest="sample_interval",
        type=float,
        default=1.0,
    )
    record_parser.add_argument("--health-interval", type=float, default=1.0)
    record_parser.add_argument("--metrics-interval", type=float, default=1.0)
    record_parser.add_argument("--process-interval", type=float, default=1.0)
    record_parser.add_argument("--gpu-interval", type=float, default=5.0)
    record_parser.add_argument("--duration", type=float)
    record_parser.add_argument("--history", type=int, default=300)
    record_parser.add_argument("--timeout", type=float, default=1.0)
    record_parser.add_argument("--kv-threshold", type=float, default=0.95)
    record_parser.add_argument("--unhealthy-samples", type=int, default=3)
    record_parser.add_argument("--preemption-delta", type=float, default=20.0)
    record_parser.add_argument("--incident-cooldown", type=float, default=60.0)
    record_parser.add_argument("--stop-on-incident", action="store_true")
    record_parser.add_argument(
        "--private-raw-timeline",
        action="store_true",
        help="write an unbounded private lab timeline (disabled by default)",
    )
    record_parser.add_argument("--max-artifact-kib", type=int, default=256)
    record_parser.add_argument("--max-artifacts", type=int, default=4)
    record_parser.add_argument(
        "--target-vllm-version",
        help="exact observed-server vLLM version; omitted rather than inferred",
    )
    record_parser.add_argument(
        "--target-torch-version",
        help="exact observed-server Torch version; omitted rather than inferred",
    )

    signal_parser = subparsers.add_parser(
        "inject-signal", help="send an explicit signal"
    )
    signal_parser.add_argument("--pid", type=int, required=True)
    signal_parser.add_argument("--signal", default="SIGKILL")
    signal_parser.add_argument("--event-log", type=Path)
    signal_parser.add_argument("--dry-run", action="store_true")

    summary_parser = subparsers.add_parser("summarize", help="render incident markdown")
    summary_parser.add_argument("--input", type=Path, required=True)
    summary_parser.add_argument("--output", type=Path, required=True)

    collect_parser = subparsers.add_parser(
        "collect", help="collect a bounded v0.2 incident evidence bundle"
    )
    collect_parser.add_argument("--base-url")
    collect_parser.add_argument("--pid", type=int)
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument("--window", type=float, default=60.0)
    collect_parser.add_argument("--no-progress-window", type=float, default=10.0)
    collect_parser.add_argument("--sample-interval", type=float, default=1.0)
    collect_parser.add_argument("--timeout", type=float, default=1.0)
    collect_parser.add_argument("--unhealthy-samples", type=int, default=3)
    collect_parser.add_argument("--observation-only", action="store_true")
    collect_parser.add_argument(
        "--decision-source",
        choices=("server_counter", "client_request"),
        default="server_counter",
    )
    collect_parser.add_argument("--progress-request", type=Path)
    collect_parser.add_argument("--stack", action="store_true")

    verify_parser = subparsers.add_parser(
        "verify", help="offline verification of a v0.2 evidence bundle"
    )
    verify_parser.add_argument("bundle", type=Path)

    replay_parser = subparsers.add_parser(
        "replay", help="fail-closed replay of a published lab result"
    )
    replay_parser.add_argument("result_dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "snapshot-env":
        payload = environment_snapshot()
        if args.output:
            atomic_write_private_json(args.output, payload)
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if args.command == "record":
        writer = IncidentWriter(
            output_dir=args.output,
            max_artifact_bytes=args.max_artifact_kib * 1024,
            max_artifacts=args.max_artifacts,
        )
        recorder = IncidentRecorder(
            base_url=args.base_url,
            output_dir=args.output,
            pid=args.pid,
            sample_interval=args.sample_interval,
            history_size=args.history,
            timeout=args.timeout,
            health_interval=args.health_interval,
            metrics_interval=args.metrics_interval,
            process_interval=args.process_interval,
            gpu_interval=args.gpu_interval,
            kv_threshold=args.kv_threshold,
            unhealthy_samples=args.unhealthy_samples,
            preemption_delta=args.preemption_delta,
            incident_cooldown=args.incident_cooldown,
            private_raw_timeline=args.private_raw_timeline,
            writer=writer,
            runtime=runtime_allowlist(
                target_vllm_version=args.target_vllm_version,
                target_torch_version=args.target_torch_version,
            ),
        )
        handled_signals = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGBREAK"):
            handled_signals.append(signal.SIGBREAK)
        for signum in handled_signals:
            signal.signal(signum, lambda _signum, _frame: recorder.request_stop())
        return recorder.run(args.duration, args.stop_on_incident)
    if args.command == "inject-signal":
        event = inject_signal(
            args.pid, args.signal, args.event_log, dry_run=args.dry_run
        )
        print(json.dumps(event, ensure_ascii=False, indent=2))
        return 0
    if args.command == "summarize":
        summarize_file(args.input, args.output)
        return 0
    if args.command == "collect":
        try:
            summary = collect_bundle(
                output_dir=args.output,
                base_url=args.base_url,
                pid=args.pid,
                window=args.window,
                no_progress_window=args.no_progress_window,
                sample_interval=args.sample_interval,
                timeout=args.timeout,
                unhealthy_samples=args.unhealthy_samples,
                observation_only=args.observation_only,
                decision_source=args.decision_source,
                progress_request=args.progress_request,
                stack=args.stack,
            )
        except (BundleError, ValueError, OSError) as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(summary["verdict"], ensure_ascii=False))
        return 0
    if args.command == "verify":
        try:
            summary = verify_bundle(args.bundle)
        except BundleError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(summary["verdict"], ensure_ascii=False))
        return 0
    if args.command == "replay":
        try:
            lines = replay(args.result_dir.resolve())
        except ReplayError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 1
        print("\n".join(lines))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")
