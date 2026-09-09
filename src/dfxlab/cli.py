from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path

from dfxlab.collectors import environment_snapshot, runtime_allowlist
from dfxlab.external_writer import IncidentWriter
from dfxlab.faults import inject_signal
from dfxlab.recorder import IncidentRecorder
from dfxlab.report import summarize_file
from dfxlab.schema import atomic_write_private_json


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
    raise AssertionError(f"unhandled command: {args.command}")
