from __future__ import annotations

import contextlib
import io
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from dfxlab.cli import main
from dfxlab.collect_bundle import collect_bundle
from dfxlab.verify_bundle import verify_bundle


class _FlatServer(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/metrics":
            body = (
                b"vllm:generation_tokens_total 100\n"
                b"vllm:num_requests_running 1\n"
                b"vllm:num_requests_waiting 0\n"
            )
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n')
        self.wfile.flush()
        time.sleep(0.075)
        self.wfile.write(b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n')
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, _format: str, *_args) -> None:
        return


class CollectBundleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _FlatServer)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_cpu_only_collect_and_offline_verify(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bundle"
            summary = collect_bundle(
                output_dir=root,
                base_url=self.base_url,
                pid=os.getpid(),
                window=0.08,
                no_progress_window=0.04,
                sample_interval=0.015,
                timeout=0.5,
                unhealthy_samples=2,
                observation_only=False,
                decision_source="server_counter",
                progress_request=None,
                stack=False,
            )
            self.assertEqual(
                "alive_health_ok_no_progress", summary["verdict"]["verdict"]
            )
            self.assertEqual(summary, verify_bundle(root))
            public = (root / "observations.json").read_text(encoding="utf-8")
            self.assertNotIn(self.base_url, public)
            self.assertNotIn("hostname", public)

    def test_endpoint_only_no_progress_is_undetermined(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            summary = collect_bundle(
                output_dir=Path(temporary) / "bundle",
                base_url=self.base_url,
                pid=None,
                window=0.06,
                no_progress_window=0.03,
                sample_interval=0.01,
                timeout=0.5,
                unhealthy_samples=2,
                observation_only=False,
                decision_source="server_counter",
                progress_request=None,
                stack=False,
            )
            self.assertEqual("undetermined", summary["verdict"]["verdict"])

    def test_client_probe_reports_request_scoped_progress_without_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = root / "request.private.json"
            request.write_text(
                '{"model":"test","messages":[{"role":"user","content":"secret"}]}',
                encoding="utf-8",
            )
            output = root / "bundle"
            summary = collect_bundle(
                output_dir=output,
                base_url=self.base_url,
                pid=os.getpid(),
                window=0.12,
                no_progress_window=0.06,
                sample_interval=0.015,
                timeout=0.5,
                unhealthy_samples=2,
                observation_only=False,
                decision_source="client_request",
                progress_request=request,
                stack=False,
            )
            self.assertEqual("progress_observed", summary["verdict"]["verdict"])
            self.assertEqual("request", summary["verdict"]["progress_scope"])
            public = (output / "observations.json").read_text(encoding="utf-8")
            self.assertNotIn("secret", public)
            self.assertNotIn("messages", public)

    def test_pid_only_observation_mode_is_undetermined(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bundle"
            summary = collect_bundle(
                output_dir=root,
                base_url=None,
                pid=os.getpid(),
                window=0.04,
                no_progress_window=0.02,
                sample_interval=0.01,
                timeout=0.5,
                unhealthy_samples=2,
                observation_only=True,
                decision_source="server_counter",
                progress_request=None,
                stack=False,
            )
            self.assertEqual("undetermined", summary["verdict"]["verdict"])
            self.assertEqual(summary, verify_bundle(root))

    def test_collection_does_not_depend_on_coarse_monotonic_ns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bundle"
            with patch("dfxlab.collect_bundle.time.monotonic_ns", return_value=1):
                summary = collect_bundle(
                    output_dir=root,
                    base_url=None,
                    pid=os.getpid(),
                    window=0.04,
                    no_progress_window=0.02,
                    sample_interval=0.01,
                    timeout=0.5,
                    unhealthy_samples=2,
                    observation_only=True,
                    decision_source="server_counter",
                    progress_request=None,
                    stack=False,
                )
            self.assertEqual("undetermined", summary["verdict"]["verdict"])
            self.assertEqual(summary, verify_bundle(root))

    def test_cli_verify_and_invalid_pid_only_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bundle"
            collect_bundle(
                output_dir=root,
                base_url=self.base_url,
                pid=os.getpid(),
                window=0.06,
                no_progress_window=0.03,
                sample_interval=0.01,
                timeout=0.5,
                unhealthy_samples=2,
                observation_only=False,
                decision_source="server_counter",
                progress_request=None,
                stack=False,
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(0, main(["verify", str(root)]))
            self.assertIn("alive_health_ok_no_progress", stdout.getvalue())
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = main(
                    ["collect", "--pid", str(os.getpid()), "--output", str(root)]
                )
            self.assertEqual(1, status)
            self.assertIn("observation-only", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
