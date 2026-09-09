from __future__ import annotations

import json
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKLOAD = ROOT / "experiments" / "overhead" / "openai_completions_workload.py"


class StreamingCompletionHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.path != "/v1/completions":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        if not payload.get("stream") or payload.get("model") != "fake-model":
            self.send_response(400)
            self.end_headers()
            return
        chunks = [
            {"choices": [{"text": "one"}], "usage": None},
            {"choices": [{"text": " two"}], "usage": None},
            {"choices": [], "usage": {"completion_tokens": 2}},
        ]
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        body += "data: [DONE]\n\n"
        encoded = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        return


class OpenAIWorkloadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), StreamingCompletionHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_streaming_metrics_and_signature(self) -> None:
        host, port = self.server.server_address
        command = [
            sys.executable,
            str(WORKLOAD),
            "--base-url",
            f"http://{host}:{port}",
            "--model",
            "fake-model",
            "--requests",
            "4",
            "--concurrency",
            "2",
            "--warmup-requests",
            "1",
            "--prompt-repeat",
            "2",
            "--max-tokens",
            "2",
        ]
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=30, check=False
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["request_count"], 4)
        self.assertEqual(payload["success_count"], 4)
        self.assertEqual(payload["metrics"]["completion_tokens"], 8)
        self.assertEqual(payload["metrics"]["failed_requests"], 0)
        self.assertIn("ttft_ms_p95", payload["metrics"])
        self.assertIn("tpot_ms_p95", payload["metrics"])
        self.assertEqual(len(payload["workload_signature"]), 64)


if __name__ == "__main__":
    unittest.main()
