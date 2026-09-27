"""Open this public, byte-pinned demo trace in the Perfetto UI once.

The file is served only from 127.0.0.1:9001. The process exits after one
successful GET or after a two-minute timeout. No Lab server is left running.
"""

from __future__ import annotations

import argparse
import hashlib
import http.server
import time
import urllib.parse
import webbrowser
from pathlib import Path

TRACE_NAME = "timeline.trace.json"
TRACE_SHA256 = "b7ea05a74fcdd8e688c60ec244e15c07be8edaa5d2813dbca1070eabb26aec8b"
ORIGIN = "https://ui.perfetto.dev"
PORT = 9001  # Perfetto's documented command-line opener uses this port.


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-open-browser", action="store_true")
    args = parser.parse_args()
    path = Path(__file__).with_name(TRACE_NAME)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != TRACE_SHA256:
        parser.error("demo trace digest differs; regenerate and review it first")

    class Handler(http.server.BaseHTTPRequestHandler):
        served = False

        def do_GET(self) -> None:
            if self.path != f"/{TRACE_NAME}":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", ORIGIN)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            type(self).served = True

        def log_message(self, format: str, *args: object) -> None:
            return

    trace_url = f"http://127.0.0.1:{PORT}/{TRACE_NAME}"
    viewer_url = f"{ORIGIN}/#!/?{urllib.parse.urlencode({'url': trace_url})}"
    try:
        with http.server.HTTPServer(("127.0.0.1", PORT), Handler) as server:
            print(f"Open in browser: {viewer_url}", flush=True)
            if not args.no_open_browser:
                webbrowser.open_new_tab(viewer_url)
            deadline = time.monotonic() + 120
            while not Handler.served and time.monotonic() < deadline:
                server.timeout = min(5, deadline - time.monotonic())
                server.handle_request()
            if not Handler.served:
                parser.error("Perfetto did not fetch the trace within two minutes")
    except OSError as error:
        parser.error(f"cannot serve trace on 127.0.0.1:{PORT}: {error}")
    print("Trace fetched once; local server stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
