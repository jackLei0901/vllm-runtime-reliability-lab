import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from dfxlab.collectors import (
    CadencedCollector,
    collect_health,
    collect_metrics,
    runtime_allowlist,
)
from dfxlab.external_schema import GpuAggregate, ProcessObservation


class FakeVllmHandler(BaseHTTPRequestHandler):
    health_status = 200
    metrics_body = "vllm:kv_cache_usage_perc 0.75\n"
    delay_seconds = 0.0

    def do_GET(self) -> None:
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        if self.path == "/health":
            self.send_response(self.health_status)
            self.end_headers()
            return
        if self.path == "/metrics":
            self.send_response(200)
            self.end_headers()
            try:
                self.wfile.write(self.metrics_body.encode("utf-8"))
            except BrokenPipeError:
                pass
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


class CollectorHttpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeVllmHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        host, port = cls.server.server_address
        cls.base_url = f"http://{host}:{port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self) -> None:
        FakeVllmHandler.health_status = 200
        FakeVllmHandler.metrics_body = "vllm:kv_cache_usage_perc 0.75\n"
        FakeVllmHandler.delay_seconds = 0.0

    def test_healthy_and_selected_metrics(self) -> None:
        health, health_error = collect_health(self.base_url, 1.0)
        metrics, metrics_error = collect_metrics(self.base_url, 1.0)
        self.assertTrue(health.ok)
        self.assertIsNone(health_error)
        self.assertEqual(metrics.kv_cache_usage, 0.75)
        self.assertIsNone(metrics_error)

    def test_health_503_is_an_observation_not_transport_error(self) -> None:
        FakeVllmHandler.health_status = 503
        health, error = collect_health(self.base_url, 1.0)
        self.assertFalse(health.ok)
        self.assertEqual(health.status, 503)
        self.assertIsNone(error)

    def test_any_2xx_health_status_is_healthy(self) -> None:
        FakeVllmHandler.health_status = 204
        health, error = collect_health(self.base_url, 1.0)
        self.assertTrue(health.ok)
        self.assertEqual(health.status, 204)
        self.assertIsNone(error)

    def test_malformed_metrics_is_bounded_error_kind(self) -> None:
        FakeVllmHandler.metrics_body = "not prometheus text"
        metrics, error = collect_metrics(self.base_url, 1.0)
        self.assertIsNone(metrics.kv_cache_usage)
        self.assertEqual(error, "malformed_metrics")

    def test_timeout_does_not_expose_exception_text(self) -> None:
        FakeVllmHandler.delay_seconds = 0.1
        health, error = collect_health(self.base_url, 0.01)
        self.assertFalse(health.ok)
        self.assertEqual(error, "timeout")


class CadenceTest(unittest.TestCase):
    def test_gpu_collector_runs_at_slower_cadence(self) -> None:
        clock = iter((0, 1_000_000_000))
        collector = CadencedCollector(
            "http://127.0.0.1:1",
            None,
            0.01,
            health_interval=1.0,
            metrics_interval=1.0,
            process_interval=1.0,
            gpu_interval=5.0,
            clock_ns=lambda: next(clock),
        )
        with (
            patch(
                "dfxlab.collectors.collect_health",
                return_value=(collector.health, None),
            ) as health,
            patch(
                "dfxlab.collectors.collect_metrics",
                return_value=(collector.metrics, None),
            ) as metrics,
            patch(
                "dfxlab.collectors.process_snapshot",
                return_value=ProcessObservation(False, None),
            ) as process,
            patch(
                "dfxlab.collectors.gpu_snapshot",
                return_value=(GpuAggregate(0), ()),
            ) as gpu,
        ):
            collector.collect(0)
            collector.collect(1)
        self.assertEqual(health.call_count, 2)
        self.assertEqual(metrics.call_count, 2)
        self.assertEqual(process.call_count, 2)
        self.assertEqual(gpu.call_count, 1)

    def test_records_collection_duration_by_source(self) -> None:
        duration_clock = iter(range(0, 8_000_001, 1_000_000))
        collector = CadencedCollector(
            "http://127.0.0.1:1",
            None,
            0.01,
            clock_ns=lambda: 0,
            duration_clock_ns=lambda: next(duration_clock),
        )
        with (
            patch(
                "dfxlab.collectors.collect_health",
                return_value=(collector.health, None),
            ),
            patch(
                "dfxlab.collectors.collect_metrics",
                return_value=(collector.metrics, None),
            ),
            patch(
                "dfxlab.collectors.process_snapshot",
                return_value=ProcessObservation(False, None),
            ),
            patch(
                "dfxlab.collectors.gpu_snapshot",
                return_value=(GpuAggregate(0), ()),
            ),
        ):
            collector.collect(0)
        for timing in collector.timing_summary().values():
            self.assertEqual(timing["count"], 1)
            self.assertEqual(timing["mean_ms"], 1.0)
            self.assertEqual(timing["max_ms"], 1.0)


class RuntimeAllowlistTest(unittest.TestCase):
    @patch("dfxlab.collectors.shutil.which", return_value=None)
    def test_target_versions_are_not_inferred_from_recorder_environment(
        self, _which: object
    ) -> None:
        runtime = runtime_allowlist()
        self.assertIsNone(runtime.vllm_version)
        self.assertIsNone(runtime.torch_version)

    @patch("dfxlab.collectors.shutil.which", return_value=None)
    def test_explicit_target_versions_are_preserved(self, _which: object) -> None:
        runtime = runtime_allowlist(
            target_vllm_version="0.23.1rc1+commit",
            target_torch_version="2.13.0+cu130",
        )
        self.assertEqual(runtime.vllm_version, "0.23.1rc1+commit")
        self.assertEqual(runtime.torch_version, "2.13.0+cu130")


if __name__ == "__main__":
    unittest.main()
