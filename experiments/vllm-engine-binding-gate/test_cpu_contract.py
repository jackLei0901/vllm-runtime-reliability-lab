"""CPU-only controls for a proposed per-engine recorder gate.

These vectors exercise existing v0.2 pure rules after an engine partition is
supplied. They do not implement label parsing, process-title capture, or a new
engine-scoped verdict.
"""

import unittest

from dfxlab.progress import derive_verdict, evaluate_demand, evaluate_producer
from dfxlab.prometheus import select_metrics


def counter(first: int, last: int, *, fresh: bool = True) -> dict:
    return {
        "producer_available": True,
        "evaluation_start_ns": 0,
        "evaluation_end_ns": 100,
        "minimum_span_ns": 100,
        "minimum_fresh_samples": 2,
        "samples": [
            {"monotonic_ns": 0, "fresh": fresh, "value": first},
            {"monotonic_ns": 100, "fresh": fresh, "value": last},
        ],
    }


def demand(*, running: int = 1) -> dict:
    return {
        "evaluation_start_ns": 0,
        "evaluation_end_ns": 100,
        "minimum_fresh_samples": 2,
        "samples": [
            {"monotonic_ns": 0, "fresh": True, "running": running, "waiting": 0},
            {"monotonic_ns": 100, "fresh": True, "running": running, "waiting": 0},
        ],
    }


def verdict(
    decision_state: str,
    *,
    process_supplied: bool = True,
    identity_stable: bool = True,
    corroborating_state: str = "producer_missing",
) -> dict:
    return derive_verdict(
        {
            "process_supplied": process_supplied,
            "process_identity_stable": identity_stable if process_supplied else False,
            "process_alive_throughout": process_supplied,
            "endpoint_supplied": True,
            "health_state": "ok",
            "decision_source": "server_counter",
            "decision_state": decision_state,
            "corroborating_state": corroborating_state,
            "demand_state": "present",
            "stack_state": "disabled",
        }
    )


class EngineBindingCpuGate(unittest.TestCase):
    def test_existing_metric_selection_erases_engine_partition(self) -> None:
        first_scrape = (
            'vllm:generation_tokens_total{model_name="m",engine="0"} 10\n'
            'vllm:generation_tokens_total{model_name="m",engine="1"} 20\n'
        )
        second_scrape = (
            'vllm:generation_tokens_total{model_name="m",engine="0"} 10\n'
            'vllm:generation_tokens_total{model_name="m",engine="1"} 21\n'
        )
        self.assertEqual(
            select_metrics(first_scrape, {"vllm:generation_tokens_total"}),
            {"vllm:generation_tokens_total": 30.0},
        )
        self.assertEqual(
            select_metrics(second_scrape, {"vllm:generation_tokens_total"}),
            {"vllm:generation_tokens_total": 31.0},
        )

    def test_one_flat_engine_is_hidden_by_rising_aggregate(self) -> None:
        engine_0 = evaluate_producer("server_counter", counter(10, 10))
        engine_1 = evaluate_producer("server_counter", counter(20, 21))
        aggregate = evaluate_producer("server_counter", counter(30, 31))
        self.assertEqual(engine_0["state"], "flat")
        self.assertEqual(engine_1["state"], "progressing")
        self.assertEqual(aggregate["state"], "progressing")
        self.assertEqual(
            evaluate_demand("server_counter", demand())["state"], "present"
        )
        self.assertEqual(
            verdict(engine_0["state"])["verdict"], "alive_health_ok_no_progress"
        )
        self.assertEqual(verdict(aggregate["state"])["verdict"], "progress_observed")
        # v0.2 labels this result "service"; the proposed engine claim needs
        # its own versioned scope contract and cannot reuse that label.
        self.assertEqual(verdict(engine_0["state"])["progress_scope"], "service")

    def test_no_work_prevents_flat_counter_from_becoming_a_claim(self) -> None:
        self.assertEqual(
            evaluate_demand("server_counter", demand(running=0))["state"], "absent"
        )
        self.assertEqual(
            derive_verdict(
                {
                    "process_supplied": True,
                    "process_identity_stable": True,
                    "process_alive_throughout": True,
                    "endpoint_supplied": True,
                    "health_state": "ok",
                    "decision_source": "server_counter",
                    "decision_state": "flat",
                    "corroborating_state": "producer_missing",
                    "demand_state": "absent",
                    "stack_state": "disabled",
                }
            )["verdict"],
            "undetermined",
        )

    def test_cached_counter_is_insufficient(self) -> None:
        self.assertEqual(
            evaluate_producer("server_counter", counter(10, 10, fresh=False))["state"],
            "insufficient_evidence",
        )

    def test_missing_binding_cannot_prove_alive_engine(self) -> None:
        # The model supplies no bound process when optional title evidence is
        # absent. This checks fail-closed composition, not title collection.
        self.assertEqual(
            verdict("flat", process_supplied=False)["verdict"], "undetermined"
        )

    def test_pid_start_identity_change_outranks_flat_progress(self) -> None:
        self.assertEqual(
            verdict("flat", identity_stable=False)["verdict"], "process_missing"
        )

    def test_stuck_request_does_not_override_its_progressing_engine(self) -> None:
        client = evaluate_producer(
            "client_request",
            {
                "producer_available": True,
                "evaluation_start_ns": 0,
                "evaluation_end_ns": 100,
                "request_started_ns": 0,
                "request_completed_ns": None,
                "chunks": [],
            },
        )
        engine = evaluate_producer("server_counter", counter(10, 11))
        result = verdict(engine["state"], corroborating_state=client["state"])
        self.assertEqual(client["state"], "flat")
        self.assertEqual(result["verdict"], "progress_observed")
        self.assertTrue(result["producer_conflict"])


if __name__ == "__main__":
    unittest.main()
