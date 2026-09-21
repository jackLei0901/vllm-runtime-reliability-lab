from __future__ import annotations

import copy
import json
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any

from test_bundle import native_observations

from dfxlab.bundle import BundleError, expected_summary, write_bundle
from dfxlab.verify_bundle import verify_bundle

ROOT = Path(__file__).resolve().parents[1]
ROLE_REGISTRY = ROOT / "schema" / "collect_verify_v0_2_field_roles.json"


def _leaf_paths(value: Any, prefix: str) -> set[str]:
    if isinstance(value, dict):
        paths: set[str] = set()
        for key, nested in value.items():
            paths.update(_leaf_paths(nested, f"{prefix}.{key}"))
        return paths
    if isinstance(value, list):
        paths = set()
        for nested in value:
            paths.update(_leaf_paths(nested, f"{prefix}[]"))
        return paths
    return {prefix}


def _complete_observations() -> dict[str, Any]:
    observations = native_observations(progressing=True)
    start = observations["intervals"]["evaluation"]["start_ns"]
    end = observations["intervals"]["evaluation"]["end_ns"]
    observations["producer_inputs"]["server_counter"] = {
        "producer_available": True,
        "evaluation_start_ns": start,
        "evaluation_end_ns": end,
        "minimum_span_ns": end - start,
        "minimum_fresh_samples": 2,
        "samples": [
            {"monotonic_ns": start, "fresh": True, "value": 1},
            {"monotonic_ns": end, "fresh": True, "value": 2},
        ],
    }
    observations["demand_inputs"]["server_counter"]["samples"] = [
        {
            "monotonic_ns": start,
            "fresh": True,
            "running": 1,
            "waiting": 0,
        },
        {
            "monotonic_ns": end,
            "fresh": True,
            "running": 1,
            "waiting": 0,
        },
    ]
    observations["stack"] = {
        "state": "unavailable",
        "producer": {
            "sampler_name": "py-spy",
            "sampler_version": "0.4.0",
            "binary_sha256": "b" * 64,
            "platform": "Linux",
            "yama_ptrace_scope": 1,
            "exit_status": 1,
            "output_produced": False,
            "raw_output_sha256": None,
            "error_kind": "permission_denied",
        },
    }
    return observations


def _stack_unavailable(observations: dict[str, Any]) -> None:
    observations["stack"] = _complete_observations()["stack"]


class FieldRoleAuditTest(unittest.TestCase):
    def test_registry_covers_every_public_leaf_exactly_once(self) -> None:
        registry = json.loads(ROLE_REGISTRY.read_text(encoding="utf-8"))
        self.assertEqual(
            {"schema_version", "scope", "decisional", "non_decisional"},
            set(registry),
        )
        decisional = registry["decisional"]
        non_decisional = registry["non_decisional"]
        self.assertEqual(decisional, sorted(set(decisional)))
        self.assertEqual(non_decisional, sorted(set(non_decisional)))
        self.assertFalse(set(decisional) & set(non_decisional))

        observations = _complete_observations()
        summary = expected_summary(observations, "a" * 64)
        actual = _leaf_paths(observations, "observations") | _leaf_paths(
            summary, "summary"
        )
        self.assertEqual(actual, set(decisional) | set(non_decisional))

    def test_schema_valid_non_decisional_variants_preserve_verdict(self) -> None:
        def change_pid(value: dict[str, Any]) -> None:
            value["targets"]["process"]["pid"] = 54321

        def change_endpoint_id(value: dict[str, Any]) -> None:
            value["targets"]["endpoint"]["endpoint_id"] = "sha256:fedcba987654"

        def change_request_digest(value: dict[str, Any]) -> None:
            value["producer_inputs"]["client_request_sha256"] = "b" * 64

        def expand_collection_interval(value: dict[str, Any]) -> None:
            value["intervals"]["collection"]["start_ns"] -= 1
            value["intervals"]["collection"]["end_ns"] += 1

        def change_healthy_status(value: dict[str, Any]) -> None:
            for sample in value["health"]["samples"]:
                sample["status"] = 204

        mutations: dict[str, Callable[[dict[str, Any]], None]] = {
            "pid": change_pid,
            "endpoint_id": change_endpoint_id,
            "request_digest": change_request_digest,
            "collection_interval": expand_collection_interval,
            "healthy_status": change_healthy_status,
            "stack_availability": _stack_unavailable,
        }
        baseline = native_observations()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline_dir = root / "baseline"
            baseline_verdict = write_bundle(baseline_dir, baseline)["verdict"]
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    observations = copy.deepcopy(baseline)
                    mutate(observations)
                    bundle = root / name
                    write_bundle(bundle, observations)
                    self.assertEqual(
                        baseline_verdict,
                        verify_bundle(bundle)["verdict"],
                    )

    def test_producer_implementation_field_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_bundle(root, native_observations())
            path = root / "observations.json"
            observations = json.loads(path.read_text(encoding="utf-8"))
            observations["producer_inputs"]["client_request"]["implementation"] = (
                "example-client"
            )
            path.write_text(json.dumps(observations), encoding="utf-8")
            with self.assertRaisesRegex(BundleError, "observation structure invalid"):
                verify_bundle(root)

    def test_inconsistent_health_provenance_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            observations = native_observations()
            observations["health"]["samples"][0]["status"] = 503
            write_bundle(root, observations)
            with self.assertRaisesRegex(BundleError, "health status inconsistent"):
                verify_bundle(root)


if __name__ == "__main__":
    unittest.main()
