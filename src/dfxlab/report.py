from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from dfxlab.external_schema import validate_external_artifact


def _metric_summary(samples: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    values: dict[str, list[float]] = {}
    for sample in samples:
        for name, value in sample.get("metrics", {}).items():
            if value is not None:
                values.setdefault(name, []).append(float(value))
    return {
        name: {"min": min(items), "max": max(items), "mean": mean(items)}
        for name, items in sorted(values.items())
    }


def render_incident_markdown(payload: dict[str, Any]) -> str:
    validate_external_artifact(payload)
    samples = payload["history"]
    health_failures = sum(not sample["health"]["ok"] for sample in samples)
    lines = [
        "# External runtime incident summary",
        "",
        f"- Incident: `{payload['incident_id']}`",
        f"- Trigger: `{payload['trigger']['kind']}`",
        f"- Internal cause: `{payload['trigger']['internal_kind']}`",
        f"- Internal stage: `{payload['trigger']['internal_stage']}`",
        f"- Captured at: `{payload['created_at']}`",
        f"- Samples retained: `{len(samples)}`",
        f"- Unhealthy samples: `{health_failures}`",
        "",
        "## Metric range",
        "",
        "| Metric | Min | Mean | Max |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, stats in _metric_summary(samples).items():
        row = (
            f"| `{name}` | {stats['min']:.6g} | {stats['mean']:.6g} | "
            f"{stats['max']:.6g} |"
        )
        lines.append(row)
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This report contains external observations only. It does not infer an "
            "EngineCore failure kind or stage from timing, health, or process loss.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize_file(input_path: Path, output_path: Path) -> None:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_incident_markdown(payload), encoding="utf-8")
