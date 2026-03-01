"""JSON report serialization for bench reports."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..bench.types import BenchReport


def report_to_dict(report: BenchReport) -> dict[str, Any]:
    """Convert a BenchReport to the JSON schema dict."""
    d: dict[str, Any] = {
        "$schema": "https://evalytic.ai/schemas/bench-report-v1.json",
        "version": "1.0",
        "name": report.name,
        "created_at": report.created_at,
        "duration_seconds": report.duration_seconds,
        "config": report.config,
        "items": [item.to_dict() for item in report.items],
        "summary": {m: s.to_dict() for m, s in report.summary.items()},
        "ranking": [[m, s] for m, s in report.ranking],
        "efficiency_ranking": [[m, s] for m, s in report.efficiency_ranking],
        "winner": report.winner,
        "best_value": report.best_value,
        "correlation": {
            c.metric_pair: c.to_dict() for c in report.correlations
        },
        "cost": report.cost.to_dict(),
        "metadata": report.metadata,
    }
    if report.consensus_mode:
        d["judges"] = report.judges
        d["consensus_mode"] = True
    if report.errors:
        d["errors"] = [e.to_dict() for e in report.errors]
    return d


def write_json(report: BenchReport, path: str) -> None:
    """Write a bench report to a JSON file."""
    data = report_to_dict(report)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
