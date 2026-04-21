"""Compare two local report files (summary-level delta).

Bench reports: compares per-model ``overall_score`` and ``dimension_averages``.
Text / RAG / Agent reports: compares ``metric_averages``.

For per-item (bench) regression detection, use ``evaly gate --baseline``
instead -- item-level delta is a gate concern, not a compare concern.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
from rich.table import Table

from ..exceptions import ValidationError
from ..report.terminal import console


def _load_report(path: str) -> dict[str, Any]:
    report_path = Path(path)
    if not report_path.exists():
        raise ValidationError(f"Report file not found: {path}")
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid JSON in {path}: {exc}") from exc
    if "summary" not in data:
        raise ValidationError(f"Invalid report format: missing 'summary' in {path}")
    return data


@click.command("compare")
@click.option("--baseline", required=True, help="Baseline report JSON.")
@click.option("--candidate", required=True, help="Candidate report JSON.")
@click.option("--json-output", default=None, help="Write structured diff JSON to file.")
def compare_cmd(baseline: str, candidate: str, json_output: str | None) -> None:
    """Compare two report files of the same eval type."""
    try:
        base = _load_report(baseline)
        cand = _load_report(candidate)
        base_type = base.get("eval_type", "bench")
        cand_type = cand.get("eval_type", "bench")
        if base_type != cand_type:
            raise ValidationError(
                f"Cannot compare different report types: {base_type!r} vs {cand_type!r}."
            )
        diff = _compare_reports(base, cand, base_type)
    except ValidationError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(2)

    if json_output:
        payload = json.dumps(diff, indent=2) + "\n"
        if json_output == "-":
            print(payload, end="")
        else:
            Path(json_output).write_text(payload, encoding="utf-8")
            console.print(f"  Compare output written to: {json_output}")
        return

    console.print()
    table = Table(title=f"Compare ({diff['eval_type']})", show_lines=True)
    table.add_column("Key", style="bold")
    table.add_column("Baseline", justify="center")
    table.add_column("Candidate", justify="center")
    table.add_column("Delta", justify="center")
    for row in diff["rows"]:
        table.add_row(row["key"], f"{row['baseline']:.4f}", f"{row['candidate']:.4f}", f"{row['delta']:+.4f}")
    console.print(table)


def _compare_reports(base: dict[str, Any], cand: dict[str, Any], eval_type: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if eval_type == "bench":
        base_summary = base["summary"]
        cand_summary = cand["summary"]
        for model_name in sorted(set(base_summary) & set(cand_summary)):
            rows.append(
                {
                    "key": f"{model_name}:overall_score",
                    "baseline": float(base_summary[model_name].get("overall_score", 0.0)),
                    "candidate": float(cand_summary[model_name].get("overall_score", 0.0)),
                    "delta": float(cand_summary[model_name].get("overall_score", 0.0))
                    - float(base_summary[model_name].get("overall_score", 0.0)),
                }
            )
            base_dims = base_summary[model_name].get("dimension_averages", {})
            cand_dims = cand_summary[model_name].get("dimension_averages", {})
            for dim in sorted(set(base_dims) & set(cand_dims)):
                rows.append(
                    {
                        "key": f"{model_name}:{dim}",
                        "baseline": float(base_dims.get(dim, 0.0)),
                        "candidate": float(cand_dims.get(dim, 0.0)),
                        "delta": float(cand_dims.get(dim, 0.0)) - float(base_dims.get(dim, 0.0)),
                    }
                )
    else:
        base_metrics = base["summary"].get("metric_averages", {})
        cand_metrics = cand["summary"].get("metric_averages", {})
        for metric_id in sorted(set(base_metrics) & set(cand_metrics)):
            rows.append(
                {
                    "key": metric_id,
                    "baseline": float(base_metrics.get(metric_id, 0.0)),
                    "candidate": float(cand_metrics.get(metric_id, 0.0)),
                    "delta": float(cand_metrics.get(metric_id, 0.0)) - float(base_metrics.get(metric_id, 0.0)),
                }
            )

    return {"eval_type": eval_type, "rows": rows}
