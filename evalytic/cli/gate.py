"""``evalytic gate`` command -- CI/CD quality gate with exit codes."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
from rich.table import Table

from ..exceptions import EvalyticError, ValidationError
from ..report.terminal import console, score_color


def _load_report(path: str) -> dict[str, Any]:
    """Load and validate a bench report JSON file."""
    p = Path(path)
    if not p.exists():
        raise ValidationError(f"Report file not found: {path}")
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid JSON in {path}: {exc}") from exc
    if "summary" not in data:
        raise ValidationError(f"Invalid report format: missing 'summary' in {path}")
    return data


def _parse_dimension_thresholds(values: tuple[str, ...]) -> dict[str, float]:
    """Parse 'dim:val' pairs into a dict."""
    result: dict[str, float] = {}
    for v in values:
        if ":" not in v:
            raise ValidationError(
                f"Invalid dimension threshold format: {v!r}. Expected 'dimension:value' (e.g. visual_quality:4.0)"
            )
        dim, val_str = v.split(":", 1)
        try:
            result[dim.strip()] = float(val_str.strip())
        except ValueError:
            raise ValidationError(f"Invalid threshold value: {val_str!r} in {v!r}")
    return result


def _build_item_index(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build item_id -> item dict from report items list."""
    result: dict[str, dict[str, Any]] = {}
    for item in data.get("items", []):
        item_id = item.get("item_id", "")
        if item_id:
            result[item_id] = item
    return result


def _extract_scores(scores_data: Any) -> dict[str, float]:
    """Extract dimension -> score from either dict or list format.

    Dict format: {"visual_quality": {"score": 5.0, ...}}
    List format: [{"dimension": "visual_quality", "score": 5.0}, ...]
    """
    result: dict[str, float] = {}
    if isinstance(scores_data, dict):
        for dim, val in scores_data.items():
            if isinstance(val, dict):
                result[dim] = val.get("score", 0.0)
            elif isinstance(val, (int, float)):
                result[dim] = float(val)
    elif isinstance(scores_data, list):
        for s in scores_data:
            dim = s.get("dimension", "")
            if dim:
                result[dim] = s.get("score", 0.0)
    return result


@click.command()
@click.option("--report", required=True, help="Path to bench report JSON file.")
@click.option("--threshold", type=float, default=None, help="Minimum overall score to pass.")
@click.option(
    "--dimension-threshold", multiple=True,
    help="Per-dimension threshold as dim:value (e.g. visual_quality:4.0).",
)
@click.option("--baseline", default=None, help="Path to baseline report JSON for regression detection.")
@click.option(
    "--regression-threshold", type=float, default=0.3,
    show_default=True,
    help="Max allowed score drop per dimension vs baseline.",
)
@click.option(
    "--min-confidence", type=float, default=None,
    help="Minimum average confidence score (0.0-1.0) to pass.",
)
@click.option(
    "--json-output", "json_output", default=None,
    help="Write gate results as JSON to file (use - for stdout).",
)
def gate(
    report: str,
    threshold: float | None,
    dimension_threshold: tuple[str, ...],
    baseline: str | None,
    regression_threshold: float,
    min_confidence: float | None,
    json_output: str | None,
) -> None:
    """CI/CD quality gate. Exit 0 = pass, 1 = fail, 2 = error."""
    try:
        data = _load_report(report)
    except EvalyticError as exc:
        if json_output:
            _write_json(json_output, {"status": "error", "error": str(exc), "checks": []})
        else:
            console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(2)

    # Parse dimension thresholds
    try:
        dim_thresholds = _parse_dimension_thresholds(dimension_threshold)
    except EvalyticError as exc:
        if json_output:
            _write_json(json_output, {"status": "error", "error": str(exc), "checks": []})
        else:
            console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(2)

    # Load baseline if provided
    baseline_data: dict[str, Any] | None = None
    if baseline:
        try:
            baseline_data = _load_report(baseline)
        except EvalyticError as exc:
            if json_output:
                _write_json(json_output, {"status": "error", "error": str(exc), "checks": []})
            else:
                console.print(f"[bold red]Error:[/bold red] {exc}")
            sys.exit(2)

    # Check if any checks are configured
    has_checks = threshold is not None or dim_thresholds or baseline_data is not None or min_confidence is not None
    if not has_checks:
        if json_output:
            _write_json(json_output, {"status": "pass", "checks": [], "warning": "No checks configured."})
        else:
            console.print("[yellow]Warning:[/yellow] No checks configured. Pass --threshold, --dimension-threshold, or --baseline.")
        sys.exit(0)

    summary = data["summary"]
    all_passed = True
    checks: list[dict[str, Any]] = []

    # Rich table (only used when not json-only)
    results_table = Table(title="Gate Results", show_lines=True)
    results_table.add_column("Model", style="bold")
    results_table.add_column("Check")
    results_table.add_column("Value", justify="center")
    results_table.add_column("Threshold", justify="center")
    results_table.add_column("Result", justify="center")

    for model_name, model_data in summary.items():
        overall = model_data.get("overall_score", 0.0)
        dim_avgs = model_data.get("dimension_averages", {})

        # Overall threshold check
        if threshold is not None:
            passed = overall >= threshold
            if not passed:
                all_passed = False
            checks.append({
                "type": "threshold", "model": model_name,
                "check": "overall_score", "value": overall,
                "threshold": threshold, "passed": passed,
            })
            color = score_color(overall)
            results_table.add_row(
                model_name, "overall_score",
                f"[{color}]{overall:.2f}[/{color}]", f"{threshold:.2f}",
                "[green]PASS[/green]" if passed else "[red]FAIL[/red]",
            )

        # Dimension threshold checks
        for dim, min_val in dim_thresholds.items():
            actual = dim_avgs.get(dim, 0.0)
            passed = actual >= min_val
            if not passed:
                all_passed = False
            checks.append({
                "type": "dimension_threshold", "model": model_name,
                "dimension": dim, "value": actual,
                "threshold": min_val, "passed": passed,
            })
            color = score_color(actual)
            results_table.add_row(
                model_name, dim,
                f"[{color}]{actual:.2f}[/{color}]", f"{min_val:.2f}",
                "[green]PASS[/green]" if passed else "[red]FAIL[/red]",
            )

        # Confidence check
        if min_confidence is not None:
            avg_conf = model_data.get("avg_confidence", 1.0)
            passed = avg_conf >= min_confidence
            if not passed:
                all_passed = False
            checks.append({
                "type": "confidence", "model": model_name,
                "value": avg_conf, "threshold": min_confidence, "passed": passed,
            })
            conf_color = "green" if avg_conf >= 0.8 else ("yellow" if avg_conf >= 0.5 else "red")
            results_table.add_row(
                model_name, "avg_confidence",
                f"[{conf_color}]{avg_conf:.2f}[/{conf_color}]", f"{min_confidence:.2f}",
                "[green]PASS[/green]" if passed else "[red]FAIL[/red]",
            )

        # Regression checks (average)
        if baseline_data is not None:
            baseline_summary = baseline_data.get("summary", {})
            baseline_model = baseline_summary.get(model_name)
            if baseline_model:
                baseline_dims = baseline_model.get("dimension_averages", {})
                for dim, current_val in dim_avgs.items():
                    baseline_val = baseline_dims.get(dim)
                    if baseline_val is not None:
                        drop = baseline_val - current_val
                        passed = drop <= regression_threshold
                        if not passed:
                            all_passed = False
                        checks.append({
                            "type": "regression", "model": model_name,
                            "dimension": dim, "baseline": baseline_val,
                            "current": current_val, "drop": round(drop, 2),
                            "threshold": regression_threshold, "passed": passed,
                        })
                        color = "green" if drop <= 0 else ("yellow" if drop <= regression_threshold else "red")
                        results_table.add_row(
                            model_name, f"regression:{dim}",
                            f"[{color}]{drop:+.2f}[/{color}]", f"{regression_threshold:.2f}",
                            "[green]PASS[/green]" if passed else "[red]FAIL[/red]",
                        )

    # Per-item regression checks
    item_regressions: list[dict[str, Any]] = []
    if baseline_data is not None:
        current_items = _build_item_index(data)
        baseline_items = _build_item_index(baseline_data)

        for item_id, current_item in current_items.items():
            baseline_item = baseline_items.get(item_id)
            if baseline_item is None:
                continue

            for model_name in summary:
                current_result = current_item.get("results", {}).get(model_name)
                baseline_result = baseline_item.get("results", {}).get(model_name)
                if not current_result or not baseline_result:
                    continue
                if current_result.get("status") != "success" or baseline_result.get("status") != "success":
                    continue

                # Build score lookups (handles both dict and list formats)
                current_scores = _extract_scores(current_result.get("scores", {}))
                baseline_scores = _extract_scores(baseline_result.get("scores", {}))

                for dim, current_val in current_scores.items():
                    baseline_val = baseline_scores.get(dim)
                    if baseline_val is None:
                        continue
                    drop = baseline_val - current_val
                    if drop > regression_threshold:
                        all_passed = False
                        entry = {
                            "type": "item_regression", "model": model_name,
                            "item_id": item_id, "dimension": dim,
                            "baseline": baseline_val, "current": current_val,
                            "drop": round(drop, 2), "threshold": regression_threshold,
                            "passed": False,
                        }
                        item_regressions.append(entry)
                        checks.append(entry)

        # Show per-item regressions in terminal (only failed ones)
        if item_regressions:
            results_table.add_section()
            for reg in item_regressions:
                results_table.add_row(
                    reg["model"],
                    f"item:{reg['item_id']}:{reg['dimension']}",
                    f"[red]{reg['drop']:+.2f}[/red]",
                    f"{regression_threshold:.2f}",
                    "[red]FAIL[/red]",
                )

    # Output
    status = "pass" if all_passed else "fail"

    if json_output:
        result = {
            "status": status,
            "checks": checks,
            "summary": {
                "total_checks": len(checks),
                "passed": sum(1 for c in checks if c.get("passed", True)),
                "failed": sum(1 for c in checks if not c.get("passed", True)),
            },
        }
        if item_regressions:
            result["item_regressions"] = len(item_regressions)
        _write_json(json_output, result)
    else:
        console.print()
        console.print(results_table)
        console.print()

    if all_passed:
        if not json_output:
            console.print("[bold green]Gate: PASS[/bold green]")
        sys.exit(0)
    else:
        if not json_output:
            console.print("[bold red]Gate: FAIL[/bold red]")
        sys.exit(1)


def _write_json(path: str, data: dict[str, Any]) -> None:
    """Write JSON to file or stdout."""
    output = json.dumps(data, indent=2)
    if path == "-":
        print(output)
    else:
        Path(path).write_text(output, encoding="utf-8")
