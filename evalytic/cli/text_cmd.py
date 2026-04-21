"""`evalytic text` command group."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import click
from rich.table import Table

from ..exceptions import EvalyticError, ValidationError
from ..report.terminal import console
from ..text.runner import evaluate_text, load_cases_from_dataset
from ..text.types import TextTestCase


def _resolve_text_judge(config: dict[str, Any] | None = None) -> str:
    cfg = (config or {}).get("text", {})
    if "judge" in cfg:
        return cfg["judge"]
    if os.environ.get("FAL_KEY") and not os.environ.get("GEMINI_API_KEY"):
        return "fal/gemini-2.5-flash"
    return "gemini-2.5-flash"


@click.group("text")
def text_group() -> None:
    """Evaluate text outputs."""


@text_group.command("eval")
@click.option("--input", "input_text", default=None, help="Original prompt or task input.")
@click.option("--output-text", "output_text", default=None, help="Model output to evaluate.")
@click.option("--expected", default=None, help="Optional expected/reference output.")
@click.option("--criteria", default=None, help="Optional free-form evaluation criteria.")
@click.option("--dataset", default=None, help="Path to a text dataset JSON file.")
@click.option("--metrics", default="factual_correctness,semantic_similarity", help="Comma-separated metric IDs.")
@click.option("--judge", "-j", default=None, help="Judge model.")
@click.option("--judges", default=None, help="Comma-separated judges for consensus mode.")
@click.option("--judge-url", default=None, help="Custom judge API base URL.")
@click.option("--output", "-o", default=None, help="Write report JSON to file.")
@click.pass_context
def text_eval_cmd(
    ctx: click.Context,
    input_text: str | None,
    output_text: str | None,
    expected: str | None,
    criteria: str | None,
    dataset: str | None,
    metrics: str,
    judge: str | None,
    judges: str | None,
    judge_url: str | None,
    output: str | None,
) -> None:
    """Evaluate one or more text outputs."""
    try:
        if dataset:
            cases = load_cases_from_dataset(dataset, "text")
        else:
            if not input_text or output_text is None:
                raise ValidationError("--input and --output-text are required without --dataset.")
            cases = [
                TextTestCase(
                    input=input_text,
                    output=output_text,
                    expected=expected,
                    criteria=criteria,
                )
            ]

        if judge is None:
            judge = _resolve_text_judge((ctx.obj or {}).get("config"))
        judge_names = [part.strip() for part in (judges or "").split(",") if part.strip()]
        report = evaluate_text(
            cases,
            metric_ids=[metric.strip() for metric in metrics.split(",") if metric.strip()],
            judge=judge,
            judges=judge_names or None,
            base_url=judge_url,
            config=(ctx.obj or {}).get("config"),
        )
    except EvalyticError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(2)

    console.print()
    summary = Table(title="Text Metric Averages", show_lines=True)
    summary.add_column("Metric", style="bold")
    summary.add_column("Average", justify="center")
    for metric_id, score in report.metric_averages().items():
        summary.add_row(metric_id, f"{score:.4f}")
    console.print(summary)

    for result in report.results:
        table = Table(title=f"Case {result.case_id}", show_lines=True)
        table.add_column("Metric", style="bold")
        table.add_column("Score", justify="center")
        table.add_column("Reason")
        for metric in result.metrics:
            table.add_row(metric.metric_id, f"{metric.score:.4f}", metric.reason or "")
        console.print(table)

    if output:
        Path(output).write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        console.print(f"\n  Report written to: {output}")
