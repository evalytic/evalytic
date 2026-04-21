"""`evalytic rag` command group."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import click
from rich.table import Table

from ..exceptions import EvalyticError, ValidationError
from ..report.terminal import console
from ..text.runner import evaluate_rag, load_cases_from_dataset
from ..text.types import RAGTestCase, RetrievedChunk


def _resolve_rag_judge(config: dict[str, Any] | None = None) -> str:
    cfg = (config or {}).get("rag", {})
    if "judge" in cfg:
        return cfg["judge"]
    if os.environ.get("FAL_KEY") and not os.environ.get("GEMINI_API_KEY"):
        return "fal/gemini-2.5-flash"
    return "gemini-2.5-flash"


@click.group("rag")
def rag_group() -> None:
    """Evaluate RAG answers and retrieval quality."""


@rag_group.command("eval")
@click.option("--query", default=None, help="Original user question.")
@click.option("--response", default=None, help="Model answer to evaluate.")
@click.option("--context", "contexts", multiple=True, help="Retrieved context chunk. Repeat for multiple chunks.")
@click.option("--reference", default=None, help="Optional reference answer for reference-based metrics.")
@click.option("--dataset", default=None, help="Path to a RAG dataset JSON file.")
@click.option("--metrics", default="faithfulness,answer_relevancy", help="Comma-separated metric IDs.")
@click.option("--judge", "-j", default=None, help="Judge model.")
@click.option("--judges", default=None, help="Comma-separated judges for consensus mode.")
@click.option("--judge-url", default=None, help="Custom judge API base URL.")
@click.option("--output", "-o", default=None, help="Write report JSON to file.")
@click.pass_context
def rag_eval_cmd(
    ctx: click.Context,
    query: str | None,
    response: str | None,
    contexts: tuple[str, ...],
    reference: str | None,
    dataset: str | None,
    metrics: str,
    judge: str | None,
    judges: str | None,
    judge_url: str | None,
    output: str | None,
) -> None:
    """Evaluate one or more RAG answers."""
    try:
        if dataset:
            cases = load_cases_from_dataset(dataset, "rag")
        else:
            if not query or not response or not contexts:
                raise ValidationError("--query, --response, and at least one --context are required without --dataset.")
            cases = [
                RAGTestCase(
                    query=query,
                    response=response,
                    contexts=[
                        RetrievedChunk(text=text, rank=index + 1)
                        for index, text in enumerate(contexts)
                    ],
                    reference=reference,
                )
            ]

        if judge is None:
            judge = _resolve_rag_judge((ctx.obj or {}).get("config"))
        judge_names = [part.strip() for part in (judges or "").split(",") if part.strip()]
        report = evaluate_rag(
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
    summary = Table(title="RAG Metric Averages", show_lines=True)
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
