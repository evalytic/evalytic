"""`evalytic agent` command group."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import click
from rich.table import Table

from ..agent.runner import evaluate_agent
from ..agent.types import AgentTestCase, AgentToolCall
from ..exceptions import EvalyticError, ValidationError
from ..report.terminal import console


def _resolve_agent_judge(config: dict[str, Any] | None = None) -> str:
    cfg = (config or {}).get("agent", {})
    if "judge" in cfg:
        return cfg["judge"]
    if os.environ.get("FAL_KEY") and not os.environ.get("GEMINI_API_KEY"):
        return "fal/gemini-2.5-flash"
    return "gemini-2.5-flash"


@click.group("agent")
def agent_group() -> None:
    """Evaluate tool-using agent runs."""


@agent_group.command("eval")
@click.option("--input", "input_text", required=True, help="User task for the agent.")
@click.option("--final-output", required=True, help="Final agent output.")
@click.option("--expected-output", default=None, help="Optional expected final output.")
@click.option("--tool-call", "tool_calls", multiple=True, help="Observed tool call name. Repeat for multiple calls.")
@click.option("--expected-tool", "expected_tools", multiple=True, help="Expected tool call name. Repeat for multiple calls.")
@click.option("--expected-max-steps", type=int, default=None, help="Optional expected maximum tool steps.")
@click.option("--judge", "-j", default=None, help="Judge model.")
@click.option("--judges", default=None, help="Comma-separated judges for consensus mode.")
@click.option("--judge-url", default=None, help="Custom judge API base URL.")
@click.option("--output", "-o", default=None, help="Write report JSON to file.")
@click.pass_context
def agent_eval_cmd(
    ctx: click.Context,
    input_text: str,
    final_output: str,
    expected_output: str | None,
    tool_calls: tuple[str, ...],
    expected_tools: tuple[str, ...],
    expected_max_steps: int | None,
    judge: str | None,
    judges: str | None,
    judge_url: str | None,
    output: str | None,
) -> None:
    """Evaluate a single agent run."""
    metadata: dict[str, Any] = {}
    if expected_tools:
        metadata["expected_tool_calls"] = list(expected_tools)
    if expected_max_steps is not None:
        metadata["expected_max_steps"] = expected_max_steps

    case = AgentTestCase(
        input=input_text,
        final_output=final_output,
        expected_output=expected_output,
        tool_calls=[AgentToolCall(name=name) for name in tool_calls],
        metadata=metadata or None,
    )

    try:
        if judge is None:
            judge = _resolve_agent_judge((ctx.obj or {}).get("config"))
        judge_names = [part.strip() for part in (judges or "").split(",") if part.strip()]
        report = evaluate_agent(
            [case],
            judge=judge,
            judges=judge_names or None,
            base_url=judge_url,
            config=(ctx.obj or {}).get("config"),
        )
    except EvalyticError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(2)

    table = Table(title="Agent Metric Averages", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Average", justify="center")
    for metric_id, score in report.metric_averages().items():
        table.add_row(metric_id, f"{score:.4f}")
    console.print()
    console.print(table)
    if output:
        Path(output).write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
        console.print(f"\n  Report written to: {output}")
