"""Rich terminal report renderer for ShopLens evaluation results."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .judge import LocalEvalResult

console = Console()


def _score_color(score: float) -> str:
    if score >= 4.0:
        return "green"
    if score >= 3.0:
        return "yellow"
    return "red"


def print_header(pipeline_name: str, description: str) -> None:
    console.print()
    console.print(Panel(
        f"[bold]{pipeline_name}[/bold]\n{description}",
        title="ShopLens",
        border_style="blue",
    ))


def print_model_comparison(results_by_model: dict[str, list[LocalEvalResult]]) -> None:
    """Print a comparison table: model x average dimension scores."""
    if not results_by_model:
        return

    # Gather all dimensions from the first model's results
    sample = next(iter(results_by_model.values()))
    dims = sorted({s.dimension for r in sample for s in r.scores})

    table = Table(title="Model Comparison", show_lines=True)
    table.add_column("Model", style="bold")
    for dim in dims:
        table.add_column(dim.replace("_", " ").title(), justify="center")
    table.add_column("Overall", justify="center", style="bold")

    best_overall = -1.0
    best_model = ""

    for model, results in results_by_model.items():
        row: list[str] = [model]
        dim_avgs: dict[str, float] = {}
        for dim in dims:
            scores = [
                r.dimension(dim).score
                for r in results
                if r.dimension(dim) is not None
            ]
            avg = sum(scores) / len(scores) if scores else 0.0
            dim_avgs[dim] = avg
            color = _score_color(avg)
            row.append(f"[{color}]{avg:.1f}[/{color}]")

        overall = sum(dim_avgs.values()) / len(dim_avgs) if dim_avgs else 0.0
        color = _score_color(overall)
        row.append(f"[{color}]{overall:.1f}[/{color}]")

        if overall > best_overall:
            best_overall = overall
            best_model = model

        table.add_row(*row)

    console.print(table)
    console.print(f"\n  Winner: [bold green]{best_model}[/bold green] ({best_overall:.1f}/5)\n")


def print_dimension_breakdown(results: list[LocalEvalResult], label: str = "") -> None:
    """Print per-image dimension breakdown."""
    if label:
        console.print(f"\n[bold]{label}[/bold]")

    for i, result in enumerate(results, 1):
        table = Table(title=f"Image {i}", show_lines=True)
        table.add_column("Dimension", style="bold")
        table.add_column("Score", justify="center")
        table.add_column("Explanation")

        for s in result.scores:
            color = _score_color(s.score)
            table.add_row(
                s.dimension.replace("_", " ").title(),
                f"[{color}]{s.score:.1f}[/{color}]",
                s.explanation[:80] + "..." if len(s.explanation) > 80 else s.explanation,
            )

        table.add_row(
            "[bold]Overall[/bold]",
            f"[bold]{result.display_score}[/bold]",
            "",
        )
        console.print(table)


def print_regression_alert(regressions: list[dict]) -> None:
    """Highlight dimension regressions in red."""
    if not regressions:
        console.print("[green]No regressions detected.[/green]\n")
        return

    console.print(Panel(
        "[bold red]REGRESSIONS DETECTED[/bold red]",
        border_style="red",
    ))

    for r in regressions:
        delta = r["candidate"] - r["baseline"]
        text = Text()
        text.append(f"  {r['dimension']}: ", style="bold")
        text.append(f"{r['baseline']:.1f}", style="green")
        text.append(" -> ")
        text.append(f"{r['candidate']:.1f}", style="red")
        text.append(f"  ({delta:+.1f})")
        console.print(text)

    console.print()


def print_summary(all_results: dict[str, dict]) -> None:
    """Print consolidated platform health summary."""
    console.print(Panel(
        "[bold]ShopLens Platform Health[/bold]",
        title="Summary",
        border_style="blue",
    ))

    table = Table(show_lines=True)
    table.add_column("Pipeline", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Score", justify="center")
    table.add_column("Details")

    for name, info in all_results.items():
        score = info.get("score", 0.0)
        color = _score_color(score)
        status = info.get("status", "ok")
        status_display = "[green]OK[/green]" if status == "ok" else "[red]ALERT[/red]"
        table.add_row(
            name,
            status_display,
            f"[{color}]{score:.1f}/5[/{color}]",
            info.get("details", ""),
        )

    console.print(table)
    console.print()
