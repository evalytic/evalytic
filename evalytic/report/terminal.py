"""Rich terminal output for bench reports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeElapsedColumn
from rich.table import Table

from ..bench.cost import format_cost

if TYPE_CHECKING:
    from ..bench.types import BenchReport, CorrelationStat, CostBreakdown

console = Console()


def score_color(score: float) -> str:
    """Return a Rich color name for a score value."""
    if score >= 4.0:
        return "green"
    if score >= 3.0:
        return "yellow"
    return "red"


def print_bench_header(
    name: str,
    model_count: int,
    item_count: int,
    pipeline: str,
) -> None:
    """Print the bench run header."""
    item_label = "prompts" if pipeline == "text2img" else "inputs"
    total = model_count * item_count
    console.print()
    console.print(
        Panel(
            f"[bold]{name}[/bold]\n"
            f"  {model_count} model{'s' if model_count != 1 else ''} x "
            f"{item_count} {item_label} = {total} images",
            title="Evalytic Bench",
            border_style="blue",
        )
    )


def print_cost_estimate(cost: CostBreakdown) -> None:
    """Print estimated costs before generation."""
    console.print("\n  [bold]Estimated cost:[/bold]")
    gen_detail = ", ".join(f"{m}: {format_cost(c)}" for m, c in cost.generation_by_model.items())
    console.print(f"    Generation: ~{format_cost(cost.generation_total_usd)}  ({gen_detail})")
    console.print(f"    Judge:      ~{format_cost(cost.judge_total_usd)}  ({cost.judge_provider})")
    console.print(f"    Total:      ~{format_cost(cost.total_usd)}")
    console.print()


def confirm_proceed() -> bool:
    """Ask user to confirm. Returns True if confirmed."""
    try:
        answer = console.input("  Proceed? [Y/n] ").strip().lower()
        return answer in ("", "y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def create_generation_progress() -> Progress:
    """Create a Rich Progress instance for image generation."""
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        console=console,
    )


def create_scoring_progress() -> Progress:
    """Create a Rich Progress instance for VLM scoring."""
    return Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        console=console,
    )


def _has_weighted_scoring(report: BenchReport) -> bool:
    """Check if any model has weighted overall scoring active."""
    if any(ms.weighted_overall_score is not None for ms in report.summary.values()):
        return True
    # Also true when dimension_weights are configured
    return bool(report.config.get("dimension_weights"))


def print_comparison_table(report: BenchReport) -> None:
    """Print the model x dimension comparison table with winner."""
    # Collect metric names present across all summaries
    metric_names: list[str] = []
    for ms in report.summary.values():
        for m in ms.metric_averages:
            if m not in metric_names:
                metric_names.append(m)

    weighted = _has_weighted_scoring(report)

    # Resolve dimension weights for display
    dim_weights: dict[str, float] = report.config.get("dimension_weights", {})

    table = Table(title="Model Comparison", show_lines=True)
    table.add_column("Model", style="bold")
    for dim in report.dimensions:
        label = dim.replace("_", " ").title()
        if dim_weights and dim in dim_weights:
            label += f" ({int(dim_weights[dim] * 100)}%)"
        table.add_column(label, justify="center")
    for m_name in metric_names:
        label = {"clip_score": "CLIP", "lpips": "LPIPS"}.get(m_name, m_name)
        table.add_column(label, justify="center")
    overall_header = "Overall*" if weighted else "Overall"
    table.add_column(overall_header, justify="center", style="bold")
    table.add_column("Avg Time", justify="center")
    table.add_column("Conf", justify="center")
    # Show agreement column in consensus mode
    if report.consensus_mode:
        table.add_column("Agree", justify="center")
    # Show success rate column if any model has failures
    has_failures = any(
        ms.total_items > 0 and ms.item_count < ms.total_items
        for ms in report.summary.values()
        if ms is not None
    )
    if has_failures:
        table.add_column("Success", justify="center")

    for model in report.models:
        ms = report.summary.get(model)
        if ms is None:
            continue
        row: list[str] = [model]
        for dim in report.dimensions:
            avg = ms.dimension_averages.get(dim, 0.0)
            color = score_color(avg)
            row.append(f"[{color}]{avg:.1f}[/{color}]")
        for m_name in metric_names:
            val = ms.metric_averages.get(m_name, 0.0)
            flag_status = ms.metric_flags.get(m_name)
            if flag_status == "flagged":
                row.append(f"[red]{val:.3f} ⚠[/red]")
            elif flag_status == "included":
                row.append(f"[green]{val:.3f} ✓[/green]")
            else:
                row.append(f"{val:.3f}")
        color = score_color(ms.overall_score)
        sd_txt = f"[dim]±{ms.overall_stddev:.1f}[/dim]" if ms.overall_stddev > 0 and ms.sample_count >= 2 else ""
        row.append(f"[{color}]{ms.overall_score:.1f}[/{color}]{sd_txt}")
        # Avg generation time column
        if ms.item_count > 0 and ms.total_generation_time_ms > 0:
            avg_time_s = ms.total_generation_time_ms / ms.item_count / 1000
            row.append(f"[dim]{avg_time_s:.1f}s[/dim]")
        else:
            row.append("[dim]-[/dim]")
        # Confidence column
        conf_pct = int(ms.avg_confidence * 100)
        conf_color = "green" if ms.avg_confidence >= 0.8 else ("yellow" if ms.avg_confidence >= 0.5 else "red")
        row.append(f"[{conf_color}]{conf_pct}%[/{conf_color}]")
        # Agreement column (consensus mode only)
        if report.consensus_mode:
            agree_pct = int(ms.agreement_rate * 100)
            agree_color = "green" if ms.agreement_rate >= 0.8 else ("yellow" if ms.agreement_rate >= 0.5 else "red")
            row.append(f"[{agree_color}]{agree_pct}%[/{agree_color}]")
        # Success rate column (only if any model has failures)
        if has_failures:
            if ms.total_items > 0:
                rate = ms.item_count / ms.total_items
                rate_color = "green" if rate == 1.0 else ("yellow" if rate >= 0.5 else "red")
                row.append(f"[{rate_color}]{ms.item_count}/{ms.total_items}[/{rate_color}]")
            else:
                row.append("-")
        table.add_row(*row)

    console.print()
    console.print(table)
    if report.winner and report.ranking:
        winner_score = report.ranking[0][1]
        ws = report.summary.get(report.winner)
        n_txt = f", n={ws.sample_count}" if ws and ws.sample_count > 0 else ""
        console.print(
            f"\n  Winner: [bold green]{report.winner}[/bold green] ({winner_score:.1f}/5{n_txt})"
        )
        # Small sample warning
        if ws and ws.sample_count < 10:
            console.print(
                f"  [dim]Note: Small sample (n={ws.sample_count}). "
                f"Scores may vary with more prompts.[/dim]"
            )
        console.print()


def print_metric_flags(report: BenchReport) -> None:
    """Print warning lines for flagged metrics."""
    has_flags = False
    for model in report.models:
        ms = report.summary.get(model)
        if ms is None:
            continue
        for m_name, status in ms.metric_flags.items():
            if status == "flagged":
                if not has_flags:
                    console.print("  [bold yellow]Metric Warnings:[/bold yellow]")
                    has_flags = True
                val = ms.metric_averages.get(m_name, 0.0)
                label = {"clip_score": "CLIP score", "lpips": "LPIPS similarity"}.get(m_name, m_name)
                console.print(
                    f"    [yellow]⚠[/yellow] {model}: {label} ({val:.4f}) below threshold — excluded from overall"
                )
    if has_flags:
        console.print()


def print_correlation(correlations: list[CorrelationStat]) -> None:
    """Print CLIP-VLM or LPIPS-VLM correlation statistics."""
    if not correlations:
        return

    for corr in correlations:
        label_map = {
            "high_agreement": "[green]high agreement[/green]",
            "moderate": "[yellow]moderate[/yellow]",
            "low_agreement": "[red]low agreement[/red]",
        }
        interp = label_map.get(corr.interpretation, corr.interpretation)
        pair = corr.metric_pair.replace("_vs_", " vs ").replace("_", " ")
        console.print(
            f"  {pair}: r={corr.pearson_r:.2f} ({interp})"
        )
    console.print()


def print_efficiency_table(report: BenchReport) -> None:
    """Print a quality-vs-cost efficiency ranking.

    Shows both raw Score/$ and contextual "vs Winner" comparison
    so readers can judge the quality–cost trade-off.
    """
    if not report.efficiency_ranking:
        return

    # Find winner stats for relative comparison
    winner_ms = report.summary.get(report.winner) if report.winner else None

    table = Table(title="Cost Efficiency", show_lines=True)
    table.add_column("Rank", justify="center")
    table.add_column("Model", style="bold")
    table.add_column("Score", justify="center")
    table.add_column("Cost/Image", justify="right")
    table.add_column("Score/$", justify="right")
    table.add_column("vs Winner", justify="center")
    table.add_column("", justify="left")

    best_spd = report.efficiency_ranking[0][1] if report.efficiency_ranking else 0

    for i, (model, spd) in enumerate(report.efficiency_ranking, 1):
        ms = report.summary.get(model)
        if ms is None:
            continue
        score_col = score_color(ms.overall_score)
        badge = ""
        if model == report.best_value and model != report.winner:
            badge = "[green]BEST VALUE[/green]"
        elif model == report.winner:
            badge = "[blue]WINNER[/blue]"

        # "vs Winner" column: quality gap and cost comparison
        if winner_ms and model != report.winner and winner_ms.cost_per_image > 0:
            score_gap = ms.overall_score - winner_ms.overall_score
            cost_ratio = ms.cost_per_image / winner_ms.cost_per_image
            gap_color = "green" if score_gap >= 0 else ("yellow" if score_gap > -0.5 else "red")
            if cost_ratio > 1:
                cost_txt = f"[red]{cost_ratio:.0f}x cost[/red]"
            else:
                pct = (1 - cost_ratio) * 100
                cost_txt = f"[green]{pct:.0f}% cheaper[/green]"
            vs_txt = f"[{gap_color}]{score_gap:+.1f}[/{gap_color}] {cost_txt}"
        elif model == report.winner:
            vs_txt = "[dim]—[/dim]"
        else:
            vs_txt = "-"

        # Bar showing relative efficiency
        bar_pct = int(spd / best_spd * 100) if best_spd > 0 else 0
        bar = f"[dim]{'█' * (bar_pct // 5)}[/dim]"
        table.add_row(
            str(i),
            model,
            f"[{score_col}]{ms.overall_score:.1f}[/{score_col}]",
            f"${ms.cost_per_image:.4f}",
            f"{spd:.0f}",
            vs_txt,
            f"{badge} {bar}",
        )

    console.print(table)

    if report.best_value and report.best_value != report.winner:
        bv = report.summary.get(report.best_value)
        w = report.summary.get(report.winner)
        if bv and w and w.cost_per_image > 0:
            savings = (1 - bv.cost_per_image / w.cost_per_image) * 100
            score_diff = w.overall_score - bv.overall_score
            console.print(
                f"\n  [bold]{report.best_value}[/bold] is [green]{savings:.0f}% cheaper[/green] "
                f"than {report.winner} with only {score_diff:.1f} points less quality\n"
            )
    elif report.best_value:
        console.print(
            f"\n  [bold green]{report.best_value}[/bold green] is both highest quality and best value\n"
        )


def print_cost_summary(report: BenchReport) -> None:
    """Print the cost breakdown table."""
    cost = report.cost
    table = Table(title="Cost Summary", show_lines=True)
    table.add_column("Category", style="bold")
    table.add_column("Requests", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Details")

    gen_detail = ", ".join(f"{m}: {format_cost(c)}" for m, c in cost.generation_by_model.items())
    table.add_row(
        "fal.ai generation",
        str(cost.generation_request_count),
        format_cost(cost.generation_total_usd),
        gen_detail,
    )
    if cost.judge_providers and cost.judge_cost_by_provider:
        # Consensus mode: show per-provider costs
        provider_detail = ", ".join(
            f"{j}: {format_cost(c)}" for j, c in cost.judge_cost_by_provider.items()
        )
        table.add_row(
            "Consensus judge",
            str(cost.judge_request_count),
            format_cost(cost.judge_total_usd),
            provider_detail,
        )
    elif cost.judge_provider:
        table.add_row(
            f"{cost.judge_provider} judge",
            str(cost.judge_request_count),
            format_cost(cost.judge_total_usd),
            "",
        )
    else:
        # No judge (--no-judge mode)
        table.add_row(
            "VLM judge",
            "0",
            format_cost(0.0),
            "Skipped (--no-judge)",
        )
    # Metrics row (always $0.00 — local computation)
    has_metrics = any(
        ms.metric_averages for ms in report.summary.values()
    )
    if has_metrics:
        table.add_row(
            "Local metrics",
            "",
            format_cost(cost.metrics_total_usd),
            "Local computation",
        )
    table.add_row(
        "[bold]Total[/bold]",
        "",
        f"[bold]{format_cost(cost.total_usd)}[/bold]",
        f"Duration: {report.duration_seconds:.0f}s",
    )

    console.print(table)
    console.print()


def print_error_summary(report: BenchReport) -> None:
    """Print a brief error summary if there were errors during the run."""
    if not report.errors:
        return
    console.print(f"  [yellow]{len(report.errors)} error(s) during this run.[/yellow]")
    for err in report.errors[:5]:
        location = f"{err.model}/{err.item_id}" if err.item_id else err.model
        msg = err.message[:80]
        console.print(f"    [{err.level.upper()}] {err.phase} {location}: {msg}")
    if len(report.errors) > 5:
        console.print(f"    ... and {len(report.errors) - 5} more")
    console.print()


def print_full_report(report: BenchReport) -> None:
    """Print the complete terminal report."""
    print_comparison_table(report)
    print_metric_flags(report)
    if report.correlations:
        print_correlation(report.correlations)
    if len(report.models) > 1:
        print_efficiency_table(report)
    print_cost_summary(report)
    print_error_summary(report)
