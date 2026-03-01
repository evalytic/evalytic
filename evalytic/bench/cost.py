"""Cost estimation and tracking for bench runs."""

from __future__ import annotations

from .registry import resolve_model
from .types import CostBreakdown


# Judge cost per scored image (USD).
JUDGE_COSTS: dict[str, float] = {
    "gemini-2.0-flash": 0.0003,
    "gemini-2.5-flash": 0.0005,
    "gemini-2.5-pro": 0.005,
    "gemini-3-flash": 0.0005,
    "gemini-3.1-pro": 0.002,
    "gpt-4o": 0.005,
    "gpt-4o-mini": 0.001,
    "gpt-5.2": 0.002,
    "claude-sonnet-4": 0.005,
    "claude-sonnet-4-6": 0.003,
    "claude-haiku": 0.001,
    "claude-haiku-4-5": 0.001,
    "claude-opus-4-6": 0.005,
}


def _resolve_judge_model(judge: str) -> str:
    """Extract the model name from a judge string for cost lookup.

    ``"openai/gpt-5.2"`` -> ``"gpt-5.2"``, ``"gemini-2.5-flash"`` -> ``"gemini-2.5-flash"``.
    """
    if "/" in judge:
        return judge.split("/", 1)[1]
    return judge


def estimate_run_cost(
    models: list[str],
    item_count: int,
    judge: str = "gemini-2.5-flash",
    dimensions_per_item: int = 2,
    judges: list[str] | None = None,
) -> CostBreakdown:
    """Pre-run cost estimate based on model registry prices.

    When *judges* is provided (consensus mode), estimates ~2.3x judge cost
    (2 primary judges always + tiebreaker ~30% of the time).
    """
    gen_by_model: dict[str, float] = {}
    gen_total = 0.0
    for m in models:
        entry = resolve_model(m)
        cost = entry.cost_per_image * item_count
        gen_by_model[m] = round(cost, 4)
        gen_total += cost

    total_images = len(models) * item_count
    base_requests = total_images * dimensions_per_item

    if judges and len(judges) >= 2:
        # Consensus mode: 2 primary + ~30% tiebreaker
        judge_total = 0.0
        judge_cost_by_provider: dict[str, float] = {}
        for j in judges[:2]:
            jcost = JUDGE_COSTS.get(_resolve_judge_model(j), 0.005) * base_requests
            judge_cost_by_provider[j] = round(jcost, 4)
            judge_total += jcost
        if len(judges) >= 3:
            # Tiebreaker called ~30% of the time
            tb = judges[2]
            tb_cost = JUDGE_COSTS.get(_resolve_judge_model(tb), 0.005) * base_requests * 0.3
            judge_cost_by_provider[tb] = round(tb_cost, 4)
            judge_total += tb_cost
        judge_requests = int(base_requests * (2.3 if len(judges) >= 3 else 2.0))
        judge_label = "consensus:" + ",".join(judges)
        total = gen_total + judge_total
        return CostBreakdown(
            generation_total_usd=round(gen_total, 4),
            generation_by_model=gen_by_model,
            generation_request_count=total_images,
            judge_total_usd=round(judge_total, 4),
            judge_provider=judge_label,
            judge_request_count=judge_requests,
            judge_providers=judges,
            judge_cost_by_provider=judge_cost_by_provider,
            metrics_total_usd=0.0,
            total_usd=round(total, 4),
        )

    judge_cost_per = JUDGE_COSTS.get(_resolve_judge_model(judge), 0.005)
    judge_requests = base_requests
    judge_total = judge_cost_per * judge_requests

    total = gen_total + judge_total
    return CostBreakdown(
        generation_total_usd=round(gen_total, 4),
        generation_by_model=gen_by_model,
        generation_request_count=total_images,
        judge_total_usd=round(judge_total, 4),
        judge_provider=judge,
        judge_request_count=judge_requests,
        metrics_total_usd=0.0,
        total_usd=round(total, 4),
    )


def build_cost_breakdown(
    generation_costs: dict[str, float],
    generation_count: int,
    judge_request_count: int,
    judge: str = "gemini-2.5-flash",
    judges: list[str] | None = None,
    judge_request_counts: dict[str, int] | None = None,
) -> CostBreakdown:
    """Build a CostBreakdown from actual run data."""
    gen_total = sum(generation_costs.values())

    if judges and len(judges) >= 2 and judge_request_counts:
        judge_total = 0.0
        judge_cost_by_provider: dict[str, float] = {}
        for j in judges:
            jcount = judge_request_counts.get(j, 0)
            jcost = JUDGE_COSTS.get(_resolve_judge_model(j), 0.005) * jcount
            judge_cost_by_provider[j] = round(jcost, 4)
            judge_total += jcost
        judge_label = "consensus:" + ",".join(judges)
        total = gen_total + judge_total
        return CostBreakdown(
            generation_total_usd=round(gen_total, 4),
            generation_by_model={k: round(v, 4) for k, v in generation_costs.items()},
            generation_request_count=generation_count,
            judge_total_usd=round(judge_total, 4),
            judge_provider=judge_label,
            judge_request_count=judge_request_count,
            judge_providers=judges,
            judge_cost_by_provider=judge_cost_by_provider,
            metrics_total_usd=0.0,
            total_usd=round(total, 4),
        )

    judge_cost_per = JUDGE_COSTS.get(_resolve_judge_model(judge), 0.005)
    judge_total = judge_cost_per * judge_request_count
    total = gen_total + judge_total
    return CostBreakdown(
        generation_total_usd=round(gen_total, 4),
        generation_by_model={k: round(v, 4) for k, v in generation_costs.items()},
        generation_request_count=generation_count,
        judge_total_usd=round(judge_total, 4),
        judge_provider=judge,
        judge_request_count=judge_request_count,
        metrics_total_usd=0.0,
        total_usd=round(total, 4),
    )


def format_cost(usd: float) -> str:
    """Format a USD amount for display."""
    if usd == 0.0:
        return "$0.00"
    if usd < 0.01:
        return f"${usd:.4f}"
    return f"${usd:.2f}"
