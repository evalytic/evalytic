#!/usr/bin/env python3
"""Pipeline 2: Regression check -- flux-dev-i2i (baseline) vs flux-kontext (candidate).

Runs both models on 4 product photos, compares dimension averages,
and flags any regressions > 0.3 points.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shoplens.config import PIPELINES
from shoplens.generate import generate_img2img
from shoplens.judge import LocalJudge, LocalEvalResult
from shoplens.report import (
    console,
    print_header,
    print_dimension_breakdown,
    print_regression_alert,
)

from golden_sets.product_enhance import PRODUCTS

REGRESSION_THRESHOLD = 0.3


def main() -> dict:
    cfg = PIPELINES["product-enhancer"]
    baseline_model = cfg.default_model
    candidate_model = cfg.challenger_models[0]

    print_header(
        cfg.name,
        f"Regression check: {baseline_model} (baseline) vs {candidate_model} (candidate)",
    )

    judge = LocalJudge()
    results: dict[str, list[LocalEvalResult]] = {baseline_model: [], candidate_model: []}

    for model in [baseline_model, candidate_model]:
        console.print(f"\n[bold cyan]{model}[/bold cyan]")
        for i, product in enumerate(PRODUCTS, 1):
            console.print(f"  [{i}/{len(PRODUCTS)}] {product['product_type']}...", end=" ")
            image_url = generate_img2img(
                image_url=product["image_url"],
                instruction=product["instruction"],
                model_key=model,
            )
            console.print("scoring...", end=" ")
            result = judge.score(
                image_url=image_url,
                dimensions=cfg.dimensions,
                input_image_url=product["image_url"],
            )
            console.print(f"[green]{result.display_score}[/green]")
            results[model].append(result)

    # Print breakdowns
    print_dimension_breakdown(results[baseline_model], label=f"Baseline: {baseline_model}")
    print_dimension_breakdown(results[candidate_model], label=f"Candidate: {candidate_model}")

    # Compute regressions
    regressions: list[dict] = []
    for dim in cfg.dimensions:
        baseline_scores = [
            r.dimension(dim).score
            for r in results[baseline_model]
            if r.dimension(dim) is not None
        ]
        candidate_scores = [
            r.dimension(dim).score
            for r in results[candidate_model]
            if r.dimension(dim) is not None
        ]
        if baseline_scores and candidate_scores:
            b_avg = sum(baseline_scores) / len(baseline_scores)
            c_avg = sum(candidate_scores) / len(candidate_scores)
            if b_avg - c_avg > REGRESSION_THRESHOLD:
                regressions.append({
                    "dimension": dim,
                    "baseline": b_avg,
                    "candidate": c_avg,
                })

    print_regression_alert(regressions)

    # Summary
    b_overall = sum(r.overall_score for r in results[baseline_model]) / len(results[baseline_model])
    c_overall = sum(r.overall_score for r in results[candidate_model]) / len(results[candidate_model])
    status = "regression" if regressions else "ok"

    return {
        "pipeline": cfg.pipeline_id,
        "baseline": {"model": baseline_model, "score": b_overall},
        "candidate": {"model": candidate_model, "score": c_overall},
        "regressions": regressions,
        "status": status,
        "score": c_overall,
    }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()
