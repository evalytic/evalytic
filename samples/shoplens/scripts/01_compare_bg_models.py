#!/usr/bin/env python3
"""Pipeline 1: Compare 3 Flux models for product background generation.

Generates 5 backgrounds with each model, scores them on visual_quality
and prompt_adherence, and announces the winner.
"""

from __future__ import annotations

import os
import sys

# Allow running from samples/shoplens/scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shoplens.config import PIPELINES
from shoplens.generate import generate_text2img
from shoplens.judge import LocalJudge, LocalEvalResult
from shoplens.report import console, print_header, print_model_comparison

from golden_sets.backgrounds import BACKGROUNDS


def main() -> dict:
    cfg = PIPELINES["bg-generator"]
    models = [cfg.default_model] + cfg.challenger_models

    print_header(cfg.name, "Comparing Flux models on 5 product background prompts")

    judge = LocalJudge()
    results_by_model: dict[str, list[LocalEvalResult]] = {}

    for model in models:
        console.print(f"\n[bold cyan]{model}[/bold cyan]")
        results: list[LocalEvalResult] = []

        for i, bg in enumerate(BACKGROUNDS, 1):
            console.print(f"  [{i}/{len(BACKGROUNDS)}] Generating: {bg['expected']}...", end=" ")
            image_url = generate_text2img(bg["prompt"], model, **cfg.fal_params)
            console.print(f"scoring...", end=" ")
            result = judge.score(
                image_url=image_url,
                dimensions=cfg.dimensions,
                prompt=bg["prompt"],
            )
            console.print(f"[green]{result.display_score}[/green]")
            results.append(result)

        results_by_model[model] = results

    print_model_comparison(results_by_model)

    # Build summary for run_all
    summary: dict[str, dict] = {}
    for model, results in results_by_model.items():
        avg = sum(r.overall_score for r in results) / len(results)
        summary[model] = {"score": avg, "count": len(results)}

    best_model = max(summary, key=lambda m: summary[m]["score"])
    return {
        "pipeline": cfg.pipeline_id,
        "winner": best_model,
        "models": summary,
        "score": summary[best_model]["score"],
    }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()
