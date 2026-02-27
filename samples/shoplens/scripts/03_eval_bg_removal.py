#!/usr/bin/env python3
"""Pipeline 3: Single-image quality eval for BiRefNet background removal.

Runs BiRefNet on 3 challenging product photos and flags any
artifact_detection score below 3.5.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shoplens.config import PIPELINES
from shoplens.generate import remove_background
from shoplens.judge import LocalJudge, LocalEvalResult
from shoplens.report import console, print_header, print_dimension_breakdown

from golden_sets.bg_removal import CUTOUTS

ARTIFACT_THRESHOLD = 3.5


def main() -> dict:
    cfg = PIPELINES["clean-cut"]

    print_header(cfg.name, "Quality gate: BiRefNet background removal on 3 edge cases")

    judge = LocalJudge()
    results: list[LocalEvalResult] = []
    flagged: list[str] = []

    for i, item in enumerate(CUTOUTS, 1):
        console.print(f"  [{i}/{len(CUTOUTS)}] {item['product_type']} ({item['challenge']})...", end=" ")
        output_url = remove_background(item["image_url"])
        console.print("scoring...", end=" ")
        result = judge.score(
            image_url=output_url,
            dimensions=cfg.dimensions,
            input_image_url=item["image_url"],
        )
        console.print(f"[green]{result.display_score}[/green]")
        results.append(result)

        artifact = result.dimension("artifact_detection")
        if artifact and artifact.score < ARTIFACT_THRESHOLD:
            flagged.append(item["product_type"])

    print_dimension_breakdown(results, label="BiRefNet Quality Report")

    if flagged:
        console.print(f"[bold red]Quality gate FAILED for: {', '.join(flagged)}[/bold red]\n")
        status = "alert"
    else:
        console.print("[bold green]All images passed quality gate.[/bold green]\n")
        status = "ok"

    avg_score = sum(r.overall_score for r in results) / len(results)
    return {
        "pipeline": cfg.pipeline_id,
        "score": avg_score,
        "flagged": flagged,
        "status": status,
    }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()
