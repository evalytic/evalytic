"""``evalytic eval`` command -- score an existing image without generation."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import click
from rich.table import Table

from ..bench.judge import DIMENSION_CONFIG, Judge, _parse_judge_string, _PROVIDER_DEFAULTS
from ..exceptions import EvalyticError, ValidationError
from ..report.terminal import console, score_color


def _auto_detect_eval_dimensions(
    prompt: str | None,
    input_image: str | None,
) -> list[str]:
    """Pick default dimensions based on what the user provided."""
    dims: list[str] = ["visual_quality"]
    if input_image:
        # img2img pipeline
        dims.extend(["input_fidelity", "transformation_quality", "artifact_detection"])
    if prompt:
        dims.append("prompt_adherence")
        # Check for text-related keywords
        _text_kw = {"text", "word", "letter", "write", "say", "font", "type", "sign"}
        if any(kw in prompt.lower() for kw in _text_kw):
            dims.append("text_rendering")
    return dims


def _resolve_default_judge(config: dict[str, Any] | None = None) -> str:
    """Pick the best default judge based on available keys and config."""
    cfg = (config or {}).get("bench", {})
    if "judge" in cfg:
        return cfg["judge"]
    if os.environ.get("FAL_KEY") and not os.environ.get("GEMINI_API_KEY"):
        return "fal/gemini-2.5-flash"
    return "gemini-2.5-flash"


@click.command("eval")
@click.option("--image", required=True, help="Path or URL to the image to evaluate.")
@click.option(
    "--dimensions", "-d", multiple=True,
    help="Dimensions to score. Default: auto-detect from context.",
)
@click.option("--prompt", default=None, help="Generation prompt (enables prompt_adherence scoring).")
@click.option("--input-image", default=None, help="Input image path/URL (enables img2img dimensions).")
@click.option("--judge", "-j", default=None, help="VLM judge (e.g. fal/gemini-2.5-flash, openai/gpt-5.2). Default: auto-detect from config/env.")
@click.option("--judge-url", default=None, help="Custom judge API base URL.")
@click.option("--no-judge", is_flag=True, help="Skip VLM judge, use local metrics only (free, no API key needed).")
@click.option("--output", "-o", default=None, help="Write results to JSON file.")
@click.pass_context
def eval_cmd(
    ctx: click.Context,
    image: str,
    dimensions: tuple[str, ...],
    prompt: str | None,
    input_image: str | None,
    judge: str | None,
    judge_url: str | None,
    no_judge: bool,
    output: str | None,
) -> None:
    """Score an existing image on quality dimensions (no generation)."""
    if no_judge:
        # Metrics-only mode
        from ..bench.metrics import SharpnessScorer, _resolve_image_path

        # Resolve image path (download URL if needed)
        local_path = _resolve_image_path(image)

        console.print()
        console.print("[bold]Eval Results (metrics only)[/bold]")

        table = Table(show_lines=True)
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="center")
        table.add_column("Description")

        metrics_data: dict[str, float] = {}

        # Sharpness (always available)
        sharpness_scorer = SharpnessScorer()
        sharpness = sharpness_scorer.score(local_path)
        metrics_data["sharpness"] = sharpness
        table.add_row("Sharpness", f"{sharpness:.4f}", "Variance of Laplacian (0-1)")

        # CLIP score (if prompt given and evalytic[metrics] installed)
        if prompt:
            try:
                from ..bench.metrics import METRICS_AVAILABLE
                if METRICS_AVAILABLE:
                    from ..bench.metrics import CLIPScorer
                    scorer = CLIPScorer()
                    clip_val = scorer.score(local_path, prompt)
                    metrics_data["clip_score"] = clip_val
                    table.add_row("CLIP Score", f"{clip_val:.4f}", "Text-image similarity")
            except Exception:
                pass

        console.print(table)
        console.print()

        # JSON output
        if output:
            data: dict[str, Any] = {
                "image": image,
                "metrics": metrics_data,
                "judge": None,
            }
            if prompt:
                data["prompt"] = prompt
            with open(output, "w") as f:
                json.dump(data, f, indent=2)
            console.print(f"  Results written to: {output}")
        return

    # Resolve judge: CLI flag > config > smart default
    if judge is None:
        config = (ctx.obj or {}).get("config")
        judge = _resolve_default_judge(config)

    # Auto-detect dimensions when not specified
    if dimensions:
        dims = list(dimensions)
    else:
        dims = _auto_detect_eval_dimensions(prompt, input_image)

    # Validate dimensions
    unknown = set(dims) - set(DIMENSION_CONFIG)
    if unknown:
        console.print(
            f"[bold red]Error:[/bold red] Unknown dimensions: {sorted(unknown)}\n"
            f"  Valid: {', '.join(DIMENSION_CONFIG)}"
        )
        sys.exit(2)

    # Check requirements per dimension
    for dim in dims:
        cfg = DIMENSION_CONFIG[dim]
        if cfg.needs_prompt and not prompt:
            console.print(
                f"[bold red]Error:[/bold red] Dimension {dim!r} requires --prompt."
            )
            sys.exit(2)
        if cfg.needs_input and not input_image:
            console.print(
                f"[bold red]Error:[/bold red] Dimension {dim!r} requires --input-image."
            )
            sys.exit(2)

    # Check API key (only for cloud providers)
    provider, _ = _parse_judge_string(judge)
    env_key = _PROVIDER_DEFAULTS.get(provider, {}).get("env_key")
    if env_key and not os.environ.get(env_key):
        help_urls = {
            "GEMINI_API_KEY": "https://aistudio.google.com/apikey",
            "OPENAI_API_KEY": "https://platform.openai.com/api-keys",
            "ANTHROPIC_API_KEY": "https://console.anthropic.com/settings/keys",
        }
        console.print(
            f"\n[bold red]Error:[/bold red] {env_key} environment variable not set.\n"
            f"  Get a key at: {help_urls.get(env_key, 'your provider dashboard')}\n"
            f"  Then: export {env_key}=your-key-here"
        )
        sys.exit(2)

    # Score
    try:
        with Judge(judge=judge, base_url=judge_url) as j:
            results = j.score(
                image_url=image,
                dimensions=dims,
                prompt=prompt,
                input_image_url=input_image,
            )
    except EvalyticError as exc:
        console.print(f"\n[bold red]Error:[/bold red] {exc}\n")
        sys.exit(1)

    # Display results
    table = Table(title="Eval Results", show_lines=True)
    table.add_column("Dimension", style="bold")
    table.add_column("Score", justify="center")
    table.add_column("Conf", justify="center")
    table.add_column("Explanation")

    for r in results:
        color = score_color(r.score)
        conf = getattr(r, "confidence", 1.0)
        conf_color = "green" if conf >= 0.8 else ("yellow" if conf >= 0.5 else "red")
        table.add_row(
            r.dimension,
            f"[{color}]{r.score:.1f}[/{color}]",
            f"[{conf_color}]{conf:.0%}[/{conf_color}]",
            r.explanation,
        )

    console.print()
    console.print(table)
    console.print()

    # JSON output
    if output:
        data = {
            "image": image,
            "dimensions": [
                {
                    "dimension": r.dimension,
                    "score": r.score,
                    "confidence": getattr(r, "confidence", 1.0),
                    "explanation": r.explanation,
                    "evidence": r.evidence,
                }
                for r in results
            ],
            "judge": judge,
        }
        if prompt:
            data["prompt"] = prompt
        if input_image:
            data["input_image"] = input_image

        with open(output, "w") as f:
            json.dump(data, f, indent=2)
        console.print(f"  Results written to: {output}")
