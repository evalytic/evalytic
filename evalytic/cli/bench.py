"""``evalytic bench`` command -- one-command model benchmarking."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from ..bench.cost import estimate_run_cost, format_cost
from ..bench.registry import UnknownModelError, list_models as _list_models, resolve_model
from ..report.terminal import (
    confirm_proceed,
    console,
    create_generation_progress,
    create_scoring_progress,
    print_bench_header,
    print_cost_estimate,
    print_full_report,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_prompts(value: str) -> tuple[str, list[dict[str, Any]]]:
    """Parse ``--prompts`` value.

    Returns ``(pipeline, items)`` where *items* is a list of dicts.
    If *value* is a file path that exists, it is loaded as JSON.
    Otherwise it is treated as a single inline prompt string.
    """
    if os.path.isfile(value):
        with open(value) as f:
            data = json.load(f)
        # Plain list of prompts (e.g. ["prompt1", "prompt2"])
        if isinstance(data, list):
            # Normalize bare strings to dicts: "A cat" → {"prompt": "A cat"}
            items = [
                {"prompt": item} if isinstance(item, str) else item
                for item in data
            ]
            if any(isinstance(item, dict) and "image_url" in item for item in items):
                pipeline = "img2img"
            else:
                pipeline = "text2img"
            return pipeline, items
        pipeline = data.get("pipeline", "")
        if "prompts" in data:
            items = data["prompts"]
            pipeline = pipeline or "text2img"
        elif "inputs" in data:
            items = data["inputs"]
            pipeline = pipeline or "img2img"
        else:
            items = [data]
        # Auto-detect pipeline from items if not specified
        if not pipeline:
            if any("image_url" in item for item in items):
                pipeline = "img2img"
            else:
                pipeline = "text2img"
        return pipeline, items
    # Inline single prompt
    return "text2img", [{"prompt": value}]


def _print_model_list() -> None:
    """Print the full model registry as a table."""
    table = Table(title="Evalytic Model Registry", show_lines=True)
    table.add_column("Short Name", style="bold")
    table.add_column("fal.ai Endpoint")
    table.add_column("Pipeline")
    table.add_column("Cost/Image", justify="right")

    for entry in _list_models():
        table.add_row(
            entry.short_name,
            entry.endpoint,
            entry.pipeline,
            entry.display_cost,
        )

    console.print()
    console.print(table)
    console.print()


def _parse_images_file(path: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse ``--images`` JSON file.

    Accepts two formats:
    1. Object with optional top-level config + ``items`` list::

        {
            "judge": "openai/gpt-4o",
            "dimensions": ["visual_quality", "prompt_adherence"],
            "items": [
                {"prompt": "A cat", "images": {"model1": "url1", "model2": "url2"}}
            ]
        }

    2. Plain list of items::

        [{"prompt": "A cat", "images": {"model1": "url1"}}]

    Returns ``(items, config)`` where *config* holds top-level overrides
    (judge, dimensions, name, etc.) extracted from the file.
    """
    with open(path) as f:
        data = json.load(f)

    config: dict[str, Any] = {}
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("items", [])
        for key in ("judge", "dimensions", "name", "metrics"):
            if key in data:
                config[key] = data[key]
    else:
        console.print(f"[bold red]Error:[/bold red] --images file must be a JSON object or array.\n")
        sys.exit(2)

    if not items:
        console.print(f"[bold red]Error:[/bold red] --images file has no items.\n")
        sys.exit(2)

    return items, config


def _run_images_mode(
    *,
    ctx: Any,
    images_path: str,
    judge: str,
    judges_list: list[str] | None,
    judge_url: str | None,
    dimensions: tuple[str, ...],
    output: tuple[str, ...],
    output_dir: str | None,
    concurrency: int | None,
    name: str | None,
    yes: bool,
    quiet: bool,
    no_terminal: bool,
    metrics: tuple[str, ...],
    review: bool,
    review_port: int,
    clip_threshold: float | None,
    clip_weight: float | None,
    lpips_threshold: float | None,
    lpips_weight: float | None,
    face_threshold: float | None,
    face_weight: float | None,
    no_metric_scoring: bool,
    verbose: bool,
) -> None:
    """Handle --images mode: score pre-existing images (no generation)."""
    items, file_config = _parse_images_file(images_path)

    # File-level config overrides CLI defaults (CLI flags still win)
    if judge == "gemini-2.5-flash" and "judge" in file_config:
        judge = file_config["judge"]
    if not dimensions and "dimensions" in file_config:
        dimensions = tuple(file_config["dimensions"])
    if name is None and "name" in file_config:
        name = file_config["name"]

    dims = list(dimensions) if dimensions else None

    # Check judge API key (no FAL_KEY needed)
    from ..bench.judge import _parse_judge_string, _PROVIDER_DEFAULTS

    provider, _ = _parse_judge_string(judge)
    env_key = _PROVIDER_DEFAULTS.get(provider, {}).get("env_key")
    if env_key:
        help_urls = {
            "GEMINI_API_KEY": "https://aistudio.google.com/apikey",
            "OPENAI_API_KEY": "https://platform.openai.com/api-keys",
            "ANTHROPIC_API_KEY": "https://console.anthropic.com/settings/keys",
        }
        _check_api_key(
            env_key,
            help_urls.get(env_key, "your provider dashboard"),
            f"The {judge} judge requires an API key.",
        )

    # Extract model count from items for header
    all_models: dict[str, None] = {}
    for item in items:
        for model_name in item.get("images", {}):
            all_models[model_name] = None
    model_count = len(all_models)

    # Detect pipeline for header
    has_input = any(item.get("input_image") for item in items)
    pipeline = "img2img" if has_input else "text2img"

    if not no_terminal:
        print_bench_header(name or "bench", model_count, len(items), pipeline)
        console.print("  [dim]Mode: score-only (pre-existing images)[/dim]\n")

    # Check metrics availability
    metrics_list = list(metrics)
    if "metrics" in file_config and not metrics_list:
        metrics_list = file_config["metrics"]
    if metrics_list:
        from ..bench.metrics import METRICS_AVAILABLE

        if not METRICS_AVAILABLE:
            console.print(
                "\n  [yellow]Warning:[/yellow] evalytic[metrics] not installed. "
                "Metrics will be skipped.\n"
                "  Install with: pip install evalytic[metrics]\n"
            )
            metrics_list = []

    # Build scoring config
    scoring_config = None
    if metrics_list and not no_metric_scoring:
        from ..bench.types import MetricScoringConfig, ScoringConfig

        sc = ScoringConfig.default()
        if "clip_score" in sc.metric_configs or clip_threshold is not None or clip_weight is not None:
            base = sc.metric_configs.get("clip_score", MetricScoringConfig(0.18, 0.20, (0.18, 0.35)))
            sc.metric_configs["clip_score"] = MetricScoringConfig(
                flag_threshold=clip_threshold if clip_threshold is not None else base.flag_threshold,
                weight=clip_weight if clip_weight is not None else base.weight,
                normalize_range=base.normalize_range,
            )
        if "lpips" in sc.metric_configs or lpips_threshold is not None or lpips_weight is not None:
            base = sc.metric_configs.get("lpips", MetricScoringConfig(0.40, 0.20, (0.40, 0.95)))
            sc.metric_configs["lpips"] = MetricScoringConfig(
                flag_threshold=lpips_threshold if lpips_threshold is not None else base.flag_threshold,
                weight=lpips_weight if lpips_weight is not None else base.weight,
                normalize_range=base.normalize_range,
            )
        if "face_similarity" in sc.metric_configs or face_threshold is not None or face_weight is not None:
            base = sc.metric_configs.get("face_similarity", MetricScoringConfig(0.60, 0.20, (0.60, 0.95)))
            sc.metric_configs["face_similarity"] = MetricScoringConfig(
                flag_threshold=face_threshold if face_threshold is not None else base.flag_threshold,
                weight=face_weight if face_weight is not None else base.weight,
                normalize_range=base.normalize_range,
            )
        scoring_config = sc

    # Run bench with pre_images
    from ..bench.runner import run_bench
    from ..exceptions import EvalyticError

    try:
        report = run_bench(
            pre_images=items,
            judge=judge,
            judges=judges_list,
            judge_url=judge_url,
            dimensions=dims,
            metrics=metrics_list if metrics_list else None,
            concurrency=concurrency,
            name=name,
            scoring_config=scoring_config,
            verbose=verbose,
        )
    except EvalyticError as exc:
        console.print(f"\n[bold red]Error:[/bold red] {exc}\n")
        sys.exit(1)

    # Terminal output
    if not no_terminal:
        print_full_report(report)

    # File outputs
    for out_path in output:
        if out_path.endswith(".json"):
            report.to_json(out_path)
            if not quiet:
                console.print(f"  JSON report written to: {out_path}")
        elif out_path.endswith(".html"):
            report.to_html(out_path)
            if not quiet:
                console.print(f"  HTML report written to: {out_path}")
        else:
            console.print(f"  [yellow]Warning:[/yellow] Unknown output format: {out_path}")

    if not quiet and output:
        console.print()

    # Output directory (timestamped subfolder)
    if output_dir:
        _write_output_dir(report, output_dir, quiet)

    # Review mode
    if review:
        from ..report.review_server import ReviewServer

        if not quiet:
            console.print(f"\n  Opening browser review on port {review_port}...")
        server = ReviewServer(report, port=review_port)
        server.start()
        server.wait()
        report = server.get_report()
        if not quiet:
            console.print("  Review complete. Human scores merged.")


def _write_output_dir(report: Any, output_dir: str, quiet: bool) -> None:
    """Write report files to a timestamped subfolder inside output_dir."""
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_dir) / f"{report.name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    report.to_html(str(run_dir / "report.html"))
    report.to_json(str(run_dir / "report.json"))

    if report.errors:
        from ..report.error_log import write_error_log

        write_error_log(report, str(run_dir / "errors.log"))

    if not quiet:
        console.print(f"  Report saved to: {run_dir}/")
        if report.errors:
            console.print(
                f"  [yellow]{len(report.errors)} errors logged to: {run_dir}/errors.log[/yellow]"
            )
        console.print()


def _check_api_key(name: str, help_url: str, help_text: str) -> None:
    """Exit with an actionable error if an API key is missing."""
    if not os.environ.get(name):
        console.print(f"\n[bold red]Error:[/bold red] {name} environment variable not set.\n")
        console.print(f"  {help_text}\n")
        console.print(f"  Get a key at: {help_url}")
        console.print(f"  Then either:")
        console.print(f"    1. Set environment variable: export {name}=your-key-here")
        console.print(f"    2. Add to .env file: {name}=your-key-here")
        console.print(f"    3. Run evalytic init to set up interactively\n")
        sys.exit(2)


def _check_expected_scores(
    report: Any,
    expected_map: dict[int, dict[str, float]],
) -> None:
    """Compare bench results against expected scores and print violations.

    *expected_map* maps item index → {dimension: expected_score}.
    """
    violations: list[tuple[str, str, str, float, float, float]] = []  # (item, model, dim, actual, expected, delta)

    for idx, item in enumerate(report.items):
        expected = expected_map.get(idx, {})
        if not expected:
            continue
        for model_name, result in item.results.items():
            for dim_result in result.scores:
                dim = dim_result.dimension
                if dim in expected:
                    delta = dim_result.score - expected[dim]
                    if delta < 0:
                        label = item.prompt or item.instruction or f"item-{idx}"
                        if len(label) > 30:
                            label = label[:27] + "..."
                        violations.append((label, model_name, dim, dim_result.score, expected[dim], delta))

            # Check overall expected
            if "overall" in expected:
                actual_overall = result.overall_score
                delta = actual_overall - expected["overall"]
                if delta < 0:
                    label = item.prompt or item.instruction or f"item-{idx}"
                    if len(label) > 30:
                        label = label[:27] + "..."
                    violations.append((label, model_name, "overall", actual_overall, expected["overall"], delta))

    if violations:
        console.print("\n  [bold yellow]Expected Score Violations:[/bold yellow]\n")
        table = Table(show_lines=False, pad_edge=True)
        table.add_column("Item", max_width=30)
        table.add_column("Model")
        table.add_column("Dimension")
        table.add_column("Actual", justify="right")
        table.add_column("Expected", justify="right")
        table.add_column("Delta", justify="right")

        for label, model_name, dim, actual, exp, delta in violations:
            delta_style = "bold red" if delta < -0.5 else "yellow"
            table.add_row(
                label,
                model_name,
                dim,
                f"{actual:.2f}",
                f"{exp:.2f}",
                f"[{delta_style}]{delta:+.2f}[/{delta_style}]",
            )
        console.print(table)
        console.print()
    else:
        console.print("\n  [green]All results meet expected scores.[/green]\n")


# ---------------------------------------------------------------------------
# Grouped help: Essential vs Advanced options
# ---------------------------------------------------------------------------

_ESSENTIAL_BENCH_PARAMS = {
    "models", "prompts", "inputs", "images", "dataset", "output",
    "output_dir", "yes", "review", "judge", "judges", "list_models",
}


class GroupedBenchCommand(click.Command):
    """Custom Click command that groups --help into Essential/Advanced."""

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        self.format_usage(ctx, formatter)
        self.format_help_text(ctx, formatter)

        essential: list[tuple[str, str]] = []
        advanced: list[tuple[str, str]] = []

        for param in self.get_params(ctx):
            record = param.get_help_record(ctx)
            if record is None:
                continue
            if param.name in _ESSENTIAL_BENCH_PARAMS:
                essential.append(record)
            else:
                advanced.append(record)

        if essential:
            with formatter.section("Essential Options"):
                formatter.write_dl(essential)
        if advanced:
            with formatter.section("Advanced Options"):
                formatter.write_dl(advanced)

        self.format_epilog(ctx, formatter)


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


@click.command(cls=GroupedBenchCommand)
@click.option("--models", "-m", multiple=True, required=False, help="Model short names or fal.ai endpoints.")
@click.option("--prompts", "-p", default=None, help="Path to JSON file or inline prompt string.")
@click.option("--inputs", "-i", default=None, help="Path to JSON file with img2img inputs.")
@click.option("--images", default=None, help="Path to JSON file with pre-existing images (skip generation).")
@click.option("--dataset", "dataset_path", default=None, help="Path to dataset file (enriched prompts with metadata/expected).")
@click.option("--check-expected", is_flag=True, help="Compare results against expected scores in dataset.")
@click.option("--judge", "-j", default=None, help="VLM judge (default: gemini-2.5-flash). E.g. gemini-3-flash, gpt-5.2, claude-sonnet-4-6, ollama/qwen3-vl.")
@click.option("--judges", default=None, help="Comma-separated judges for consensus (e.g. 'gemini-3-flash,gpt-4o-mini'). Min 2, max 3.")
@click.option("--judge-url", default=None, help="Custom judge API base URL.")
@click.option("--dimensions", "-d", multiple=True, help="Quality dimensions to score.")
@click.option("--output", "-o", multiple=True, help="Output file paths (.json, .html).")
@click.option("--output-dir", default=None, help="Output directory. Creates timestamped subfolder with report.html, report.json, and errors.log.")
@click.option("--concurrency", default=None, type=int, help="Max parallel generation requests (default: 4).")
@click.option("--image-size", default=None, help="Image size (e.g., landscape_16_9).")
@click.option("--seed", default=None, type=int, help="Fixed seed for reproducible generation.")
@click.option("--fal-params", default=None, help="Path to JSON file with additional fal.ai params.")
@click.option("--cache-dir", default=None, help="Local image cache directory.")
@click.option("--timeout", default=300, show_default=True, help="Max seconds per generation request.")
@click.option("--name", default=None, help="Human-readable name for this bench run.")
@click.option("--yes", "-y", is_flag=True, help="Skip cost confirmation.")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress bars.")
@click.option("--no-terminal", is_flag=True, help="Suppress Rich terminal output.")
@click.option("--metrics", multiple=True, help="Local metrics to compute: clip, lpips.")
@click.option("--review", is_flag=True, help="Open browser review after scoring.")
@click.option("--review-port", default=3847, show_default=True, help="Port for review server.")
@click.option("--clip-threshold", default=None, type=float, help="CLIP flag threshold (default: 0.18).")
@click.option("--clip-weight", default=None, type=float, help="CLIP weight in overall score (default: 0.20).")
@click.option("--lpips-threshold", default=None, type=float, help="LPIPS flag threshold (default: 0.40).")
@click.option("--lpips-weight", default=None, type=float, help="LPIPS weight in overall score (default: 0.20).")
@click.option("--face-threshold", default=None, type=float, help="Face similarity flag threshold (default: 0.60).")
@click.option("--face-weight", default=None, type=float, help="Face similarity weight in overall score (default: 0.20).")
@click.option("--no-metric-scoring", is_flag=True, help="Show metrics but exclude from overall score.")
@click.option("--list-models", is_flag=True, help="Print model registry and exit.")
@click.pass_context
def bench(
    ctx: click.Context,
    models: tuple[str, ...],
    prompts: str | None,
    inputs: str | None,
    images: str | None,
    dataset_path: str | None,
    check_expected: bool,
    judge: str | None,
    judges: str | None,
    judge_url: str | None,
    dimensions: tuple[str, ...],
    output: tuple[str, ...],
    output_dir: str | None,
    concurrency: int | None,
    image_size: str | None,
    seed: int | None,
    fal_params: str | None,
    cache_dir: str | None,
    timeout: int,
    name: str | None,
    yes: bool,
    quiet: bool,
    no_terminal: bool,
    metrics: tuple[str, ...],
    review: bool,
    review_port: int,
    clip_threshold: float | None,
    clip_weight: float | None,
    lpips_threshold: float | None,
    lpips_weight: float | None,
    face_threshold: float | None,
    face_weight: float | None,
    no_metric_scoring: bool,
    list_models: bool,
    **_: Any,
) -> None:
    """Benchmark image generation models: generate, score, report."""
    # Merge config: CLI flags override evalytic.toml [bench] values
    cfg = (ctx.obj or {}).get("config", {}).get("bench", {})
    verbose = (ctx.obj or {}).get("verbose", False)

    if judge is None:
        judge = cfg.get("judge", "gemini-2.5-flash")
    # Parse --judges (comma-separated) or from config
    judges_list: list[str] | None = None
    if judges:
        judges_list = [j.strip() for j in judges.split(",") if j.strip()]
    elif cfg.get("judges"):
        judges_list = cfg["judges"]
    if concurrency is None:
        concurrency = cfg.get("concurrency", 4)
    if not dimensions and "dimensions" in cfg:
        dimensions = tuple(cfg["dimensions"])
    if image_size is None:
        image_size = cfg.get("image_size")
    if seed is None:
        seed = cfg.get("seed")
    if output_dir is None:
        output_dir = cfg.get("output_dir")

    # Metric config from toml
    cfg_metrics = cfg.get("metrics", {})
    if clip_threshold is None:
        clip_threshold = cfg_metrics.get("clip_threshold")
    if clip_weight is None:
        clip_weight = cfg_metrics.get("clip_weight")
    if lpips_threshold is None:
        lpips_threshold = cfg_metrics.get("lpips_threshold")
    if lpips_weight is None:
        lpips_weight = cfg_metrics.get("lpips_weight")
    if face_threshold is None:
        face_threshold = cfg_metrics.get("face_threshold")
    if face_weight is None:
        face_weight = cfg_metrics.get("face_weight")

    # Handle --list-models
    if list_models:
        _print_model_list()
        return

    # Check API keys for consensus judges
    if judges_list:
        from ..bench.judge import _parse_judge_string, _PROVIDER_DEFAULTS

        help_urls = {
            "GEMINI_API_KEY": "https://aistudio.google.com/apikey",
            "OPENAI_API_KEY": "https://platform.openai.com/api-keys",
            "ANTHROPIC_API_KEY": "https://console.anthropic.com/settings/keys",
        }
        for j in judges_list:
            prov, _ = _parse_judge_string(j)
            ek = _PROVIDER_DEFAULTS.get(prov, {}).get("env_key")
            if ek:
                _check_api_key(
                    ek,
                    help_urls.get(ek, "your provider dashboard"),
                    f"The {j} judge requires an API key.",
                )

    # ---- PRE-IMAGES MODE (--images): score-only, no generation ----
    if images is not None:
        _run_images_mode(
            ctx=ctx,
            images_path=images,
            judge=judge,
            judges_list=judges_list,
            judge_url=judge_url,
            dimensions=dimensions,
            output=output,
            output_dir=output_dir,
            concurrency=concurrency,
            name=name,
            yes=yes,
            quiet=quiet,
            no_terminal=no_terminal,
            metrics=metrics,
            review=review,
            review_port=review_port,
            clip_threshold=clip_threshold,
            clip_weight=clip_weight,
            lpips_threshold=lpips_threshold,
            lpips_weight=lpips_weight,
            face_threshold=face_threshold,
            face_weight=face_weight,
            no_metric_scoring=no_metric_scoring,
            verbose=verbose,
        )
        return

    # ---- DATASET MODE ----
    expected_map: dict[int, dict[str, float]] = {}

    if dataset_path is not None:
        if prompts is not None or inputs is not None:
            console.print("[bold red]Error:[/bold red] --dataset cannot be used with --prompts or --inputs.\n")
            sys.exit(2)

        from .dataset_cmd import _load_dataset

        ds_data = _load_dataset(dataset_path)
        ds_items = ds_data.get("items", [])
        ds_pipeline = ds_data.get("pipeline", "")

        if not ds_items:
            console.print("[bold red]Error:[/bold red] Dataset has no items.\n")
            sys.exit(2)

        # Auto-detect pipeline
        if not ds_pipeline:
            if any("image_url" in item for item in ds_items):
                ds_pipeline = "img2img"
            else:
                ds_pipeline = "text2img"

        # Collect expected scores for --check-expected
        if check_expected:
            for idx, item in enumerate(ds_items):
                exp = item.get("expected", {})
                if exp:
                    expected_map[idx] = exp

        # Feed items to bench
        if ds_pipeline == "img2img":
            inputs = dataset_path
        else:
            prompts = dataset_path

    # ---- GENERATION MODE (default) ----

    # Smart default: models from config or interactive demo
    if not models:
        cfg_models = cfg.get("models", [])
        if cfg_models:
            models = tuple(cfg_models) if isinstance(cfg_models, list) else (cfg_models,)
        elif sys.stdin.isatty():
            console.print("  No models specified. Run demo with [bold]flux-schnell[/bold]? [Y/n] ", end="")
            answer = input().strip().lower()
            if answer in ("", "y", "yes"):
                models = ("flux-schnell",)
            else:
                console.print("\n  Example: evalytic bench -m flux-schnell -p \"A cat\" -y")
                console.print("  List all: evalytic bench --list-models\n")
                sys.exit(0)
        else:
            console.print("[bold red]Error:[/bold red] --models is required. Specify one or more models.\n")
            console.print("  Example: evalytic bench --models flux-schnell flux-pro --prompts \"A cat\"")
            console.print("  List all: evalytic bench --list-models")
            sys.exit(2)

    # Resolve and validate model names
    for m in models:
        try:
            resolve_model(m)
        except UnknownModelError as exc:
            console.print(f"\n[bold red]Error:[/bold red] {exc}\n")
            console.print("  List all models: evalytic bench --list-models")
            console.print("  Use custom endpoint: evalytic bench --models fal-ai/my-model/v1 ...\n")
            sys.exit(2)

    # Smart default: prompts from config or demo prompt
    if prompts is None and inputs is None:
        cfg_prompts = cfg.get("prompts")
        if cfg_prompts:
            prompts = cfg_prompts if isinstance(cfg_prompts, str) else json.dumps(cfg_prompts)
        else:
            prompts = "A cat sitting on a windowsill at sunset"
            if not no_terminal:
                console.print(f"  [dim]Using demo prompt: \"{prompts}\"[/dim]\n")

    if inputs:
        pipeline, items = _parse_prompts(inputs)
        pipeline = "img2img"
    else:
        pipeline, items = _parse_prompts(prompts)  # type: ignore[arg-type]

    # Check API keys
    _check_api_key(
        "FAL_KEY",
        "https://fal.ai/dashboard/keys",
        "The bench command requires a fal.ai API key for image generation.",
    )
    from ..bench.judge import _parse_judge_string, _PROVIDER_DEFAULTS

    provider, _ = _parse_judge_string(judge)
    env_key = _PROVIDER_DEFAULTS.get(provider, {}).get("env_key")
    if env_key:
        help_urls = {
            "GEMINI_API_KEY": "https://aistudio.google.com/apikey",
            "OPENAI_API_KEY": "https://platform.openai.com/api-keys",
            "ANTHROPIC_API_KEY": "https://console.anthropic.com/settings/keys",
        }
        _check_api_key(
            env_key,
            help_urls.get(env_key, "your provider dashboard"),
            f"The {judge} judge requires an API key.",
        )

    # Parse fal params
    extra_fal_params: dict[str, Any] | None = None
    if fal_params:
        with open(fal_params) as f:
            extra_fal_params = json.load(f)

    dims = list(dimensions) if dimensions else None

    # Check metrics availability
    metrics_list = list(metrics)
    if metrics_list:
        from ..bench.metrics import METRICS_AVAILABLE

        if not METRICS_AVAILABLE:
            console.print(
                "\n  [yellow]Warning:[/yellow] evalytic[metrics] not installed. "
                "Metrics will be skipped.\n"
                "  Install with: pip install evalytic[metrics]\n"
            )
            metrics_list = []

    # Cost estimate
    cost_est = estimate_run_cost(
        list(models),
        len(items),
        judge,
        dimensions_per_item=len(dims) if dims else 2,
        judges=judges_list,
    )

    if not no_terminal:
        print_bench_header(name or "bench", len(models), len(items), pipeline)
        print_cost_estimate(cost_est)

    if not yes:
        if not confirm_proceed():
            console.print("  Aborted.")
            sys.exit(0)

    # Build scoring config
    scoring_config = None
    if metrics_list and not no_metric_scoring:
        from ..bench.types import MetricScoringConfig, ScoringConfig

        sc = ScoringConfig.default()
        # Apply CLI overrides
        if "clip_score" in sc.metric_configs or clip_threshold is not None or clip_weight is not None:
            base = sc.metric_configs.get("clip_score", MetricScoringConfig(0.18, 0.20, (0.18, 0.35)))
            sc.metric_configs["clip_score"] = MetricScoringConfig(
                flag_threshold=clip_threshold if clip_threshold is not None else base.flag_threshold,
                weight=clip_weight if clip_weight is not None else base.weight,
                normalize_range=base.normalize_range,
            )
        if "lpips" in sc.metric_configs or lpips_threshold is not None or lpips_weight is not None:
            base = sc.metric_configs.get("lpips", MetricScoringConfig(0.40, 0.20, (0.40, 0.95)))
            sc.metric_configs["lpips"] = MetricScoringConfig(
                flag_threshold=lpips_threshold if lpips_threshold is not None else base.flag_threshold,
                weight=lpips_weight if lpips_weight is not None else base.weight,
                normalize_range=base.normalize_range,
            )
        if "face_similarity" in sc.metric_configs or face_threshold is not None or face_weight is not None:
            base = sc.metric_configs.get("face_similarity", MetricScoringConfig(0.60, 0.20, (0.60, 0.95)))
            sc.metric_configs["face_similarity"] = MetricScoringConfig(
                flag_threshold=face_threshold if face_threshold is not None else base.flag_threshold,
                weight=face_weight if face_weight is not None else base.weight,
                normalize_range=base.normalize_range,
            )
        scoring_config = sc

    # Run bench
    from ..bench.runner import run_bench
    from ..exceptions import EvalyticError

    try:
        report = run_bench(
            models=list(models),
            prompts=items if pipeline == "text2img" else None,
            inputs=items if pipeline == "img2img" else None,
            judge=judge,
            judges=judges_list,
            judge_url=judge_url,
            dimensions=dims,
            metrics=metrics_list if metrics_list else None,
            concurrency=concurrency,
            seed=seed,
            image_size=image_size,
            fal_params=extra_fal_params,
            cache_dir=cache_dir,
            timeout=timeout,
            name=name,
            scoring_config=scoring_config,
            verbose=verbose,
        )
    except EvalyticError as exc:
        console.print(f"\n[bold red]Error:[/bold red] {exc}\n")
        sys.exit(1)

    # Terminal output
    if not no_terminal:
        print_full_report(report)

    # Check expected scores from dataset
    if check_expected and expected_map:
        _check_expected_scores(report, expected_map)

    # File outputs
    for out_path in output:
        if out_path.endswith(".json"):
            report.to_json(out_path)
            if not quiet:
                console.print(f"  JSON report written to: {out_path}")
        elif out_path.endswith(".html"):
            report.to_html(out_path)
            if not quiet:
                console.print(f"  HTML report written to: {out_path}")
        else:
            console.print(f"  [yellow]Warning:[/yellow] Unknown output format: {out_path}")

    if not quiet and output:
        console.print()

    # Output directory (timestamped subfolder)
    if output_dir:
        _write_output_dir(report, output_dir, quiet)

    # Review mode
    if review:
        from ..report.review_server import ReviewServer

        if not quiet:
            console.print(f"\n  Opening browser review on port {review_port}...")
        server = ReviewServer(report, port=review_port)
        server.start()
        server.wait()
        report = server.get_report()
        if not quiet:
            console.print("  Review complete. Human scores merged.")
