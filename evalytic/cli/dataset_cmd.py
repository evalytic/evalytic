"""``evalytic dataset`` command group -- manage evaluation datasets."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_dataset(data: Any) -> dict[str, Any]:
    """Normalize all supported formats into ``{"items": [...]}``.

    Accepted inputs:
    - Plain list of strings: ``["prompt1", "prompt2"]``
    - Plain list of dicts: ``[{"prompt": "..."}, ...]``
    - Object with ``"items"`` key
    - Object with ``"prompts"`` key (text2img legacy)
    - Object with ``"inputs"`` key (img2img legacy)
    """
    if isinstance(data, list):
        items = [
            {"prompt": item} if isinstance(item, str) else item
            for item in data
        ]
        return {"items": items}

    if not isinstance(data, dict):
        return {"items": []}

    # Already canonical
    if "items" in data:
        return data

    result = {k: v for k, v in data.items() if k not in ("prompts", "inputs")}

    if "prompts" in data:
        items = data["prompts"]
        result["items"] = [
            {"prompt": item} if isinstance(item, str) else item
            for item in items
        ]
        result.setdefault("pipeline", "text2img")
    elif "inputs" in data:
        result["items"] = data["inputs"]
        result.setdefault("pipeline", "img2img")
    else:
        result["items"] = []

    return result


def _load_dataset(path: str) -> dict[str, Any]:
    """Load a dataset JSON file and normalize it."""
    p = Path(path)
    if not p.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {path}\n")
        sys.exit(2)
    try:
        with open(p) as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        console.print(f"[bold red]Error:[/bold red] Invalid JSON in {path}: {exc}\n")
        sys.exit(2)
    return _normalize_dataset(data)


def _validate_dataset(data: dict[str, Any]) -> list[str]:
    """Validate a normalized dataset, returning a list of warnings."""
    warnings: list[str] = []
    items = data.get("items", [])
    pipeline = data.get("pipeline", "")

    if not items:
        warnings.append("Dataset has no items.")

    for i, item in enumerate(items):
        # Check pipeline consistency
        has_image = "image_url" in item
        has_prompt = "prompt" in item

        if pipeline == "text2img" and has_image and not has_prompt:
            warnings.append(f"Item {i}: has image_url but pipeline is text2img.")
        if pipeline == "img2img" and not has_image:
            warnings.append(f"Item {i}: missing image_url for img2img pipeline.")

        # Validate expected scores
        expected = item.get("expected", {})
        for dim, score in expected.items():
            if not isinstance(score, (int, float)):
                warnings.append(f"Item {i}: expected[{dim}] is not a number.")
            elif score < 0 or score > 5:
                warnings.append(f"Item {i}: expected[{dim}]={score} is outside 0-5 range.")

    return warnings


def _write_dataset(data: dict[str, Any], path: str) -> None:
    """Write dataset dict to JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _detect_raw_format(path: str) -> str | None:
    """Detect the original key format of a JSON file.

    Returns ``"prompts"``, ``"inputs"``, ``"items"``, ``"list"``, or *None*.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    if isinstance(data, list):
        return "list"
    if isinstance(data, dict):
        if "items" in data:
            return "items"
        if "prompts" in data:
            return "prompts"
        if "inputs" in data:
            return "inputs"
    return None


# ---------------------------------------------------------------------------
# Command group
# ---------------------------------------------------------------------------


@click.group("dataset")
def dataset_group() -> None:
    """Manage evaluation datasets."""


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


@dataset_group.command("create")
@click.option("--name", "-n", required=True, help="Dataset name.")
@click.option(
    "--pipeline",
    type=click.Choice(["text2img", "img2img"]),
    default="text2img",
    show_default=True,
    help="Pipeline type.",
)
@click.option("--description", "-d", default="", help="Dataset description.")
@click.option("--output", "-o", default=None, help="Output file path (default: <name>.json).")
def dataset_create(name: str, pipeline: str, description: str, output: str | None) -> None:
    """Create a new empty dataset file."""
    out_path = output or f"{name}.json"
    if Path(out_path).exists():
        console.print(f"[bold red]Error:[/bold red] File already exists: {out_path}\n")
        sys.exit(2)

    data: dict[str, Any] = {
        "name": name,
        "description": description,
        "pipeline": pipeline,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "items": [],
    }
    _write_dataset(data, out_path)
    console.print(f"  Created dataset: [bold]{out_path}[/bold]\n")


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


@dataset_group.command("show")
@click.argument("path")
def dataset_show(path: str) -> None:
    """Show dataset contents as a table."""
    data = _load_dataset(path)
    items = data.get("items", [])
    pipeline = data.get("pipeline", "")
    ds_name = data.get("name", Path(path).stem)

    # Auto-detect pipeline
    if not pipeline:
        if any("image_url" in item for item in items):
            pipeline = "img2img"
        else:
            pipeline = "text2img"

    console.print(f"\n  [bold]{ds_name}[/bold]  |  {pipeline}  |  {len(items)} items\n")

    if not items:
        console.print("  (empty dataset)\n")
        return

    table = Table(show_lines=False, pad_edge=True)
    table.add_column("#", style="dim", justify="right")

    if pipeline == "img2img":
        table.add_column("Image URL", max_width=40)
        table.add_column("Instruction", max_width=40)
    else:
        table.add_column("Prompt", max_width=60)

    table.add_column("Metadata", style="dim")
    table.add_column("Expected", style="cyan")

    for i, item in enumerate(items):
        row: list[str] = [str(i + 1)]
        if pipeline == "img2img":
            img_url = item.get("image_url", "")
            if len(img_url) > 40:
                img_url = img_url[:37] + "..."
            row.append(img_url)
            row.append(item.get("instruction", item.get("prompt", "")))
        else:
            row.append(item.get("prompt", ""))

        meta = item.get("metadata", {})
        row.append(", ".join(f"{k}={v}" for k, v in meta.items()) if meta else "")

        expected = item.get("expected", {})
        if expected:
            parts = [f"{k}:{v}" for k, v in expected.items()]
            row.append(", ".join(parts))
        else:
            row.append("")

        table.add_row(*row)

    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


@dataset_group.command("add")
@click.argument("path")
@click.option("--prompt", default=None, help="Prompt text (text2img).")
@click.option("--image", default=None, help="Input image URL (img2img).")
@click.option("--instruction", default=None, help="Edit instruction (img2img).")
@click.option("--metadata", "-m", multiple=True, help="Metadata key=value pairs.")
@click.option("--expected", "-e", multiple=True, help="Expected scores dim:value pairs.")
def dataset_add(
    path: str,
    prompt: str | None,
    image: str | None,
    instruction: str | None,
    metadata: tuple[str, ...],
    expected: tuple[str, ...],
) -> None:
    """Add an item to an existing dataset."""
    if not prompt and not image:
        console.print("[bold red]Error:[/bold red] --prompt or --image is required.\n")
        sys.exit(2)

    p = Path(path)
    if not p.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {path}\n")
        sys.exit(2)

    # Read raw file to preserve original format
    try:
        with open(p) as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as exc:
        console.print(f"[bold red]Error:[/bold red] Invalid JSON in {path}: {exc}\n")
        sys.exit(2)

    # Build new item
    item: dict[str, Any] = {}
    if prompt:
        item["prompt"] = prompt
    if image:
        item["image_url"] = image
    if instruction:
        item["instruction"] = instruction

    # Parse metadata
    if metadata:
        meta: dict[str, str] = {}
        for pair in metadata:
            if "=" not in pair:
                console.print(f"[bold red]Error:[/bold red] Invalid metadata format: {pair} (expected key=value)\n")
                sys.exit(2)
            k, v = pair.split("=", 1)
            meta[k] = v
        item["metadata"] = meta

    # Parse expected scores
    if expected:
        exp: dict[str, float] = {}
        for pair in expected:
            if ":" not in pair:
                console.print(f"[bold red]Error:[/bold red] Invalid expected format: {pair} (expected dim:value)\n")
                sys.exit(2)
            k, v = pair.split(":", 1)
            try:
                exp[k] = float(v)
            except ValueError:
                console.print(f"[bold red]Error:[/bold red] Invalid score value: {v}\n")
                sys.exit(2)
        item["expected"] = exp

    # Append to the correct key in raw data
    if isinstance(raw_data, list):
        raw_data.append(item)
    elif isinstance(raw_data, dict):
        if "prompts" in raw_data:
            raw_data["prompts"].append(item)
        elif "inputs" in raw_data:
            raw_data["inputs"].append(item)
        elif "items" in raw_data:
            raw_data["items"].append(item)
        else:
            raw_data.setdefault("items", []).append(item)
    else:
        raw_data = {"items": [item]}

    _write_dataset(raw_data, path)

    # Count items for display
    ds = _normalize_dataset(raw_data)
    count = len(ds.get("items", []))
    console.print(f"  Added item. Dataset now has {count} items.\n")


# ---------------------------------------------------------------------------
# from-bench
# ---------------------------------------------------------------------------


@dataset_group.command("from-bench")
@click.argument("report_path")
@click.option("--min-score", default=None, type=float, help="Minimum overall score to include an item.")
@click.option("--model", default=None, help="Use scores from this specific model.")
@click.option("--name", "-n", default=None, help="Dataset name.")
@click.option("--output", "-o", required=True, help="Output file path.")
def dataset_from_bench(
    report_path: str,
    min_score: float | None,
    model: str | None,
    name: str | None,
    output: str,
) -> None:
    """Create a dataset from a bench report JSON with expected scores."""
    p = Path(report_path)
    if not p.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {report_path}\n")
        sys.exit(2)

    try:
        with open(p) as f:
            report = json.load(f)
    except json.JSONDecodeError as exc:
        console.print(f"[bold red]Error:[/bold red] Invalid JSON: {exc}\n")
        sys.exit(2)

    report_items = report.get("items", [])
    if not report_items:
        console.print("[bold red]Error:[/bold red] Report has no items.\n")
        sys.exit(2)

    # Determine pipeline from report config
    pipeline = report.get("config", {}).get("pipeline", "")

    # Determine which model to use for scores
    available_models = list(report.get("summary", {}).keys())
    if model:
        if model not in available_models:
            console.print(f"[bold red]Error:[/bold red] Model '{model}' not found in report. Available: {', '.join(available_models)}\n")
            sys.exit(2)
        target_model = model
    else:
        # Use the winner or first model
        target_model = report.get("winner", "")
        if not target_model and available_models:
            target_model = available_models[0]

    # Build dataset items
    ds_items: list[dict[str, Any]] = []
    for report_item in report_items:
        result = report_item.get("results", {}).get(target_model, {})
        scores = result.get("scores", {})

        # Calculate overall score
        if scores:
            score_values = [s.get("score", 0) if isinstance(s, dict) else s for s in scores.values()]
            overall = sum(score_values) / len(score_values) if score_values else 0.0
        else:
            overall = result.get("overall_score", 0.0)

        # Apply min-score filter
        if min_score is not None and overall < min_score:
            continue

        ds_item: dict[str, Any] = {}
        if report_item.get("prompt"):
            ds_item["prompt"] = report_item["prompt"]
        if report_item.get("image_url"):
            ds_item["image_url"] = report_item["image_url"]
        if report_item.get("instruction"):
            ds_item["instruction"] = report_item["instruction"]

        # Auto-detect pipeline
        if not pipeline:
            if "image_url" in ds_item and "instruction" in ds_item:
                pipeline = "img2img"
            else:
                pipeline = "text2img"

        # Write expected scores from the target model
        if scores:
            expected: dict[str, float] = {}
            for dim, score_data in scores.items():
                if isinstance(score_data, dict):
                    expected[dim] = score_data.get("score", 0.0)
                else:
                    expected[dim] = float(score_data)
            if expected:
                expected["overall"] = round(overall, 2)
                ds_item["expected"] = expected

        ds_items.append(ds_item)

    if not ds_items:
        console.print("[bold red]Error:[/bold red] No items pass the filter criteria.\n")
        sys.exit(2)

    ds_name = name or f"from-{p.stem}"
    dataset: dict[str, Any] = {
        "name": ds_name,
        "description": f"Generated from bench report: {p.name}",
        "pipeline": pipeline,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_report": report_path,
        "items": ds_items,
    }

    _write_dataset(dataset, output)
    console.print(f"  Created dataset with {len(ds_items)} items: [bold]{output}[/bold]")
    console.print(f"  Source model: {target_model}\n")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@dataset_group.command("validate")
@click.argument("path")
def dataset_validate(path: str) -> None:
    """Validate a dataset file."""
    data = _load_dataset(path)
    warnings = _validate_dataset(data)

    if warnings:
        console.print(f"\n  [yellow]Warnings ({len(warnings)}):[/yellow]")
        for w in warnings:
            console.print(f"    [yellow]- {w}[/yellow]")
        console.print()
    else:
        items = data.get("items", [])
        console.print(f"\n  [green]Valid[/green] -- {len(items)} items, no issues found.\n")


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


@dataset_group.command("stats")
@click.argument("path")
def dataset_stats(path: str) -> None:
    """Show dataset statistics."""
    data = _load_dataset(path)
    items = data.get("items", [])
    ds_name = data.get("name", Path(path).stem)
    pipeline = data.get("pipeline", "unknown")

    console.print(f"\n  [bold]{ds_name}[/bold]  |  {pipeline}")
    console.print(f"  Items: {len(items)}")

    # Count items with expected scores
    items_with_expected = [item for item in items if item.get("expected")]
    console.print(f"  Items with expected scores: {len(items_with_expected)}")

    # Expected score statistics
    if items_with_expected:
        dim_scores: dict[str, list[float]] = {}
        for item in items_with_expected:
            for dim, score in item.get("expected", {}).items():
                if isinstance(score, (int, float)):
                    dim_scores.setdefault(dim, []).append(score)

        if dim_scores:
            console.print()
            table = Table(title="Expected Score Statistics", show_lines=False, pad_edge=True)
            table.add_column("Dimension", style="bold")
            table.add_column("Min", justify="right")
            table.add_column("Avg", justify="right")
            table.add_column("Max", justify="right")
            table.add_column("Count", justify="right", style="dim")

            for dim in sorted(dim_scores.keys()):
                scores = dim_scores[dim]
                table.add_row(
                    dim,
                    f"{min(scores):.2f}",
                    f"{sum(scores) / len(scores):.2f}",
                    f"{max(scores):.2f}",
                    str(len(scores)),
                )
            console.print(table)

    # Metadata key distribution
    meta_keys: dict[str, int] = {}
    for item in items:
        for k in item.get("metadata", {}):
            meta_keys[k] = meta_keys.get(k, 0) + 1

    if meta_keys:
        console.print()
        meta_table = Table(title="Metadata Keys", show_lines=False, pad_edge=True)
        meta_table.add_column("Key", style="bold")
        meta_table.add_column("Count", justify="right")

        for k, count in sorted(meta_keys.items(), key=lambda x: -x[1]):
            meta_table.add_row(k, str(count))
        console.print(meta_table)

    console.print()
