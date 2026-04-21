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

DATASET_TYPES = ["text2img", "img2img", "rag", "text", "agent"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_dataset_type(data: Any) -> str:
    """Detect the canonical dataset type."""
    if isinstance(data, dict):
        if data.get("type"):
            return str(data["type"])
        if data.get("pipeline"):
            return str(data["pipeline"])
        if "inputs" in data:
            return "img2img"
        if "prompts" in data:
            return "text2img"
        items = data.get("items")
        sample = items[0] if isinstance(items, list) and items else data
    elif isinstance(data, list) and data:
        sample = data[0]
    else:
        return "text2img"

    if isinstance(sample, dict):
        if "query" in sample and "response" in sample:
            return "rag"
        if "input" in sample and "final_output" in sample:
            return "agent"
        if "input" in sample and "output" in sample:
            return "text"
        if "image_url" in sample:
            return "img2img"
        if "prompt" in sample:
            return "text2img"
    if isinstance(sample, str):
        return "text2img"
    return "text2img"


def _normalize_dataset(data: Any) -> dict[str, Any]:
    """Normalize all supported formats into ``{"type": "...", "items": [...]}``.

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
        return {"type": _detect_dataset_type(items), "items": items}

    if not isinstance(data, dict):
        return {"type": "text2img", "items": []}

    # Already canonical
    if "items" in data:
        result = dict(data)
        dataset_type = _detect_dataset_type(result)
        result.setdefault("type", dataset_type)
        if dataset_type in ("text2img", "img2img"):
            result.setdefault("pipeline", dataset_type)
        return result

    result = {k: v for k, v in data.items() if k not in ("prompts", "inputs")}

    if "prompts" in data:
        items = data["prompts"]
        result["items"] = [
            {"prompt": item} if isinstance(item, str) else item
            for item in items
        ]
        result.setdefault("type", "text2img")
        result.setdefault("pipeline", "text2img")
    elif "inputs" in data:
        result["items"] = data["inputs"]
        result.setdefault("type", "img2img")
        result.setdefault("pipeline", "img2img")
    else:
        result["items"] = []
        result.setdefault("type", _detect_dataset_type(result))

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
    dataset_type = data.get("type") or data.get("pipeline") or _detect_dataset_type(data)

    if not items:
        warnings.append("Dataset has no items.")

    for i, item in enumerate(items):
        if not isinstance(item, dict):
            warnings.append(f"Item {i}: must be an object.")
            continue

        if dataset_type == "text2img":
            has_image = "image_url" in item
            has_prompt = "prompt" in item
            if has_image and not has_prompt:
                warnings.append(f"Item {i}: has image_url but type is text2img.")
            if not has_prompt:
                warnings.append(f"Item {i}: missing prompt for text2img dataset.")
        elif dataset_type == "img2img":
            if "image_url" not in item:
                warnings.append(f"Item {i}: missing image_url for img2img dataset.")
        elif dataset_type == "rag":
            if "query" not in item:
                warnings.append(f"Item {i}: missing query for rag dataset.")
            if "response" not in item:
                warnings.append(f"Item {i}: missing response for rag dataset.")
            if not item.get("contexts"):
                warnings.append(f"Item {i}: missing contexts for rag dataset.")
        elif dataset_type == "text":
            if "input" not in item:
                warnings.append(f"Item {i}: missing input for text dataset.")
            if "output" not in item:
                warnings.append(f"Item {i}: missing output for text dataset.")
        elif dataset_type == "agent":
            if "input" not in item:
                warnings.append(f"Item {i}: missing input for agent dataset.")
            if "final_output" not in item:
                warnings.append(f"Item {i}: missing final_output for agent dataset.")

        # Validate expected scores where expected is numeric map
        expected = item.get("expected")
        if isinstance(expected, dict):
            score_max = 5.0 if dataset_type in ("text2img", "img2img") else 1.0
            for dim, score in expected.items():
                if not isinstance(score, (int, float)):
                    warnings.append(f"Item {i}: expected[{dim}] is not a number.")
                elif score < 0 or score > score_max:
                    warnings.append(
                        f"Item {i}: expected[{dim}]={score} is outside 0-{int(score_max) if score_max.is_integer() else score_max} range."
                    )

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


def _item_label(dataset_type: str) -> str:
    return {
        "text2img": "prompts",
        "img2img": "images",
        "rag": "queries",
        "text": "cases",
        "agent": "runs",
    }.get(dataset_type, "items")


def _truncate(value: Any, max_len: int = 60) -> str:
    text = str(value or "")
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _expected_cell(expected: Any) -> str:
    if isinstance(expected, dict):
        parts = [f"{k}:{v}" for k, v in expected.items()]
        return ", ".join(parts)
    if expected is None:
        return ""
    return _truncate(expected, 50)


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
    "--type", "dataset_type",
    type=click.Choice(DATASET_TYPES),
    default=None,
    help="Canonical dataset type.",
)
@click.option(
    "--pipeline", "legacy_pipeline",
    type=click.Choice(["text2img", "img2img"]),
    default=None,
    help="Legacy alias for visual dataset types.",
)
@click.option("--description", "-d", default="", help="Dataset description.")
@click.option("--output", "-o", default=None, help="Output file path (default: <name>.json).")
def dataset_create(
    name: str,
    dataset_type: str | None,
    legacy_pipeline: str | None,
    description: str,
    output: str | None,
) -> None:
    """Create a new empty dataset file."""
    out_path = output or f"{name}.json"
    if Path(out_path).exists():
        console.print(f"[bold red]Error:[/bold red] File already exists: {out_path}\n")
        sys.exit(2)
    if dataset_type and legacy_pipeline and dataset_type != legacy_pipeline:
        console.print("[bold red]Error:[/bold red] --type and --pipeline disagree.\n")
        sys.exit(2)
    resolved_type = dataset_type or legacy_pipeline or "text2img"

    data: dict[str, Any] = {
        "name": name,
        "description": description,
        "type": resolved_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "items": [],
    }
    if resolved_type in ("text2img", "img2img"):
        data["pipeline"] = resolved_type
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
    dataset_type = data.get("type") or data.get("pipeline") or _detect_dataset_type(data)
    ds_name = data.get("name", Path(path).stem)

    console.print(f"\n  [bold]{ds_name}[/bold]  |  {dataset_type}  |  {len(items)} items\n")

    if not items:
        console.print("  (empty dataset)\n")
        return

    table = Table(show_lines=False, pad_edge=True)
    table.add_column("#", style="dim", justify="right")

    if dataset_type == "img2img":
        table.add_column("Image URL", max_width=40)
        table.add_column("Instruction", max_width=40)
    elif dataset_type == "rag":
        table.add_column("Query", max_width=36)
        table.add_column("Response", max_width=36)
        table.add_column("Contexts", justify="right")
    elif dataset_type == "text":
        table.add_column("Input", max_width=36)
        table.add_column("Output", max_width=36)
        table.add_column("Expected", max_width=24)
    elif dataset_type == "agent":
        table.add_column("Input", max_width=32)
        table.add_column("Final Output", max_width=32)
        table.add_column("Tools", justify="right")
    else:
        table.add_column("Prompt", max_width=60)

    table.add_column("Metadata", style="dim")
    if dataset_type not in ("text",):
        table.add_column("Expected", style="cyan")

    for i, item in enumerate(items):
        row: list[str] = [str(i + 1)]
        if dataset_type == "img2img":
            img_url = item.get("image_url", "")
            if len(img_url) > 40:
                img_url = img_url[:37] + "..."
            row.append(img_url)
            row.append(item.get("instruction", item.get("prompt", "")))
        elif dataset_type == "rag":
            row.append(_truncate(item.get("query", ""), 36))
            row.append(_truncate(item.get("response", ""), 36))
            row.append(str(len(item.get("contexts", []) or [])))
        elif dataset_type == "text":
            row.append(_truncate(item.get("input", ""), 36))
            row.append(_truncate(item.get("output", ""), 36))
            row.append(_expected_cell(item.get("expected")))
        elif dataset_type == "agent":
            row.append(_truncate(item.get("input", ""), 32))
            row.append(_truncate(item.get("final_output", ""), 32))
            row.append(str(len(item.get("tool_calls", []) or [])))
        else:
            row.append(item.get("prompt", ""))

        meta = item.get("metadata", {})
        row.append(", ".join(f"{k}={v}" for k, v in meta.items()) if meta else "")

        if dataset_type not in ("text",):
            row.append(_expected_cell(item.get("expected")))

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
    if report.get("eval_type", "bench") != "bench":
        console.print("[bold red]Error:[/bold red] dataset from-bench only supports bench reports.\n")
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
        "type": pipeline,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_report": report_path,
        "items": ds_items,
    }
    if pipeline in ("text2img", "img2img"):
        dataset["pipeline"] = pipeline

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
    dataset_type = data.get("type") or data.get("pipeline") or _detect_dataset_type(data)

    console.print(f"\n  [bold]{ds_name}[/bold]  |  {dataset_type}")
    console.print(f"  Items: {len(items)}")

    items_with_expected = [item for item in items if item.get("expected") is not None]
    console.print(f"  Items with expected scores: {len(items_with_expected)}")

    if dataset_type == "rag":
        context_counts = [len(item.get("contexts", []) or []) for item in items if isinstance(item, dict)]
        if context_counts:
            console.print(f"  Avg contexts per item: {sum(context_counts) / len(context_counts):.1f}")
    elif dataset_type == "agent":
        tool_counts = [len(item.get("tool_calls", []) or []) for item in items if isinstance(item, dict)]
        if tool_counts:
            console.print(f"  Avg tool calls per run: {sum(tool_counts) / len(tool_counts):.1f}")

    # Expected score statistics
    if items_with_expected:
        dim_scores: dict[str, list[float]] = {}
        for item in items_with_expected:
            expected = item.get("expected")
            if not isinstance(expected, dict):
                continue
            for dim, score in expected.items():
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
