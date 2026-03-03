"""``evalytic demo`` — open showcase benchmark reports in browser."""

from __future__ import annotations

import webbrowser

import click
from rich.console import Console
from rich.table import Table

console = Console()

SHOWCASE_URL = "https://docs.evalytic.ai/showcase"

_DEMOS = {
    "flagship": {
        "title": "The $0.003 Question",
        "models": "Flux Schnell · Dev · Pro · Recraft · Ideogram",
        "finding": "Same price band, 0.5-point quality gap",
        "url": "https://docs.evalytic.ai/demo/flagship.html",
    },
    "face": {
        "title": "That's Not Me",
        "models": "5 img2img models, ArcFace metric",
        "finding": "r=0.99 VLM↔metric correlation",
        "url": "https://docs.evalytic.ai/demo/face.html",
    },
    "prompt-trap": {
        "title": "The Prompt Trap",
        "models": "Simple vs detailed prompts",
        "finding": "Visual quality tied, fidelity ≠ tied",
        "url": "https://docs.evalytic.ai/demo/prompt-trap.html",
    },
    "product": {
        "title": "Is That Still My Product?",
        "models": "Seedream · Flux Kontext · FireRed · Reve",
        "finding": "Input fidelity as the differentiator",
        "url": "https://docs.evalytic.ai/demo/product.html",
    },
}


@click.command("demo")
@click.argument(
    "case",
    required=False,
    default=None,
    type=click.Choice(list(_DEMOS.keys())),
)
def demo_cmd(case: str | None) -> None:
    """Open real benchmark showcase in your browser."""
    console.print()
    console.print("  [bold]Evalytic Benchmark Showcase[/bold]")
    console.print(
        "  [dim]Real data from 9 models · $0.55 total cost · Gemini 2.5 Flash judge[/dim]"
    )
    console.print()

    table = Table(show_header=True, show_lines=False, padding=(0, 2))
    table.add_column("#", style="dim", width=3)
    table.add_column("Case", style="bold")
    table.add_column("Models", style="dim")
    table.add_column("Key Finding")

    for i, (key, info) in enumerate(_DEMOS.items(), 1):
        marker = " ←" if key == case else ""
        table.add_row(
            f"0{i}",
            info["title"],
            info["models"],
            info["finding"] + marker,
        )
    console.print(table)
    console.print()

    if case:
        demo = _DEMOS[case]
        url = demo["url"]
        console.print(f"  Opening → [link={url}]{url}[/link]")
        webbrowser.open(url)
    else:
        console.print(f"  Opening → [link={SHOWCASE_URL}]{SHOWCASE_URL}[/link]")
        webbrowser.open(SHOWCASE_URL)

    console.print()
    console.print("  [dim]─────────────────────────────────────────────[/dim]")
    console.print(
        '  Run your own:  [bold]evalytic bench -m flux-schnell -p "A cat" -y[/bold]'
    )
    console.print(
        "  Full docs:     [link=https://docs.evalytic.ai]https://docs.evalytic.ai[/link]"
    )
    console.print()
