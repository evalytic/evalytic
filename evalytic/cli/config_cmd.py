"""``evalytic config`` command group -- configuration management."""

from __future__ import annotations

import os
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()

_KEY_MAP = [
    ("FAL_KEY", "fal"),
    ("GEMINI_API_KEY", "gemini"),
    ("OPENAI_API_KEY", "openai"),
    ("ANTHROPIC_API_KEY", "anthropic"),
]


def _mask_key(key: str) -> str:
    """Mask an API key: first 4 + *** + last 4."""
    if len(key) <= 8:
        return "***"
    return key[:4] + "***" + key[-4:]


def _detect_source(env_var: str, toml_key: str, config: dict) -> str:
    """Detect where a key value came from."""
    toml_keys = config.get("keys", {})
    toml_val = toml_keys.get(toml_key)
    env_val = os.environ.get(env_var)

    if not env_val:
        return "--"

    # Check .env file
    dotenv_path = Path(".env")
    if dotenv_path.exists():
        try:
            content = dotenv_path.read_text()
            if f"{env_var}=" in content:
                return ".env"
        except OSError:
            pass

    # Check evalytic.toml
    if toml_val and env_val == toml_val:
        return "evalytic.toml"

    return "environment"


@click.group("config")
def config_group() -> None:
    """Configuration management."""


@config_group.command("show")
@click.pass_context
def config_show(ctx: click.Context) -> None:
    """Show active configuration and API key status."""
    config = (ctx.obj or {}).get("config", {})
    bench_cfg = config.get("bench", {})

    table = Table(title="Active Configuration", show_lines=False, pad_edge=True)
    table.add_column("Key / Setting", style="bold")
    table.add_column("Value")
    table.add_column("Source", style="dim")

    # API keys
    for env_var, toml_key in _KEY_MAP:
        val = os.environ.get(env_var)
        source = _detect_source(env_var, toml_key, config)
        if val:
            display = _mask_key(val)
        else:
            display = "(not set)"
        table.add_row(env_var, display, source)

    # Bench settings
    table.add_section()
    from .eval_cmd import _resolve_default_judge
    judge = _resolve_default_judge(config)
    judge_source = "evalytic.toml" if "judge" in bench_cfg else "auto"
    table.add_row("judge", judge, judge_source)

    concurrency = bench_cfg.get("concurrency")
    if concurrency:
        table.add_row("concurrency", str(concurrency), "evalytic.toml")

    console.print()
    console.print(table)

    # Config file path
    config_paths = [Path("evalytic.toml"), Path.home() / ".evalytic" / "config.toml"]
    found = next((p for p in config_paths if p.exists()), None)
    if found:
        console.print(f"\n  Config file: {found.resolve()}")
    else:
        console.print("\n  Config file: (none found)")
        console.print("  Run [bold]evalytic init[/bold] to create one.")
    console.print()
