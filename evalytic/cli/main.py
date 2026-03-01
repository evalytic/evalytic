"""Evalytic CLI entry point."""

from __future__ import annotations

import click
from dotenv import load_dotenv

from .bench import bench
from .config_cmd import config_group
from .dataset_cmd import dataset_group
from .eval_cmd import eval_cmd
from .gate import gate
from .init_cmd import init_cmd


@click.group(invoke_without_command=True)
@click.version_option(package_name="evalytic")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output.")
@click.option("--config", "config_path", default=None, help="Path to evalytic.toml config file.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, config_path: str | None) -> None:
    """Evalytic -- Evals for visual AI."""
    load_dotenv()

    from ..config import apply_keys, load_config

    config = load_config(config_path)
    apply_keys(config)

    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["config"] = config

    if ctx.invoked_subcommand is None:
        from .. import __version__

        click.echo(f"""
  Evalytic v{__version__} -- Evals for visual AI

  Quick Start:
    1. Set API keys:
       export FAL_KEY=...          # fal.ai (image generation)
       export GEMINI_API_KEY=...   # Google Gemini (free, for scoring)

    2. Run your first benchmark:
       evalytic bench -m flux-schnell -p "A cat in a top hat" -y

    3. Compare models:
       evalytic bench -m flux-schnell -m flux-dev -p "A cat" -y

  Commands:
    bench    Benchmark models: generate, score, report
    eval     Score an existing image (no generation)
    gate     CI/CD quality gate with exit codes
    dataset  Manage evaluation datasets
    init     Interactive setup wizard
    config   Configuration management

  Run evalytic <command> --help for details.
  Docs: https://docs.evalytic.ai
""")


cli.add_command(bench)
cli.add_command(eval_cmd)
cli.add_command(gate)
cli.add_command(dataset_group)
cli.add_command(init_cmd)
cli.add_command(config_group)
