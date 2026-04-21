"""Evalytic CLI entry point."""

from __future__ import annotations

import click
from dotenv import load_dotenv

from .bench import bench
from .compare_cmd import compare_cmd
from .config_cmd import config_group
from .dataset_cmd import dataset_group
from .demo_cmd import demo_cmd
from .eval_cmd import eval_cmd
from .gate import gate
from .init_cmd import init_cmd
from .rag_cmd import rag_group
from .text_cmd import text_group
from .agent_cmd import agent_group


@click.group(invoke_without_command=True)
@click.version_option(package_name="evalytic")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output.")
@click.option("--config", "config_path", default=None, help="Path to evalytic.toml config file.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, config_path: str | None) -> None:
    """Evalytic -- Evals for AI outputs."""
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
  Evalytic v{__version__} -- Evals for AI outputs

  Quick Start:
    1. See what Evalytic can do (no API key needed):
       evalytic demo

    2. Set API keys:
       export FAL_KEY=...          # fal.ai (image generation)
       export GEMINI_API_KEY=...   # Google Gemini (free, for scoring)

    3. Run your first benchmark:
       evalytic bench -m flux-schnell -p "A cat in a top hat" -y

    4. Evaluate a RAG answer:
       evalytic rag eval --query "What is Evalytic?" --response "..." --context "..."

    5. Compare reports:
       evalytic compare --baseline run-a.json --candidate run-b.json

  Commands:
    bench    Benchmark image/video generation models
    eval     Score an existing image (no generation)
    rag      Evaluate RAG answers and retrieval quality
    text     Evaluate text outputs against references or criteria
    agent    Evaluate tool-using agent runs
    compare  Compare two report files
    gate     CI/CD quality gate with exit codes
    demo     Open real benchmark showcase
    dataset  Manage evaluation datasets
    init     Interactive setup wizard
    config   Configuration management

  Run evalytic <command> --help for details.
  Docs: https://docs.evalytic.ai
""")


cli.add_command(bench)
cli.add_command(eval_cmd)
cli.add_command(rag_group)
cli.add_command(text_group)
cli.add_command(agent_group)
cli.add_command(compare_cmd)
cli.add_command(gate)
cli.add_command(demo_cmd)
cli.add_command(dataset_group)
cli.add_command(init_cmd)
cli.add_command(config_group)
