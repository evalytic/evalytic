"""Dataset command coverage for rag/text/agent dataset types."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from evalytic.cli.main import cli


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def test_dataset_show_rag_type(runner: CliRunner, tmp_path: Path) -> None:
    data = {
        "type": "rag",
        "name": "rag-set",
        "items": [
            {
                "query": "What is Evalytic?",
                "response": "Evalytic evaluates AI outputs.",
                "contexts": [{"text": "Evalytic evaluates AI outputs."}],
            }
        ],
    }
    path = tmp_path / "rag.json"
    path.write_text(json.dumps(data))

    result = runner.invoke(cli, ["dataset", "show", str(path)])

    assert result.exit_code == 0
    assert "rag-set" in result.output
    assert "What is Evalytic?" in result.output


def test_dataset_validate_text_type(runner: CliRunner, tmp_path: Path) -> None:
    data = {
        "type": "text",
        "items": [{"input": "Summarize this."}],
    }
    path = tmp_path / "text.json"
    path.write_text(json.dumps(data))

    result = runner.invoke(cli, ["dataset", "validate", str(path)])

    assert result.exit_code == 0
    assert "missing output for text dataset" in result.output


def test_dataset_validate_agent_type(runner: CliRunner, tmp_path: Path) -> None:
    data = {
        "type": "agent",
        "items": [{"input": "Do the task", "final_output": "Done"}],
    }
    path = tmp_path / "agent.json"
    path.write_text(json.dumps(data))

    result = runner.invoke(cli, ["dataset", "validate", str(path)])

    assert result.exit_code == 0
    assert "Valid" in result.output


# ------------------------------------------------------------------
# Additional dataset coverage (C10)
# ------------------------------------------------------------------


def test_dataset_legacy_text2img_without_type_detected(runner: CliRunner, tmp_path: Path) -> None:
    """Legacy bare list of prompts still works (backward compat)."""
    data = ["A cat in a hat", "A dog in the park"]
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(data))

    result = runner.invoke(cli, ["dataset", "show", str(path)])
    assert result.exit_code == 0


def test_dataset_pipeline_field_respected(runner: CliRunner, tmp_path: Path) -> None:
    """A legacy dataset with `pipeline: img2img` should keep working."""
    data = {
        "pipeline": "img2img",
        "inputs": [{"image_url": "https://example.com/1.jpg", "prompt": "edit"}],
    }
    path = tmp_path / "legacy-pipeline.json"
    path.write_text(json.dumps(data))

    result = runner.invoke(cli, ["dataset", "validate", str(path)])
    assert result.exit_code == 0


def test_dataset_rag_missing_contexts_warning(runner: CliRunner, tmp_path: Path) -> None:
    data = {
        "type": "rag",
        "items": [{"query": "Q?", "response": "A"}],  # missing contexts
    }
    path = tmp_path / "rag-incomplete.json"
    path.write_text(json.dumps(data))

    result = runner.invoke(cli, ["dataset", "validate", str(path)])
    assert result.exit_code == 0  # warnings, not errors
    assert "missing contexts" in result.output


def test_dataset_rag_valid_no_warnings(runner: CliRunner, tmp_path: Path) -> None:
    data = {
        "type": "rag",
        "items": [
            {
                "query": "Q?",
                "response": "A",
                "contexts": [{"text": "support", "rank": 1}],
            }
        ],
    }
    path = tmp_path / "rag-ok.json"
    path.write_text(json.dumps(data))

    result = runner.invoke(cli, ["dataset", "validate", str(path)])
    assert result.exit_code == 0
    assert "Valid" in result.output


def test_dataset_stats_runs_for_rag(runner: CliRunner, tmp_path: Path) -> None:
    data = {
        "type": "rag",
        "items": [
            {
                "query": f"Q{i}",
                "response": f"A{i}",
                "contexts": [{"text": f"ctx{i}"}],
            }
            for i in range(3)
        ],
    }
    path = tmp_path / "rag-stats.json"
    path.write_text(json.dumps(data))

    result = runner.invoke(cli, ["dataset", "stats", str(path)])
    assert result.exit_code == 0
