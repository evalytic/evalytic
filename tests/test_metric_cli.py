"""CLI tests for rag/text/agent eval entry points."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from evalytic.agent.types import AgentTestCase
from evalytic.cli.main import cli
from evalytic.text.types import (
    MetricEvalReport,
    MetricEvalResult,
    MetricResult,
    RAGTestCase,
    RetrievedChunk,
    TextTestCase,
)


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def test_rag_eval_cli_writes_report(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import evalytic.cli.rag_cmd as rag_cmd

    def fake_evaluate_rag(*args, **kwargs) -> MetricEvalReport:
        case = RAGTestCase(
            query="What is Evalytic?",
            response="Evalytic evaluates AI outputs.",
            contexts=[RetrievedChunk(text="Evalytic evaluates AI outputs.")],
        )
        return MetricEvalReport(
            eval_type="rag",
            judge="fake-judge",
            results=[
                MetricEvalResult(
                    case_id="case-1",
                    test_case=case,
                    metrics=[
                        MetricResult(metric_id="faithfulness", score=0.9, reason="Supported."),
                        MetricResult(metric_id="answer_relevancy", score=0.8, reason="Relevant."),
                    ],
                    total_cost=0.0,
                    duration_ms=5,
                )
            ],
        )

    monkeypatch.setattr(rag_cmd, "evaluate_rag", fake_evaluate_rag)
    out_file = tmp_path / "rag-report.json"
    result = runner.invoke(
        cli,
        [
            "rag",
            "eval",
            "--query",
            "What is Evalytic?",
            "--response",
            "Evalytic evaluates AI outputs.",
            "--context",
            "Evalytic evaluates AI outputs.",
            "--output",
            str(out_file),
        ],
    )
    assert result.exit_code == 0
    data = json.loads(out_file.read_text())
    assert data["eval_type"] == "rag"
    assert data["summary"]["metric_averages"]["faithfulness"] == 0.9


def test_text_eval_cli_writes_report(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import evalytic.cli.text_cmd as text_cmd

    def fake_evaluate_text(*args, **kwargs) -> MetricEvalReport:
        case = TextTestCase(
            input="Summarize Evalytic.",
            output="Evalytic evaluates AI outputs.",
            expected="Evalytic evaluates AI outputs.",
        )
        return MetricEvalReport(
            eval_type="text",
            judge="fake-judge",
            results=[
                MetricEvalResult(
                    case_id="case-1",
                    test_case=case,
                    metrics=[
                        MetricResult(metric_id="semantic_similarity", score=1.0, reason="Match."),
                    ],
                    total_cost=0.0,
                    duration_ms=3,
                )
            ],
        )

    monkeypatch.setattr(text_cmd, "evaluate_text", fake_evaluate_text)
    out_file = tmp_path / "text-report.json"
    result = runner.invoke(
        cli,
        [
            "text",
            "eval",
            "--input",
            "Summarize Evalytic.",
            "--output-text",
            "Evalytic evaluates AI outputs.",
            "--expected",
            "Evalytic evaluates AI outputs.",
            "--output",
            str(out_file),
        ],
    )
    assert result.exit_code == 0
    data = json.loads(out_file.read_text())
    assert data["eval_type"] == "text"
    assert data["summary"]["metric_averages"]["semantic_similarity"] == 1.0


def test_agent_eval_cli_writes_report(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import evalytic.cli.agent_cmd as agent_cmd

    def fake_evaluate_agent(*args, **kwargs) -> MetricEvalReport:
        case = AgentTestCase(
            input="Look up the score.",
            final_output="The score is 0.9.",
        )
        return MetricEvalReport(
            eval_type="agent",
            judge="fake-judge",
            results=[
                MetricEvalResult(
                    case_id="case-1",
                    test_case=case,
                    metrics=[
                        MetricResult(metric_id="tool_call_accuracy", score=1.0, reason="Matched."),
                        MetricResult(metric_id="goal_accuracy", score=0.9, reason="Goal met."),
                        MetricResult(metric_id="step_efficiency", score=0.8, reason="Reasonable."),
                    ],
                    total_cost=0.0,
                    duration_ms=2,
                )
            ],
        )

    monkeypatch.setattr(agent_cmd, "evaluate_agent", fake_evaluate_agent)
    out_file = tmp_path / "agent-report.json"
    result = runner.invoke(
        cli,
        [
            "agent",
            "eval",
            "--input",
            "Look up the score.",
            "--final-output",
            "The score is 0.9.",
            "--output",
            str(out_file),
        ],
    )
    assert result.exit_code == 0
    data = json.loads(out_file.read_text())
    assert data["eval_type"] == "agent"
    assert data["summary"]["metric_averages"]["goal_accuracy"] == 0.9


# ------------------------------------------------------------------
# CLI negative paths (C11)
# ------------------------------------------------------------------


def test_rag_eval_requires_query_when_no_dataset(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["rag", "eval"])
    assert result.exit_code == 2
    assert "--query" in result.output or "required" in result.output


def test_rag_eval_requires_context(runner: CliRunner) -> None:
    result = runner.invoke(
        cli,
        ["rag", "eval", "--query", "Q?", "--response", "A"],
    )
    assert result.exit_code == 2
    assert "context" in result.output.lower() or "required" in result.output.lower()


def test_rag_eval_dataset_type_mismatch_exits_2(runner: CliRunner, tmp_path: Path) -> None:
    dataset = tmp_path / "text-not-rag.json"
    dataset.write_text(json.dumps({"type": "text", "items": [{"input": "i", "output": "o"}]}))
    result = runner.invoke(cli, ["rag", "eval", "--dataset", str(dataset)])
    assert result.exit_code == 2
    assert "mismatch" in result.output.lower()


def test_rag_eval_unknown_metric_exits_2(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Prevent real TextJudge construction
    import evalytic.text.runner as text_runner

    class DummyJudge:
        judge_string = "dummy"

        def __init__(self, *_a, **_kw) -> None:
            pass

        def complete_json(self, prompt: str, *_a, **_kw) -> dict:
            return {}

        def close(self) -> None:
            return None

    monkeypatch.setattr(text_runner, "TextJudge", DummyJudge)

    result = runner.invoke(
        cli,
        [
            "rag", "eval",
            "--query", "Q?",
            "--response", "A",
            "--context", "ctx",
            "--metrics", "definitely_not_a_metric",
        ],
    )
    assert result.exit_code == 2


def test_agent_eval_missing_final_output_click_error(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["agent", "eval", "--input", "find"])
    assert result.exit_code != 0
    assert "final-output" in result.output.lower() or "usage" in result.output.lower()
