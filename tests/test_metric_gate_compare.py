"""Tests for metric-first gate and compare flows."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from evalytic.cli.main import cli


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def rag_report(tmp_path: Path) -> Path:
    data = {
        "eval_type": "rag",
        "summary": {
            "total_cases": 2,
            "metric_averages": {
                "faithfulness": 0.82,
                "answer_relevancy": 0.77,
            },
        },
        "results": [],
    }
    path = tmp_path / "rag-report.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture()
def rag_baseline_report(tmp_path: Path) -> Path:
    data = {
        "eval_type": "rag",
        "summary": {
            "total_cases": 2,
            "metric_averages": {
                "faithfulness": 0.9,
                "answer_relevancy": 0.8,
            },
        },
        "results": [],
    }
    path = tmp_path / "rag-baseline.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture()
def bench_report(tmp_path: Path) -> Path:
    data = {
        "eval_type": "bench",
        "summary": {
            "flux-pro": {
                "overall_score": 4.3,
                "dimension_averages": {"visual_quality": 4.4},
            },
        },
    }
    path = tmp_path / "bench-report.json"
    path.write_text(json.dumps(data))
    return path


def test_gate_metric_threshold_pass(runner: CliRunner, rag_report: Path) -> None:
    result = runner.invoke(
        cli,
        ["gate", "--report", str(rag_report), "--metric-threshold", "faithfulness:0.8"],
    )
    assert result.exit_code == 0
    assert "PASS" in result.output


def test_gate_metric_threshold_fail(runner: CliRunner, rag_report: Path) -> None:
    result = runner.invoke(
        cli,
        ["gate", "--report", str(rag_report), "--metric-threshold", "answer_relevancy:0.9"],
    )
    assert result.exit_code == 1
    assert "FAIL" in result.output


def test_gate_threshold_invalid_for_metric_reports(runner: CliRunner, rag_report: Path) -> None:
    result = runner.invoke(
        cli,
        ["gate", "--report", str(rag_report), "--threshold", "0.8"],
    )
    assert result.exit_code == 2
    assert "--threshold" in result.output


def test_gate_baseline_type_mismatch_errors(
    runner: CliRunner,
    rag_report: Path,
    bench_report: Path,
) -> None:
    result = runner.invoke(
        cli,
        ["gate", "--report", str(rag_report), "--baseline", str(bench_report)],
    )
    assert result.exit_code == 2
    assert "type mismatch" in result.output


def test_compare_metric_reports(runner: CliRunner, rag_report: Path, rag_baseline_report: Path) -> None:
    result = runner.invoke(
        cli,
        ["compare", "--baseline", str(rag_baseline_report), "--candidate", str(rag_report)],
    )
    assert result.exit_code == 0
    assert "faithfulness" in result.output
    assert "answer_relevancy" in result.output


def test_compare_metric_reports_json_stdout(
    runner: CliRunner,
    rag_report: Path,
    rag_baseline_report: Path,
) -> None:
    result = runner.invoke(
        cli,
        [
            "compare",
            "--baseline",
            str(rag_baseline_report),
            "--candidate",
            str(rag_report),
            "--json-output",
            "-",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["eval_type"] == "rag"
    assert {row["key"] for row in data["rows"]} == {"faithfulness", "answer_relevancy"}


def test_compare_rejects_cross_type_reports(runner: CliRunner, rag_report: Path, bench_report: Path) -> None:
    result = runner.invoke(
        cli,
        ["compare", "--baseline", str(bench_report), "--candidate", str(rag_report)],
    )
    assert result.exit_code == 2
    assert "Cannot compare different report types" in result.output


# ------------------------------------------------------------------
# Additional edge cases (C9)
# ------------------------------------------------------------------


@pytest.fixture()
def agent_report(tmp_path: Path) -> Path:
    data = {
        "eval_type": "agent",
        "summary": {
            "total_cases": 1,
            "metric_averages": {
                "tool_call_accuracy": 0.95,
                "goal_accuracy": 0.78,
                "step_efficiency": 0.85,
            },
        },
    }
    path = tmp_path / "agent-report.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture()
def agent_baseline(tmp_path: Path) -> Path:
    data = {
        "eval_type": "agent",
        "summary": {
            "total_cases": 1,
            "metric_averages": {
                "tool_call_accuracy": 0.99,
                "goal_accuracy": 0.90,
                "step_efficiency": 0.85,
            },
        },
    }
    path = tmp_path / "agent-baseline.json"
    path.write_text(json.dumps(data))
    return path


def test_gate_agent_metric_threshold_pass(runner: CliRunner, agent_report: Path) -> None:
    result = runner.invoke(
        cli,
        ["gate", "--report", str(agent_report), "--metric-threshold", "goal_accuracy:0.7"],
    )
    assert result.exit_code == 0
    assert "PASS" in result.output


def test_gate_agent_metric_threshold_fail(runner: CliRunner, agent_report: Path) -> None:
    result = runner.invoke(
        cli,
        ["gate", "--report", str(agent_report), "--metric-threshold", "goal_accuracy:0.95"],
    )
    assert result.exit_code == 1
    assert "FAIL" in result.output


def test_gate_regression_detected_for_metric_report(
    runner: CliRunner,
    agent_report: Path,
    agent_baseline: Path,
) -> None:
    # goal_accuracy drop = 0.90 - 0.78 = 0.12 > default 0.3? no, within. set tight threshold
    result = runner.invoke(
        cli,
        [
            "gate",
            "--report", str(agent_report),
            "--baseline", str(agent_baseline),
            "--regression-threshold", "0.05",
        ],
    )
    assert result.exit_code == 1
    assert "FAIL" in result.output


def test_gate_json_output_schema(runner: CliRunner, rag_report: Path, tmp_path: Path) -> None:
    json_out = tmp_path / "gate-result.json"
    result = runner.invoke(
        cli,
        [
            "gate",
            "--report", str(rag_report),
            "--metric-threshold", "faithfulness:0.5",
            "--json-output", str(json_out),
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(json_out.read_text())
    assert payload["status"] == "pass"
    assert payload["eval_type"] == "rag"
    assert any(check["type"] == "metric_threshold" for check in payload["checks"])


def test_gate_bench_threshold_still_works(runner: CliRunner, bench_report: Path) -> None:
    """Regression guard: existing visual bench gate behavior unchanged."""
    result = runner.invoke(
        cli,
        ["gate", "--report", str(bench_report), "--threshold", "0.0"],
    )
    assert result.exit_code == 0
    assert "PASS" in result.output


def test_compare_delta_sign_direction(
    runner: CliRunner,
    rag_report: Path,
    rag_baseline_report: Path,
    tmp_path: Path,
) -> None:
    """candidate has lower scores than baseline => deltas should be negative."""
    result = runner.invoke(
        cli,
        [
            "compare",
            "--baseline", str(rag_baseline_report),
            "--candidate", str(rag_report),
            "--json-output", "-",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    deltas = {row["key"]: row["delta"] for row in data["rows"]}
    assert deltas["faithfulness"] < 0
    assert deltas["answer_relevancy"] < 0
