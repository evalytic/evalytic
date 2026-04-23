"""Tests for the evalytic.testing pytest integration."""

from __future__ import annotations

import pytest

from evalytic import testing as evalytic_testing
from evalytic.agent.types import AgentTestCase, AgentToolCall
from evalytic.text.types import (
    MetricEvalReport,
    MetricEvalResult,
    MetricResult,
    RAGTestCase,
    RetrievedChunk,
    TextTestCase,
)


def _report_with(eval_type: str, metrics: dict[str, float]) -> MetricEvalReport:
    metric_results = [
        MetricResult(metric_id=metric_id, score=score)
        for metric_id, score in metrics.items()
    ]
    return MetricEvalReport(
        eval_type=eval_type,
        judge="fake-judge",
        results=[
            MetricEvalResult(
                case_id="case-0",
                test_case=None,
                metrics=metric_results,
                total_cost=0.0,
                duration_ms=0,
            )
        ],
    )


@pytest.fixture
def rag_case() -> RAGTestCase:
    return RAGTestCase(
        query="What is Evalytic?",
        response="Evalytic is an eval SDK.",
        contexts=[RetrievedChunk(text="Evalytic evaluates AI outputs.")],
    )


@pytest.fixture
def text_case() -> TextTestCase:
    return TextTestCase(input="hi", output="Hello!", expected="Hello!")


@pytest.fixture
def agent_case() -> AgentTestCase:
    return AgentTestCase(
        input="Book a flight.",
        final_output="Booked.",
        tool_calls=[AgentToolCall(name="search_flights")],
    )


def test_assert_test_passes_when_all_metrics_meet_threshold(monkeypatch, rag_case):
    def fake_rag(cases, *, metric_ids=None, **kwargs):
        assert metric_ids == ["faithfulness", "hallucination"]
        return _report_with("rag", {"faithfulness": 0.9, "hallucination": 0.95})

    monkeypatch.setattr(evalytic_testing, "evaluate_rag", fake_rag)

    report = evalytic_testing.assert_test(
        rag_case, {"faithfulness": 0.8, "hallucination": 0.9}
    )
    assert report.metric_averages() == {"faithfulness": 0.9, "hallucination": 0.95}


def test_assert_test_reports_all_failures_together(monkeypatch, rag_case):
    def fake_rag(cases, *, metric_ids=None, **kwargs):
        return _report_with("rag", {"faithfulness": 0.5, "hallucination": 0.7})

    monkeypatch.setattr(evalytic_testing, "evaluate_rag", fake_rag)

    with pytest.raises(AssertionError) as excinfo:
        evalytic_testing.assert_test(
            rag_case, {"faithfulness": 0.8, "hallucination": 0.9}
        )

    message = str(excinfo.value)
    assert "faithfulness: 0.5000 < threshold 0.8000" in message
    assert "hallucination: 0.7000 < threshold 0.9000" in message


def test_assert_test_flags_missing_metric(monkeypatch, rag_case):
    def fake_rag(cases, *, metric_ids=None, **kwargs):
        return _report_with("rag", {"faithfulness": 0.95})

    monkeypatch.setattr(evalytic_testing, "evaluate_rag", fake_rag)

    with pytest.raises(AssertionError) as excinfo:
        evalytic_testing.assert_test(
            rag_case, {"faithfulness": 0.8, "hallucination": 0.9}
        )

    assert "hallucination: not produced by runner" in str(excinfo.value)


def test_assert_metric_single_metric(monkeypatch, rag_case):
    captured: dict = {}

    def fake_rag(cases, *, metric_ids=None, **kwargs):
        captured["metric_ids"] = metric_ids
        return _report_with("rag", {"hallucination": 0.99})

    monkeypatch.setattr(evalytic_testing, "evaluate_rag", fake_rag)

    evalytic_testing.assert_metric(rag_case, "hallucination", threshold=0.9)
    assert captured["metric_ids"] == ["hallucination"]


def test_assert_test_dispatches_to_text_runner(monkeypatch, text_case):
    def fake_text(cases, *, metric_ids=None, **kwargs):
        assert metric_ids == ["factual_correctness"]
        return _report_with("text", {"factual_correctness": 0.9})

    def fail_rag(*args, **kwargs):
        raise AssertionError("should not dispatch to evaluate_rag for TextTestCase")

    monkeypatch.setattr(evalytic_testing, "evaluate_text", fake_text)
    monkeypatch.setattr(evalytic_testing, "evaluate_rag", fail_rag)

    evalytic_testing.assert_test(text_case, {"factual_correctness": 0.8})


def test_assert_test_dispatches_to_agent_runner(monkeypatch, agent_case):
    captured: dict = {}

    def fake_agent(cases, **kwargs):
        captured["kwargs"] = kwargs
        return _report_with(
            "agent",
            {"tool_call_accuracy": 1.0, "goal_accuracy": 0.9, "step_efficiency": 1.0},
        )

    monkeypatch.setattr(evalytic_testing, "evaluate_agent", fake_agent)

    evalytic_testing.assert_test(agent_case, {"goal_accuracy": 0.8})
    # agent runner doesn't accept metric_ids; ensure we didn't pass it
    assert "metric_ids" not in captured["kwargs"]


def test_assert_test_rejects_unsupported_case_type():
    with pytest.raises(TypeError):
        evalytic_testing.assert_test(object(), {"faithfulness": 0.8})


def test_assert_test_requires_metrics(rag_case):
    with pytest.raises(ValueError):
        evalytic_testing.assert_test(rag_case, {})


def test_assert_test_passes_judge_override(monkeypatch, rag_case):
    captured: dict = {}

    def fake_rag(cases, *, metric_ids=None, **kwargs):
        captured.update(kwargs)
        return _report_with("rag", {"faithfulness": 0.95})

    monkeypatch.setattr(evalytic_testing, "evaluate_rag", fake_rag)

    evalytic_testing.assert_test(
        rag_case,
        {"faithfulness": 0.8},
        judge="gpt-5.2",
        judges=["gemini-2.5-flash", "gpt-5.2"],
        base_url="http://localhost:11434",
    )

    assert captured["judge"] == "gpt-5.2"
    assert captured["judges"] == ["gemini-2.5-flash", "gpt-5.2"]
    assert captured["base_url"] == "http://localhost:11434"
