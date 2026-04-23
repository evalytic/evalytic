"""Unit tests for the contextual_relevancy metric."""

from __future__ import annotations

import pytest

from evalytic.exceptions import ValidationError
from evalytic.text.metrics.contextual_relevancy import ContextualRelevancyMetric
from evalytic.text.types import RAGTestCase, RetrievedChunk


class QueueJudge:
    def __init__(self, responses: list[dict], *, last_cost: float = 0.0) -> None:
        self._responses = list(responses)
        self.judge_string = "fake-judge"
        self.last_cost = last_cost

    def complete_json(self, prompt: str, *args, **kwargs) -> dict:
        assert prompt
        if not self._responses:
            raise AssertionError("No queued fake judge response available.")
        return self._responses.pop(0)


def _case_with(chunks: list[str]) -> RAGTestCase:
    return RAGTestCase(
        query="What is Evalytic?",
        response="Evalytic is an evaluation SDK for AI outputs.",
        contexts=[RetrievedChunk(text=text) for text in chunks],
    )


def test_contextual_relevancy_partial_relevance() -> None:
    metric = ContextualRelevancyMetric()
    case = _case_with(
        [
            "Evalytic is an evaluation SDK for AI outputs.",
            "The capital of France is Paris.",
            "Evalytic supports RAG and text eval.",
        ]
    )
    judge = QueueJudge(
        [
            {
                "chunks": [
                    {"index": 1, "relevant": True, "reason": "Direct definition."},
                    {"index": 2, "relevant": False, "reason": "Unrelated."},
                    {"index": 3, "relevant": True, "reason": "Describes capabilities."},
                ],
                "reason": "2 of 3 chunks are relevant.",
            }
        ],
        last_cost=0.08,
    )

    result = metric.score(case, judge=judge)

    assert result.metric_id == "contextual_relevancy"
    assert result.score == pytest.approx(2 / 3, rel=1e-6)
    assert result.cost == pytest.approx(0.08, rel=1e-6)
    assert result.details["relevant_count"] == 2
    assert result.details["total_count"] == 3
    assert result.judge == "fake-judge"


def test_contextual_relevancy_all_relevant() -> None:
    metric = ContextualRelevancyMetric()
    case = _case_with(["a", "b"])
    judge = QueueJudge(
        [
            {
                "chunks": [
                    {"index": 1, "relevant": True},
                    {"index": 2, "relevant": True},
                ],
            }
        ]
    )

    result = metric.score(case, judge=judge)

    assert result.score == pytest.approx(1.0)
    assert result.details["relevant_count"] == 2


def test_contextual_relevancy_none_relevant() -> None:
    metric = ContextualRelevancyMetric()
    case = _case_with(["x", "y"])
    judge = QueueJudge(
        [
            {
                "chunks": [
                    {"index": 1, "relevant": False},
                    {"index": 2, "relevant": False},
                ],
            }
        ]
    )

    result = metric.score(case, judge=judge)

    assert result.score == pytest.approx(0.0)
    assert result.details["relevant_count"] == 0


def test_contextual_relevancy_empty_chunks_raises() -> None:
    metric = ContextualRelevancyMetric()
    case = RAGTestCase(query="q", response="a", contexts=[])
    judge = QueueJudge([])

    with pytest.raises(ValidationError):
        metric.score(case, judge=judge)


def test_contextual_relevancy_requires_judge() -> None:
    metric = ContextualRelevancyMetric()
    case = _case_with(["a"])

    with pytest.raises(ValidationError):
        metric.score(case, judge=None)


def test_contextual_relevancy_empty_judgments_returns_zero() -> None:
    metric = ContextualRelevancyMetric()
    case = _case_with(["a", "b"])
    judge = QueueJudge([{"chunks": [], "reason": "Judge failed to classify chunks."}])

    result = metric.score(case, judge=judge)

    assert result.score == pytest.approx(0.0)
    assert result.details["total_count"] == 2
    assert result.details["relevant_count"] == 0
