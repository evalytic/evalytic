"""Unit tests for the hallucination metric."""

from __future__ import annotations

import pytest

from evalytic.exceptions import ValidationError
from evalytic.text.metrics.hallucination import HallucinationMetric
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


def _rag_case() -> RAGTestCase:
    return RAGTestCase(
        query="Where is Evalytic used?",
        response="Evalytic is used in CI, production monitoring, and on the moon.",
        contexts=[
            RetrievedChunk(text="Evalytic is used in CI workflows."),
            RetrievedChunk(text="Teams use Evalytic in production; Evalytic is never deployed on the moon."),
        ],
    )


def test_hallucination_single_contradiction() -> None:
    metric = HallucinationMetric()
    case = _rag_case()
    judge = QueueJudge(
        [
            {
                "claims": [
                    {"text": "Used in CI", "contradicted": False, "evidence": "chunk 1"},
                    {"text": "Used in production monitoring", "contradicted": False, "evidence": "chunk 2"},
                    {"text": "Deployed on the moon", "contradicted": True, "evidence": "chunk 2"},
                    {"text": "Supports many formats", "contradicted": False, "evidence": ""},
                ],
                "reason": "One claim contradicts the context.",
            }
        ],
        last_cost=0.15,
    )

    result = metric.score(case, judge=judge)

    assert result.metric_id == "hallucination"
    assert result.score == pytest.approx(0.75, rel=1e-6)
    assert result.cost == pytest.approx(0.15, rel=1e-6)
    assert result.details["contradicted_count"] == 1
    assert result.details["total_count"] == 4


def test_hallucination_no_claims_returns_one() -> None:
    metric = HallucinationMetric()
    case = _rag_case()
    judge = QueueJudge([{"claims": [], "reason": "No claims to evaluate."}])

    result = metric.score(case, judge=judge)

    assert result.score == pytest.approx(1.0)
    assert result.details["total_count"] == 0
    assert result.details["contradicted_count"] == 0


def test_hallucination_all_contradicted() -> None:
    metric = HallucinationMetric()
    case = _rag_case()
    judge = QueueJudge(
        [
            {
                "claims": [
                    {"text": "A", "contradicted": True},
                    {"text": "B", "contradicted": True},
                ],
            }
        ]
    )

    result = metric.score(case, judge=judge)

    assert result.score == pytest.approx(0.0)
    assert result.details["contradicted_count"] == 2


def test_hallucination_requires_judge() -> None:
    metric = HallucinationMetric()
    case = _rag_case()

    with pytest.raises(ValidationError):
        metric.score(case, judge=None)


def test_hallucination_requires_contexts() -> None:
    metric = HallucinationMetric()
    case = RAGTestCase(query="q", response="a", contexts=[])
    judge = QueueJudge([])

    with pytest.raises(ValidationError):
        metric.score(case, judge=judge)


def test_hallucination_differs_from_faithfulness_on_unsupported_but_not_contradicted() -> None:
    """A claim that context doesn't mention should NOT be counted as hallucination."""
    metric = HallucinationMetric()
    case = _rag_case()
    judge = QueueJudge(
        [
            {
                "claims": [
                    {"text": "Used in CI", "contradicted": False},
                    {"text": "Mentioned by the CEO", "contradicted": False},
                ],
            }
        ]
    )

    result = metric.score(case, judge=judge)

    assert result.score == pytest.approx(1.0)
    assert result.details["contradicted_count"] == 0
