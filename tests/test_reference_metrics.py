"""Reference-based RAG + text metric unit tests (claim-level LLM judging)."""

from __future__ import annotations

import pytest

from evalytic.exceptions import ValidationError
from evalytic.text.metrics.context_precision import ContextPrecisionMetric
from evalytic.text.metrics.context_recall import ContextRecallMetric
from evalytic.text.metrics.factual_correctness import FactualCorrectnessMetric
from evalytic.text.types import RAGTestCase, RetrievedChunk, TextTestCase


class QueueJudge:
    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.judge_string = "fake-judge"

    def complete_json(self, prompt: str, *_args, **_kwargs) -> dict:
        assert prompt
        if not self._responses:
            raise AssertionError("Queue empty")
        return self._responses.pop(0)


class TestContextPrecision:
    def test_rank_weighted_precision_formula(self) -> None:
        case = RAGTestCase(
            query="Q?",
            response="A",
            reference="Reference",
            contexts=[RetrievedChunk(text=f"c{i}", rank=i) for i in range(1, 4)],
        )
        judge = QueueJudge([{
            "contexts": [
                {"rank": 1, "relevant": True},
                {"rank": 2, "relevant": False},
                {"rank": 3, "relevant": True},
            ],
            "reason": "rank 1 and 3 relevant",
        }])
        result = ContextPrecisionMetric().score(case, judge=judge)
        # precision@1 = 1/1, precision@3 = 2/3, sum = 1 + 2/3 = 5/3, mean over 2 relevant = 5/6
        assert result.score == pytest.approx(5 / 6, rel=1e-6)
        assert result.details["relevant_count"] == 2

    def test_empty_judgments_returns_0(self) -> None:
        case = RAGTestCase(query="Q", response="A", reference="R", contexts=[RetrievedChunk(text="c1")])
        judge = QueueJudge([{"contexts": [], "reason": "no data"}])
        result = ContextPrecisionMetric().score(case, judge=judge)
        assert result.score == 0.0

    def test_requires_reference(self) -> None:
        case = RAGTestCase(query="Q", response="A", contexts=[RetrievedChunk(text="c")])
        with pytest.raises(ValidationError):
            ContextPrecisionMetric().score(case, judge=QueueJudge([]))

    def test_requires_judge(self) -> None:
        case = RAGTestCase(query="Q", response="A", reference="R", contexts=[RetrievedChunk(text="c")])
        with pytest.raises(ValidationError):
            ContextPrecisionMetric().score(case, judge=None)


class TestContextRecall:
    def test_all_claims_supported_returns_1(self) -> None:
        case = RAGTestCase(
            query="Q?",
            response="A",
            reference="fact1; fact2",
            contexts=[RetrievedChunk(text="fact1 and fact2 both shown")],
        )
        judge = QueueJudge([{
            "claims": [
                {"text": "fact1", "supported": True},
                {"text": "fact2", "supported": True},
            ],
            "reason": "both grounded",
        }])
        result = ContextRecallMetric().score(case, judge=judge)
        assert result.score == 1.0
        assert result.details["supported_count"] == 2

    def test_partial_coverage_ratio(self) -> None:
        case = RAGTestCase(query="Q", response="A", reference="R", contexts=[RetrievedChunk(text="c")])
        judge = QueueJudge([{
            "claims": [
                {"text": "a", "supported": True},
                {"text": "b", "supported": False},
                {"text": "c", "supported": False},
            ],
        }])
        result = ContextRecallMetric().score(case, judge=judge)
        assert result.score == pytest.approx(1 / 3, rel=1e-6)

    def test_no_claims_returns_0(self) -> None:
        case = RAGTestCase(query="Q", response="A", reference="R", contexts=[RetrievedChunk(text="c")])
        judge = QueueJudge([{"claims": [], "reason": "empty"}])
        result = ContextRecallMetric().score(case, judge=judge)
        assert result.score == 0.0


class TestFactualCorrectness:
    def test_f1_computation(self) -> None:
        case = TextTestCase(input="in", output="out", expected="ref")
        judge = QueueJudge([{
            "true_positives": ["x", "y"],
            "false_positives": ["z"],
            "false_negatives": [],
            "reason": "ok",
        }])
        result = FactualCorrectnessMetric().score(case, judge=judge)
        # precision = 2/3, recall = 2/2, f1 = 2 * (2/3) * 1 / (2/3 + 1) = 0.8
        assert result.score == pytest.approx(0.8, abs=1e-3)
        assert result.details["precision"] == pytest.approx(2 / 3, abs=1e-3)

    def test_all_false_positives_returns_0(self) -> None:
        case = TextTestCase(input="in", output="out", expected="ref")
        judge = QueueJudge([{
            "true_positives": [],
            "false_positives": ["bad1", "bad2"],
            "false_negatives": ["missed"],
        }])
        result = FactualCorrectnessMetric().score(case, judge=judge)
        assert result.score == 0.0

    def test_requires_expected(self) -> None:
        case = TextTestCase(input="in", output="out")
        with pytest.raises(ValidationError):
            FactualCorrectnessMetric().score(case, judge=QueueJudge([]))
