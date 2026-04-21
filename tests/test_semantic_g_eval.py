"""Semantic similarity + G-Eval metric tests."""

from __future__ import annotations

import pytest

from evalytic.exceptions import ValidationError
from evalytic.text.metrics.g_eval import GEvalMetric
from evalytic.text.metrics.semantic_similarity import SemanticSimilarityMetric
from evalytic.text.types import TextTestCase


class FakeEmbedder:
    def __init__(self, mapping: dict[str, list[float]], *, last_cost: float = 0.0) -> None:
        self.mapping = mapping
        self.last_cost = last_cost

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.mapping[text] for text in texts]


class QueueJudge:
    def __init__(self, responses: list[dict], *, last_cost: float = 0.0) -> None:
        self._responses = list(responses)
        self.judge_string = "fake-judge"
        self.last_cost = last_cost

    def complete_json(self, prompt: str, *_args, **_kwargs) -> dict:
        assert prompt
        if not self._responses:
            raise AssertionError("Queue empty")
        return self._responses.pop(0)


class TestSemanticSimilarity:
    def test_identical_vectors_return_1(self) -> None:
        case = TextTestCase(input="", output="hello", expected="hello")
        embedder = FakeEmbedder({"hello": [1.0, 0.0, 0.0]}, last_cost=0.07)
        result = SemanticSimilarityMetric().score(case, embedder=embedder)
        assert result.score == pytest.approx(1.0, rel=1e-6)
        assert result.cost == pytest.approx(0.07, rel=1e-6)

    def test_orthogonal_vectors_return_0(self) -> None:
        case = TextTestCase(input="", output="a", expected="b")
        embedder = FakeEmbedder({"a": [1.0, 0.0], "b": [0.0, 1.0]})
        result = SemanticSimilarityMetric().score(case, embedder=embedder)
        assert result.score == 0.0

    def test_missing_expected_raises(self) -> None:
        case = TextTestCase(input="", output="hi")
        with pytest.raises(ValidationError):
            SemanticSimilarityMetric().score(case, embedder=FakeEmbedder({}))

    def test_missing_embedder_raises(self) -> None:
        case = TextTestCase(input="", output="a", expected="b")
        with pytest.raises(ValidationError):
            SemanticSimilarityMetric().score(case, embedder=None)


class TestGEval:
    def test_rubric_score_normalized_to_0_1(self) -> None:
        case = TextTestCase(
            input="Write a haiku",
            output="A lonely cat / watches the moon / falls asleep",
            criteria="Evaluate poetic quality on 1-5 scale.",
        )
        judge = QueueJudge([{"score": 4, "max_score": 5, "reason": "Good", "rubric": []}], last_cost=0.11)
        result = GEvalMetric().score(case, judge=judge)
        assert result.score == pytest.approx(0.8, rel=1e-6)
        assert result.cost == pytest.approx(0.11, rel=1e-6)
        assert result.details["raw_score"] == 4.0
        assert result.details["max_score"] == 5

    def test_missing_criteria_raises(self) -> None:
        case = TextTestCase(input="x", output="y")
        with pytest.raises(ValidationError):
            GEvalMetric().score(case, judge=QueueJudge([]))

    def test_score_clamped_to_0_1(self) -> None:
        case = TextTestCase(input="x", output="y", criteria="c")
        judge = QueueJudge([{"score": 99, "max_score": 5, "reason": "bad judge"}])
        result = GEvalMetric().score(case, judge=judge)
        assert 0.0 <= result.score <= 1.0

    def test_max_score_zero_falls_back_to_scale_max(self) -> None:
        """`max_score=0` falsy -> `or self.scale_max` picks 5. No ZeroDivisionError."""
        case = TextTestCase(input="x", output="y", criteria="c")
        judge = QueueJudge([{"score": 5, "max_score": 0, "reason": "weird"}])
        result = GEvalMetric().score(case, judge=judge)
        # No crash; score remains well-defined (clamped to [0,1]).
        assert 0.0 <= result.score <= 1.0

    def test_metadata_criteria_fallback(self) -> None:
        case = TextTestCase(
            input="x",
            output="y",
            metadata={"criteria": "judge creativity"},
        )
        judge = QueueJudge([{"score": 3, "max_score": 5, "reason": "ok"}])
        result = GEvalMetric().score(case, judge=judge)
        assert result.score == pytest.approx(0.6, rel=1e-6)
