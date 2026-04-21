"""Unit tests for text/RAG metric internals."""

from __future__ import annotations

from pathlib import Path

import pytest

from evalytic.judge.consensus import resolve_metric_consensus
from evalytic.text.metrics.answer_relevancy import AnswerRelevancyMetric
from evalytic.text.metrics.faithfulness import FaithfulnessMetric
from evalytic.text.runner import detect_dataset_type
from evalytic.text.types import MetricResult, RAGTestCase, RetrievedChunk


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


class FakeEmbedder:
    def __init__(self, mapping: dict[str, list[float]], *, last_cost: float = 0.0) -> None:
        self.mapping = mapping
        self.last_cost = last_cost

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.mapping[text] for text in texts]


def test_faithfulness_aggregates_claim_support() -> None:
    metric = FaithfulnessMetric()
    case = RAGTestCase(
        query="Where is Evalytic used?",
        response="Evalytic is used in CI, production monitoring, and on the moon.",
        contexts=[
            RetrievedChunk(text="Evalytic is used in CI workflows."),
            RetrievedChunk(text="Teams also use Evalytic for production monitoring."),
        ],
    )
    judge = QueueJudge(
        [
            {
                "claims": [
                    {"text": "Used in CI", "supported": True, "evidence": "chunk 1"},
                    {"text": "Used in production monitoring", "supported": True, "evidence": "chunk 2"},
                    {"text": "Used on the moon", "supported": False, "evidence": ""},
                ],
                "reason": "Two claims are grounded and one is hallucinated.",
            }
        ],
        last_cost=0.12,
    )

    result = metric.score(case, judge=judge)

    assert result.metric_id == "faithfulness"
    assert result.score == pytest.approx(2 / 3, rel=1e-6)
    assert result.cost == pytest.approx(0.12, rel=1e-6)
    assert result.details == {
        "claims": [
            {"text": "Used in CI", "supported": True, "evidence": "chunk 1"},
            {"text": "Used in production monitoring", "supported": True, "evidence": "chunk 2"},
            {"text": "Used on the moon", "supported": False, "evidence": ""},
        ],
        "supported_count": 2,
        "total_count": 3,
    }


def test_answer_relevancy_uses_reverse_questions_and_cosine_average() -> None:
    metric = AnswerRelevancyMetric()
    case = RAGTestCase(
        query="What is Evalytic?",
        response="Evalytic is an evaluation platform for AI outputs.",
        contexts=[RetrievedChunk(text="Evalytic evaluates AI outputs.")],
    )
    judge = QueueJudge(
        [
            {
                "generated_questions": [
                    "What is Evalytic?",
                    "Explain Evalytic",
                    "Describe Evalytic",
                ],
                "reason": "Reverse questions stay close to the original query.",
            }
        ],
        last_cost=0.2,
    )
    embedder = FakeEmbedder(
        {
            "What is Evalytic?": [1.0, 0.0],
            "Explain Evalytic": [0.9, 0.1],
            "Describe Evalytic": [0.8, 0.2],
        },
        last_cost=0.3,
    )

    result = metric.score(case, judge=judge, embedder=embedder)

    assert result.metric_id == "answer_relevancy"
    assert result.score == pytest.approx((1.0 + 0.9939 + 0.9701) / 3, rel=1e-3)
    assert result.cost == pytest.approx(0.5, rel=1e-6)
    assert result.details is not None
    assert result.details["generated_questions"] == [
        "What is Evalytic?",
        "Explain Evalytic",
        "Describe Evalytic",
    ]


def test_detect_dataset_type_prefers_new_type_and_supports_agent() -> None:
    assert detect_dataset_type({"type": "rag", "items": [{"prompt": "ignored"}]}) == "rag"
    assert detect_dataset_type({"pipeline": "img2img", "items": []}) == "img2img"
    assert detect_dataset_type({"items": [{"query": "Q", "response": "A"}]}) == "rag"
    assert detect_dataset_type({"items": [{"input": "task", "output": "result"}]}) == "text"
    assert detect_dataset_type({"items": [{"input": "task", "final_output": "done"}]}) == "agent"


def test_metric_consensus_uses_tiebreaker_after_primary_failure() -> None:
    calls: list[str] = []

    def scorer(judge_name: str) -> MetricResult:
        calls.append(judge_name)
        if judge_name == "judge-a":
            raise RuntimeError("boom")
        if judge_name == "judge-b":
            return MetricResult(metric_id="faithfulness", score=0.8, judge="judge-b")
        return MetricResult(metric_id="faithfulness", score=0.82, judge="judge-c")

    def merge(values: list[MetricResult], judge_names: list[str], agreement: str) -> MetricResult:
        return MetricResult(
            metric_id="faithfulness",
            score=sum(value.score for value in values) / len(values),
            agreement=agreement,
            judge_scores={name: value.score for name, value in zip(judge_names, values)},
        )

    result = resolve_metric_consensus(
        ["judge-a", "judge-b", "judge-c"],
        scorer=scorer,
        score_of=lambda item: item.score,
        merge=merge,
    )

    assert calls == ["judge-a", "judge-b", "judge-c"]
    assert result.score == pytest.approx(0.81, rel=1e-6)
    assert result.agreement == "high"
