"""Runner integration tests: evaluate_rag / evaluate_text / evaluate_agent.

All external dependencies (TextJudge, resolve_embedder) are patched so these
tests run offline and quickly.
"""

from __future__ import annotations

from typing import Any

import pytest

from evalytic.agent.runner import evaluate_agent
from evalytic.agent.types import AgentTestCase, AgentToolCall
from evalytic.text import runner as text_runner
from evalytic.agent import runner as agent_runner
from evalytic.text.runner import evaluate_rag, evaluate_text
from evalytic.text.types import RAGTestCase, RetrievedChunk, TextTestCase


class DummyJudge:
    """Returns deterministic LLM responses keyed by prompt-contains."""

    def __init__(self, judge_name: str = "dummy") -> None:
        self.judge_string = judge_name
        self.last_cost = 0.3

    def complete_json(self, prompt: str, *_a, **_kw) -> dict:
        if "faithfulness" in prompt or "faithful" in prompt:
            return {
                "claims": [
                    {"text": "c1", "supported": True, "evidence": "ctx1"},
                    {"text": "c2", "supported": False, "evidence": ""},
                ],
                "reason": "one claim supported",
            }
        if "reverse-engineered" in prompt or "reverse" in prompt:
            return {
                "generated_questions": ["What is X?", "Explain X", "Describe X"],
                "reason": "reverse questions",
            }
        if "precision" in prompt.lower():
            return {
                "contexts": [{"rank": 1, "relevant": True}],
                "reason": "single relevant",
            }
        if "goal" in prompt.lower():
            return {"score": 0.85, "reason": "goal achieved"}
        # Default: factual_correctness
        return {
            "true_positives": ["a"],
            "false_positives": [],
            "false_negatives": [],
            "reason": "match",
        }

    def close(self) -> None:
        return None


class ConstEmbedder:
    """Returns a canonical vector for any input (cosine = 1 among calls)."""

    last_cost = 0.1

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def close(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _patch_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(text_runner, "TextJudge", lambda judge, base_url=None: DummyJudge(judge))
    monkeypatch.setattr(text_runner, "resolve_embedder", lambda cfg=None: ConstEmbedder())
    monkeypatch.setattr(agent_runner, "TextJudge", lambda judge, base_url=None: DummyJudge(judge))
    monkeypatch.setattr(agent_runner, "resolve_embedder", lambda cfg=None: ConstEmbedder())


class TestEvaluateRAG:
    def test_single_judge_emits_metrics_and_eval_type(self) -> None:
        case = RAGTestCase(
            query="What is X?",
            response="X is something",
            contexts=[RetrievedChunk(text="X is defined here")],
        )
        report = evaluate_rag([case])
        assert report.eval_type == "rag"
        metric_ids = {m.metric_id for m in report.results[0].metrics}
        assert "faithfulness" in metric_ids
        assert "answer_relevancy" in metric_ids

    def test_consensus_mode_labels_judge_name(self) -> None:
        case = RAGTestCase(
            query="What is X?",
            response="X is something",
            contexts=[RetrievedChunk(text="X")],
        )
        report = evaluate_rag([case], judges=["gemini-2.5-flash", "gpt-5.2"])
        assert report.consensus_mode is True
        assert "consensus" in report.judge

    def test_rejects_unknown_metric(self) -> None:
        case = RAGTestCase(query="x", response="y", contexts=[RetrievedChunk(text="z")])
        from evalytic.exceptions import ValidationError

        with pytest.raises(ValidationError):
            evaluate_rag([case], metric_ids=["not_a_real_metric"])

    def test_multiple_cases_aggregate_averages(self) -> None:
        cases = [
            RAGTestCase(
                query=f"q{i}",
                response=f"r{i}",
                contexts=[RetrievedChunk(text=f"c{i}")],
            )
            for i in range(3)
        ]
        report = evaluate_rag(cases)
        averages = report.metric_averages()
        assert "faithfulness" in averages
        assert 0.0 <= averages["faithfulness"] <= 1.0


class TestEvaluateText:
    def test_default_metrics_run(self) -> None:
        case = TextTestCase(
            input="summarize X",
            output="X is something",
            expected="X is something",
        )
        report = evaluate_text([case])
        assert report.eval_type == "text"
        metric_ids = {m.metric_id for m in report.results[0].metrics}
        assert metric_ids == {"factual_correctness", "semantic_similarity"}
        assert report.results[0].total_cost == pytest.approx(0.4, rel=1e-6)
        assert report.to_dict()["summary"]["total_cost"] == pytest.approx(0.4, rel=1e-6)

    def test_statistical_metrics_no_judge_or_embedder(self) -> None:
        case = TextTestCase(input="i", output="hello world", expected="hello world")
        report = evaluate_text(
            [case],
            metric_ids=["exact_match", "levenshtein", "string_presence"],
        )
        scores = {m.metric_id: m.score for m in report.results[0].metrics}
        assert scores["exact_match"] == 1.0
        assert scores["levenshtein"] == pytest.approx(1.0, rel=1e-6)
        assert scores["string_presence"] == 1.0


class TestEvaluateAgent:
    def test_runs_three_agent_metrics(self) -> None:
        case = AgentTestCase(
            input="find answer",
            final_output="42",
            expected_output="42",
            tool_calls=[AgentToolCall(name="search")],
            metadata={"expected_tool_calls": ["search"], "expected_max_steps": 3},
        )
        report = evaluate_agent([case])
        ids = {m.metric_id for m in report.results[0].metrics}
        assert ids == {"tool_call_accuracy", "goal_accuracy", "step_efficiency"}
        assert report.eval_type == "agent"
