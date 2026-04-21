"""Agent metric unit tests (tool_call_accuracy / goal_accuracy / step_efficiency)."""

from __future__ import annotations

import pytest

from evalytic.agent.runner import (
    _goal_accuracy,
    _step_efficiency,
    _tool_call_accuracy,
    evaluate_agent,
)
from evalytic.agent.types import AgentTestCase, AgentToolCall


class FakeEmbedder:
    def __init__(self, mapping: dict[str, list[float]], *, last_cost: float = 0.0) -> None:
        self.mapping = mapping
        self.last_cost = last_cost

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.mapping[text] for text in texts]

    def close(self) -> None:
        return None


class QueueJudge:
    def __init__(self, responses: list[dict], *, last_cost: float = 0.0) -> None:
        self._responses = list(responses)
        self.judge_string = "fake-agent-judge"
        self.last_cost = last_cost

    def complete_json(self, prompt: str, *_args, **_kwargs) -> dict:
        assert prompt
        if not self._responses:
            raise AssertionError("Queue empty")
        return self._responses.pop(0)

    def close(self) -> None:
        return None


class TestToolCallAccuracy:
    def test_exact_match_returns_1(self) -> None:
        case = AgentTestCase(
            input="task",
            final_output="done",
            tool_calls=[AgentToolCall(name="search"), AgentToolCall(name="summarize")],
            metadata={"expected_tool_calls": ["search", "summarize"]},
        )
        result = _tool_call_accuracy(case)
        assert result.score == 1.0
        assert result.details["expected_tool_calls"] == ["search", "summarize"]

    def test_partial_overlap_f1(self) -> None:
        case = AgentTestCase(
            input="task",
            final_output="done",
            tool_calls=[AgentToolCall(name="search")],
            metadata={"expected_tool_calls": ["search", "summarize"]},
        )
        # precision = 1/1, recall = 1/2, f1 = 2 * 1 * 0.5 / 1.5 = 2/3
        result = _tool_call_accuracy(case)
        assert result.score == pytest.approx(2 / 3, rel=1e-6)

    def test_empty_both_sides_returns_1(self) -> None:
        case = AgentTestCase(input="task", final_output="done", tool_calls=[])
        result = _tool_call_accuracy(case)
        # No expected + no actual -> heuristic path: score = 1.0
        assert result.score == 1.0

    def test_no_expected_many_actuals_gets_penalized(self) -> None:
        case = AgentTestCase(
            input="task",
            final_output="done",
            tool_calls=[AgentToolCall(name="a"), AgentToolCall(name="b"), AgentToolCall(name="c")],
        )
        result = _tool_call_accuracy(case)
        # Heuristic: 1 - (3-1)*0.1 = 0.8
        assert result.score == pytest.approx(0.8, abs=1e-6)


class TestGoalAccuracy:
    def test_with_embedder_uses_cosine(self) -> None:
        case = AgentTestCase(
            input="get answer",
            final_output="42",
            expected_output="42",
        )
        embedder = FakeEmbedder({"42": [1.0, 0.0]}, last_cost=0.09)
        result = _goal_accuracy(case, QueueJudge([]), embedder)
        assert result.score == pytest.approx(1.0, rel=1e-6)
        assert result.cost == pytest.approx(0.09, rel=1e-6)

    def test_without_embedder_but_expected_falls_back_to_llm(self) -> None:
        """A3 regression guard: embedder=None should LLM-fallback, not raise."""
        case = AgentTestCase(
            input="get answer",
            final_output="42",
            expected_output="42",
        )
        judge = QueueJudge([{"score": 0.9, "reason": "expected matches"}], last_cost=0.14)
        result = _goal_accuracy(case, judge, embedder=None)
        assert result.score == pytest.approx(0.9, rel=1e-6)
        assert result.reason == "expected matches"
        assert result.cost == pytest.approx(0.14, rel=1e-6)

    def test_without_expected_uses_judge(self) -> None:
        case = AgentTestCase(input="get answer", final_output="42")
        judge = QueueJudge([{"score": 0.7, "reason": "decent"}])
        result = _goal_accuracy(case, judge, embedder=None)
        assert result.score == pytest.approx(0.7, rel=1e-6)

    def test_llm_score_clamped_to_0_1(self) -> None:
        case = AgentTestCase(input="go", final_output="out")
        judge = QueueJudge([{"score": 99, "reason": "weird"}])
        result = _goal_accuracy(case, judge, embedder=None)
        assert 0.0 <= result.score <= 1.0


class TestStepEfficiency:
    def test_zero_tool_calls_returns_1(self) -> None:
        case = AgentTestCase(input="task", final_output="done", tool_calls=[])
        assert _step_efficiency(case).score == 1.0

    def test_under_max_steps_returns_1(self) -> None:
        case = AgentTestCase(
            input="task",
            final_output="done",
            tool_calls=[AgentToolCall(name="a"), AgentToolCall(name="b")],
            metadata={"expected_max_steps": 5},
        )
        assert _step_efficiency(case).score == 1.0

    def test_above_max_steps_penalizes(self) -> None:
        case = AgentTestCase(
            input="task",
            final_output="done",
            tool_calls=[AgentToolCall(name=str(i)) for i in range(10)],
            metadata={"expected_max_steps": 5},
        )
        # 5/10 = 0.5
        assert _step_efficiency(case).score == pytest.approx(0.5, rel=1e-6)

    def test_no_max_heuristic_penalty(self) -> None:
        case = AgentTestCase(
            input="task",
            final_output="done",
            tool_calls=[AgentToolCall(name=str(i)) for i in range(5)],
        )
        # 1 - (5-3)*0.15 = 0.7
        assert _step_efficiency(case).score == pytest.approx(0.7, rel=1e-6)


class TestEvaluateAgentIntegration:
    def test_runs_three_metrics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """evaluate_agent emits all three agent metrics in a single report."""
        import evalytic.agent.runner as agent_runner

        # Patch TextJudge so we don't hit the provider factory
        class DummyJudge:
            judge_string = "fake-dummy"

            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def complete_json(self, prompt: str, *_a, **_k) -> dict:
                return {"score": 0.75, "reason": "ok"}

            def close(self) -> None:
                return None

        monkeypatch.setattr(agent_runner, "TextJudge", DummyJudge)
        # Make resolve_embedder raise so A3 fallback path runs
        monkeypatch.setattr(
            agent_runner, "resolve_embedder", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("no embedder")),
        )

        case = AgentTestCase(
            input="find the answer",
            final_output="42",
            expected_output="42",  # triggers embedder attempt -> fallback to LLM
            tool_calls=[AgentToolCall(name="search")],
            metadata={"expected_tool_calls": ["search"]},
        )
        report = evaluate_agent([case])
        metric_ids = {metric.metric_id for metric in report.results[0].metrics}
        assert metric_ids == {"tool_call_accuracy", "goal_accuracy", "step_efficiency"}
        assert report.eval_type == "agent"
