"""Tests for the shared ``merge_median_or_average`` helper used by text +
agent runners. Unifies the K1 regression guard.
"""

from __future__ import annotations

import pytest

from evalytic.exceptions import ValidationError
from evalytic.judge.consensus import (
    merge_median_or_average,
    resolve_metric_consensus,
)
from evalytic.text.types import MetricResult


def _make(metric: str, score: float, reason: str = "", details: dict | None = None) -> MetricResult:
    return MetricResult(metric_id=metric, score=score, reason=reason, details=details)


class TestMergeMedianOrAverage:
    def test_three_judges_picks_median(self) -> None:
        values = [
            _make("faithfulness", 0.60, "low"),
            _make("faithfulness", 0.80, "mid"),
            _make("faithfulness", 0.95, "high"),
        ]
        merged = merge_median_or_average(
            values,
            judge_names=["a", "b", "c"],
            agreement="high",
            metric_result_cls=MetricResult,
        )
        assert merged.score == pytest.approx(0.80, rel=1e-6)
        # Median judge's reason is taken
        assert merged.reason == "mid"
        assert merged.agreement == "high"

    def test_two_judges_averages(self) -> None:
        values = [
            _make("answer_relevancy", 0.70),
            _make("answer_relevancy", 0.90),
        ]
        merged = merge_median_or_average(
            values,
            judge_names=["a", "b"],
            agreement="high",
            metric_result_cls=MetricResult,
        )
        assert merged.score == pytest.approx(0.80, rel=1e-6)
        assert merged.judge == "consensus(a,b)"

    def test_one_judge_passthrough(self) -> None:
        value = _make("faithfulness", 0.55, "only-judge", details={"k": "v"})
        merged = merge_median_or_average(
            [value],
            judge_names=["solo"],
            agreement="degraded",
            metric_result_cls=MetricResult,
        )
        assert merged.score == pytest.approx(0.55, rel=1e-6)
        assert merged.reason == "only-judge"
        assert merged.judge == "consensus(solo)"
        assert merged.agreement == "degraded"
        assert merged.details is not None
        assert merged.details.get("k") == "v"

    def test_empty_raises(self) -> None:
        with pytest.raises(ValidationError):
            merge_median_or_average([], [], "high", metric_result_cls=MetricResult)

    def test_judge_scores_populated(self) -> None:
        values = [_make("m", 0.1), _make("m", 0.2), _make("m", 0.3)]
        merged = merge_median_or_average(
            values,
            judge_names=["x", "y", "z"],
            agreement="high",
            metric_result_cls=MetricResult,
        )
        assert merged.judge_scores == {"x": 0.1, "y": 0.2, "z": 0.3}

    def test_extra_details_builder_invoked(self) -> None:
        def builder(values, names, chosen):
            return {"judge_reasons": {name: v.reason for name, v in zip(names, values)}}

        values = [_make("m", 0.5, "ra"), _make("m", 0.7, "rb")]
        merged = merge_median_or_average(
            values,
            judge_names=["a", "b"],
            agreement="high",
            metric_result_cls=MetricResult,
            extra_details_builder=builder,
        )
        assert merged.details is not None
        assert merged.details["judge_reasons"] == {"a": "ra", "b": "rb"}


class TestResolveMetricConsensusIntegration:
    def test_two_judge_high_agreement_short_circuit(self) -> None:
        calls: list[str] = []

        def scorer(judge: str) -> MetricResult:
            calls.append(judge)
            return MetricResult(metric_id="x", score=0.70, judge=judge)

        merged = resolve_metric_consensus(
            ["a", "b", "c"],
            scorer=scorer,
            score_of=lambda item: item.score,
            merge=lambda values, names, agreement: merge_median_or_average(
                values, names, agreement, metric_result_cls=MetricResult
            ),
        )
        # Two agreeing judges -> third skipped
        assert calls == ["a", "b"]
        assert merged.agreement == "high"

    def test_disputed_with_third_tiebreaker(self) -> None:
        scores = {"a": 0.2, "b": 0.9, "c": 0.5}

        def scorer(judge: str) -> MetricResult:
            return MetricResult(metric_id="x", score=scores[judge], judge=judge)

        merged = resolve_metric_consensus(
            ["a", "b", "c"],
            scorer=scorer,
            score_of=lambda item: item.score,
            merge=lambda values, names, agreement: merge_median_or_average(
                values, names, agreement, metric_result_cls=MetricResult
            ),
        )
        # Median of {0.2, 0.9, 0.5} = 0.5
        assert merged.score == pytest.approx(0.5, rel=1e-6)
