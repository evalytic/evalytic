"""Rubric-style general criteria scoring."""

from __future__ import annotations

import time
from typing import Any

from ...exceptions import ValidationError
from .base import BaseTextMetric, last_cost_of
from ..types import MetricResult, TextTestCase


class GEvalMetric(BaseTextMetric):
    metric_id = "g_eval"
    requires_judge = True

    def __init__(self, scale_max: int = 5) -> None:
        self.scale_max = scale_max

    def score(
        self,
        test_case: TextTestCase,
        *,
        judge: Any | None = None,
        embedder: Any | None = None,
    ) -> MetricResult:
        if judge is None:
            raise ValidationError("g_eval requires an LLM judge.")
        criteria = test_case.criteria or (test_case.metadata or {}).get("criteria")
        if not criteria:
            raise ValidationError("g_eval requires evaluation criteria.")

        started = time.perf_counter()
        prompt = (
            "Evaluate the output against the criteria and score it on the provided rubric.\n"
            "Return JSON with this shape:\n"
            '{"score": 4, "max_score": 5, "reason":"...", "rubric":["..."]}\n\n'
            f"Criteria:\n{criteria}\n\n"
            f"Input:\n{test_case.input}\n\n"
            f"Output:\n{test_case.output}\n"
        )
        if test_case.expected:
            prompt += f"\nReference:\n{test_case.expected}\n"
        raw = judge.complete_json(prompt)
        max_score = int(raw.get("max_score", self.scale_max) or self.scale_max)
        score = float(raw.get("score", 0.0))
        normalized = max(0.0, min(1.0, score / max_score if max_score else 0.0))
        return MetricResult(
            metric_id=self.metric_id,
            score=normalized,
            reason=raw.get("reason") or "Rubric-based LLM evaluation.",
            details={
                "raw_score": score,
                "max_score": max_score,
                "rubric": raw.get("rubric", []),
            },
            judge=judge.judge_string,
            cost=last_cost_of(judge),
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
