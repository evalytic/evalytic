"""Reference-based factual correctness metric."""

from __future__ import annotations

import time
from typing import Any

from ...exceptions import ValidationError
from .base import BaseTextMetric, last_cost_of
from ..types import MetricResult, TextTestCase


class FactualCorrectnessMetric(BaseTextMetric):
    metric_id = "factual_correctness"
    requires_judge = True

    def score(
        self,
        test_case: TextTestCase,
        *,
        judge: Any | None = None,
        embedder: Any | None = None,
    ) -> MetricResult:
        if judge is None:
            raise ValidationError("factual_correctness requires an LLM judge.")
        if not test_case.expected:
            raise ValidationError("factual_correctness requires an expected/reference output.")

        started = time.perf_counter()
        prompt = (
            "Compare a model output to the reference answer. Break both into claims and classify "
            "response claims into true_positives, false_positives, and false_negatives relative to the reference.\n"
            "Return JSON with this shape:\n"
            '{"true_positives":["..."], "false_positives":["..."], "false_negatives":["..."], "reason":"..."}\n\n'
            f"Input:\n{test_case.input}\n\n"
            f"Output:\n{test_case.output}\n\n"
            f"Reference:\n{test_case.expected}"
        )
        raw = judge.complete_json(prompt)
        tp = [item for item in raw.get("true_positives", []) if item]
        fp = [item for item in raw.get("false_positives", []) if item]
        fn = [item for item in raw.get("false_negatives", []) if item]
        precision = len(tp) / (len(tp) + len(fp)) if (tp or fp) else 0.0
        recall = len(tp) / (len(tp) + len(fn)) if (tp or fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        return MetricResult(
            metric_id=self.metric_id,
            score=f1,
            reason=raw.get("reason")
            or "Computed claim-level factual overlap F1 between output and reference.",
            details={
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            },
            judge=judge.judge_string,
            cost=last_cost_of(judge),
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
