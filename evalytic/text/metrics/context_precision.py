"""Reference-based context precision metric."""

from __future__ import annotations

import time
from typing import Any

from ...exceptions import ValidationError
from .base import BaseTextMetric, last_cost_of
from ..types import MetricResult, RAGTestCase


class ContextPrecisionMetric(BaseTextMetric):
    metric_id = "context_precision"
    requires_judge = True

    def score(
        self,
        test_case: RAGTestCase,
        *,
        judge: Any | None = None,
        embedder: Any | None = None,
    ) -> MetricResult:
        if judge is None:
            raise ValidationError("context_precision requires an LLM judge.")
        if not test_case.reference:
            raise ValidationError("context_precision requires a reference answer.")

        started = time.perf_counter()
        prompt = (
            "Determine whether each retrieved context chunk is relevant for answering the question "
            "given the reference answer. Return JSON with this shape:\n"
            '{"contexts":[{"rank":1, "relevant": true, "reason":"..."}], "reason":"..."}\n\n'
            f"Question:\n{test_case.query}\n\n"
            f"Reference answer:\n{test_case.reference}\n\n"
            "Retrieved contexts:\n"
            + "\n".join(
                f"[{chunk.rank or idx + 1}] {chunk.text}"
                for idx, chunk in enumerate(test_case.contexts)
            )
        )
        raw = judge.complete_json(prompt)
        judge_cost = last_cost_of(judge)
        judged_contexts = raw.get("contexts", [])
        if not judged_contexts:
            return MetricResult(
                metric_id=self.metric_id,
                score=0.0,
                reason=raw.get("reason") or "No context relevance judgments were produced.",
                details={"contexts": []},
                judge=judge.judge_string,
                cost=judge_cost,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        relevant_total = sum(1 for item in judged_contexts if item.get("relevant"))
        precision_sum = 0.0
        true_positive_count = 0
        normalized = []
        for idx, item in enumerate(judged_contexts, start=1):
            relevant = bool(item.get("relevant", False))
            if relevant:
                true_positive_count += 1
                precision_sum += true_positive_count / idx
            normalized.append(
                {
                    "rank": item.get("rank", idx),
                    "relevant": relevant,
                    "reason": item.get("reason", ""),
                }
            )

        score = precision_sum / relevant_total if relevant_total else 0.0
        return MetricResult(
            metric_id=self.metric_id,
            score=score,
            reason=raw.get("reason")
            or "Computed mean precision@k across chunks marked relevant to the reference answer.",
            details={"contexts": normalized, "relevant_count": relevant_total},
            judge=judge.judge_string,
            cost=judge_cost,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
