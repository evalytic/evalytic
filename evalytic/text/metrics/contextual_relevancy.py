"""Reference-free contextual relevancy metric."""

from __future__ import annotations

import time
from typing import Any

from ...exceptions import ValidationError
from .base import BaseTextMetric, last_cost_of
from ..types import MetricResult, RAGTestCase


class ContextualRelevancyMetric(BaseTextMetric):
    metric_id = "contextual_relevancy"
    requires_judge = True

    def score(
        self,
        test_case: RAGTestCase,
        *,
        judge: Any | None = None,
        embedder: Any | None = None,
    ) -> MetricResult:
        if judge is None:
            raise ValidationError("contextual_relevancy requires an LLM judge.")
        if not test_case.contexts:
            raise ValidationError("contextual_relevancy requires at least one context chunk.")

        started = time.perf_counter()
        contexts = "\n\n".join(
            f"[{idx + 1}] {chunk.text}" for idx, chunk in enumerate(test_case.contexts)
        )
        prompt = (
            "Evaluate whether each retrieved context chunk is relevant to the query. "
            "Judge each chunk independently; do not rank. A chunk is relevant if it "
            "contains information that helps answer the query, even partially.\n"
            "Return JSON with this shape:\n"
            '{"chunks":[{"index":1, "relevant": true, "reason":"..."}], "reason":"..."}\n\n'
            f"Query:\n{test_case.query}\n\n"
            f"Retrieved contexts:\n{contexts}"
        )
        raw = judge.complete_json(prompt)
        judge_cost = last_cost_of(judge)
        judged_chunks = raw.get("chunks", [])
        total = len(test_case.contexts)

        if not judged_chunks:
            return MetricResult(
                metric_id=self.metric_id,
                score=0.0,
                reason=raw.get("reason") or "No chunk relevance judgments were produced.",
                details={"chunks": [], "relevant_count": 0, "total_count": total},
                judge=judge.judge_string,
                cost=judge_cost,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        normalized = []
        for idx, item in enumerate(judged_chunks, start=1):
            normalized.append(
                {
                    "index": item.get("index", idx),
                    "relevant": bool(item.get("relevant", False)),
                    "reason": item.get("reason", ""),
                }
            )
        relevant_count = sum(1 for item in normalized if item["relevant"])
        score = relevant_count / total
        reason = raw.get("reason") or (
            f"{relevant_count} of {total} retrieved chunks are relevant to the query."
        )

        return MetricResult(
            metric_id=self.metric_id,
            score=score,
            reason=reason,
            details={
                "chunks": normalized,
                "relevant_count": relevant_count,
                "total_count": total,
            },
            judge=judge.judge_string,
            cost=judge_cost,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
