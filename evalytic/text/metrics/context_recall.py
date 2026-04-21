"""Reference-based context recall metric."""

from __future__ import annotations

import time
from typing import Any

from ...exceptions import ValidationError
from .base import BaseTextMetric, last_cost_of
from ..types import MetricResult, RAGTestCase


class ContextRecallMetric(BaseTextMetric):
    metric_id = "context_recall"
    requires_judge = True

    def score(
        self,
        test_case: RAGTestCase,
        *,
        judge: Any | None = None,
        embedder: Any | None = None,
    ) -> MetricResult:
        if judge is None:
            raise ValidationError("context_recall requires an LLM judge.")
        if not test_case.reference:
            raise ValidationError("context_recall requires a reference answer.")

        started = time.perf_counter()
        prompt = (
            "Break the reference answer into atomic claims and judge whether each claim is supported "
            "by the retrieved contexts. Return JSON with this shape:\n"
            '{"claims":[{"text":"...", "supported": true, "evidence":"..."}], "reason":"..."}\n\n'
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
        claims = raw.get("claims", [])
        if not claims:
            return MetricResult(
                metric_id=self.metric_id,
                score=0.0,
                reason=raw.get("reason") or "No reference claims were extracted.",
                details={"claims": []},
                judge=judge.judge_string,
                cost=judge_cost,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        normalized = [
            {
                "text": claim.get("text", ""),
                "supported": bool(claim.get("supported", False)),
                "evidence": claim.get("evidence", ""),
            }
            for claim in claims
        ]
        supported_count = sum(1 for claim in normalized if claim["supported"])
        score = supported_count / len(normalized)
        return MetricResult(
            metric_id=self.metric_id,
            score=score,
            reason=raw.get("reason")
            or "Measured how many reference claims are covered by retrieved contexts.",
            details={
                "claims": normalized,
                "supported_count": supported_count,
                "total_count": len(normalized),
            },
            judge=judge.judge_string,
            cost=judge_cost,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
