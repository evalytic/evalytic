"""Hallucination detection metric (contradiction-based)."""

from __future__ import annotations

import time
from typing import Any

from ...exceptions import ValidationError
from .base import BaseTextMetric, last_cost_of
from ..types import MetricResult, RAGTestCase


class HallucinationMetric(BaseTextMetric):
    metric_id = "hallucination"
    requires_judge = True

    def score(
        self,
        test_case: RAGTestCase,
        *,
        judge: Any | None = None,
        embedder: Any | None = None,
    ) -> MetricResult:
        if judge is None:
            raise ValidationError("hallucination requires an LLM judge.")
        if not test_case.contexts:
            raise ValidationError("hallucination requires at least one context chunk.")

        started = time.perf_counter()
        contexts = "\n\n".join(
            f"[{idx + 1}] {chunk.text}" for idx, chunk in enumerate(test_case.contexts)
        )
        prompt = (
            "Detect hallucinations in a RAG answer. Decompose the answer into atomic claims. "
            "For each claim, decide whether any of the retrieved context chunks CONTRADICTS it. "
            "A claim is contradicted only when the context states something incompatible with "
            "the claim; a claim that is simply absent from the context is NOT contradicted.\n"
            "Return JSON with this shape:\n"
            '{"claims":[{"text":"...", "contradicted": true, "evidence":"..."}], "reason":"..."}\n\n'
            f"Question:\n{test_case.query}\n\n"
            f"Answer:\n{test_case.response}\n\n"
            f"Retrieved context:\n{contexts}"
        )
        raw = judge.complete_json(prompt)
        judge_cost = last_cost_of(judge)
        claims = raw.get("claims", [])

        if not claims:
            return MetricResult(
                metric_id=self.metric_id,
                score=1.0,
                reason=raw.get("reason") or "No atomic claims extracted from the answer.",
                details={"claims": [], "contradicted_count": 0, "total_count": 0},
                judge=judge.judge_string,
                cost=judge_cost,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        normalized_claims = [
            {
                "text": claim.get("text", ""),
                "contradicted": bool(claim.get("contradicted", False)),
                "evidence": claim.get("evidence", ""),
            }
            for claim in claims
        ]
        contradicted_count = sum(1 for claim in normalized_claims if claim["contradicted"])
        score = 1.0 - (contradicted_count / len(normalized_claims))
        reason = raw.get("reason") or (
            f"{contradicted_count} of {len(normalized_claims)} claims are contradicted by the retrieved context."
        )

        return MetricResult(
            metric_id=self.metric_id,
            score=score,
            reason=reason,
            details={
                "claims": normalized_claims,
                "contradicted_count": contradicted_count,
                "total_count": len(normalized_claims),
            },
            judge=judge.judge_string,
            cost=judge_cost,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
