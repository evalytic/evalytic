"""RAG faithfulness metric."""

from __future__ import annotations

import json
import time
from typing import Any

from ...exceptions import ValidationError
from ..metrics.base import BaseTextMetric, last_cost_of
from ..types import MetricResult, RAGTestCase


class FaithfulnessMetric(BaseTextMetric):
    metric_id = "faithfulness"
    requires_judge = True

    def score(
        self,
        test_case: RAGTestCase,
        *,
        judge: Any | None = None,
        embedder: Any | None = None,
    ) -> MetricResult:
        if judge is None:
            raise ValidationError("faithfulness requires an LLM judge.")
        if not test_case.contexts:
            raise ValidationError("faithfulness requires at least one context chunk.")

        started = time.perf_counter()
        contexts = "\n\n".join(
            f"[{idx + 1}] {chunk.text}" for idx, chunk in enumerate(test_case.contexts)
        )
        prompt = (
            "Evaluate the factual faithfulness of a RAG answer against retrieved context.\n"
            "Return JSON with this shape:\n"
            '{"claims":[{"text":"...", "supported": true, "evidence":"..."}], "reason":"..."}\n\n'
            f"Question:\n{test_case.query}\n\n"
            f"Answer:\n{test_case.response}\n\n"
            f"Retrieved context:\n{contexts}"
        )
        raw = judge.complete_json(prompt)
        claims = raw.get("claims", [])
        if not claims:
            score = 0.0
            reason = raw.get("reason") or "No claims could be extracted from the answer."
            details = {"claims": [], "supported_count": 0, "total_count": 0}
        else:
            normalized_claims = [
                {
                    "text": claim.get("text", ""),
                    "supported": bool(claim.get("supported", False)),
                    "evidence": claim.get("evidence", ""),
                }
                for claim in claims
            ]
            supported_count = sum(1 for claim in normalized_claims if claim["supported"])
            score = supported_count / len(normalized_claims)
            reason = raw.get("reason") or (
                f"{supported_count} of {len(normalized_claims)} claims are supported by the retrieved context."
            )
            details = {
                "claims": normalized_claims,
                "supported_count": supported_count,
                "total_count": len(normalized_claims),
            }

        return MetricResult(
            metric_id=self.metric_id,
            score=score,
            reason=reason,
            details=details,
            judge=judge.judge_string,
            cost=last_cost_of(judge),
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
