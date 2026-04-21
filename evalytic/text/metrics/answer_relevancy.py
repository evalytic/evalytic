"""RAG answer relevancy metric."""

from __future__ import annotations

import time
from typing import Any

from ...exceptions import ValidationError
from ..embeddings import cosine_similarity
from ..metrics.base import BaseTextMetric, last_cost_of
from ..types import MetricResult, RAGTestCase


class AnswerRelevancyMetric(BaseTextMetric):
    metric_id = "answer_relevancy"
    requires_judge = True
    requires_embeddings = True

    def score(
        self,
        test_case: RAGTestCase,
        *,
        judge: Any | None = None,
        embedder: Any | None = None,
    ) -> MetricResult:
        if judge is None:
            raise ValidationError("answer_relevancy requires an LLM judge.")
        if embedder is None:
            raise ValidationError("answer_relevancy requires embeddings.")

        started = time.perf_counter()
        prompt = (
            "Generate 3 reverse-engineered user questions that this answer appears to answer.\n"
            "Return JSON with this shape:\n"
            '{"generated_questions":["...","...","..."], "reason":"..."}\n\n'
            f"Original user question:\n{test_case.query}\n\n"
            f"Answer:\n{test_case.response}"
        )
        raw = judge.complete_json(prompt)
        judge_cost = last_cost_of(judge)
        generated_questions = [
            question for question in raw.get("generated_questions", []) if question
        ]
        if not generated_questions:
            return MetricResult(
                metric_id=self.metric_id,
                score=0.0,
                reason=raw.get("reason")
                or "Judge did not produce any reverse questions; treating as irrelevant.",
                details={"generated_questions": [], "similarities": []},
                judge=judge.judge_string,
                cost=judge_cost,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        vectors = embedder.embed_texts([test_case.query, *generated_questions])
        embedding_cost = last_cost_of(embedder)
        query_vector = vectors[0]
        similarities = [cosine_similarity(query_vector, vector) for vector in vectors[1:]]
        score = sum(similarities) / len(similarities) if similarities else 0.0

        return MetricResult(
            metric_id=self.metric_id,
            score=score,
            reason=raw.get("reason")
            or "Generated reverse questions were embedded and compared to the original query.",
            details={
                "generated_questions": generated_questions,
                "similarities": [round(value, 4) for value in similarities],
            },
            judge=judge.judge_string,
            cost=judge_cost + embedding_cost,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
