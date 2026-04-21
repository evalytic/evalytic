"""Embedding-based semantic similarity metric."""

from __future__ import annotations

from typing import Any

from ...exceptions import ValidationError
from ..embeddings import cosine_similarity
from .base import BaseTextMetric, last_cost_of
from ..types import MetricResult, TextTestCase


class SemanticSimilarityMetric(BaseTextMetric):
    metric_id = "semantic_similarity"
    requires_embeddings = True

    def score(
        self,
        test_case: TextTestCase,
        *,
        judge: Any | None = None,
        embedder: Any | None = None,
    ) -> MetricResult:
        if embedder is None:
            raise ValidationError("semantic_similarity requires embeddings.")
        if not test_case.expected:
            raise ValidationError("semantic_similarity requires an expected/reference output.")

        vectors = embedder.embed_texts([test_case.output, test_case.expected])
        score = cosine_similarity(vectors[0], vectors[1])
        return MetricResult(
            metric_id=self.metric_id,
            score=score,
            reason="Computed cosine similarity between output and expected embeddings.",
            details={"comparison": "output_vs_expected"},
            cost=last_cost_of(embedder),
        )
