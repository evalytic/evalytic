"""Deterministic text comparison metrics.

Pure scoring functions live in ``_deterministic.py`` so the Judge Lambda can
share them verbatim. This module is the SDK-side metric wrapper (adds
``MetricResult`` envelope + input validation).
"""

from __future__ import annotations

from typing import Any

from ...exceptions import ValidationError
from ._deterministic import (
    bleu_score,
    exact_match_score,
    levenshtein_distance,
    levenshtein_similarity,
    normalize,
    rouge_l_f1,
    string_presence_score,
    tokens,
)
from .base import BaseTextMetric
from ..types import MetricResult, TextTestCase


class BleuMetric(BaseTextMetric):
    metric_id = "bleu"

    def score(self, test_case: TextTestCase, *, judge: Any | None = None, embedder: Any | None = None) -> MetricResult:
        if not test_case.expected:
            raise ValidationError("bleu requires an expected/reference output.")
        if not tokens(test_case.output) or not tokens(test_case.expected):
            return MetricResult(metric_id=self.metric_id, score=0.0, reason="Missing candidate or reference tokens.")
        return MetricResult(
            metric_id=self.metric_id,
            score=bleu_score(test_case.output, test_case.expected),
            reason="Computed BLEU-4 with brevity penalty.",
        )


class RougeMetric(BaseTextMetric):
    metric_id = "rouge"

    def score(self, test_case: TextTestCase, *, judge: Any | None = None, embedder: Any | None = None) -> MetricResult:
        if not test_case.expected:
            raise ValidationError("rouge requires an expected/reference output.")
        if not tokens(test_case.output) or not tokens(test_case.expected):
            return MetricResult(metric_id=self.metric_id, score=0.0, reason="Missing candidate or reference tokens.")
        return MetricResult(
            metric_id=self.metric_id,
            score=rouge_l_f1(test_case.output, test_case.expected),
            reason="Computed ROUGE-L F1.",
        )


class ExactMatchMetric(BaseTextMetric):
    metric_id = "exact_match"

    def score(self, test_case: TextTestCase, *, judge: Any | None = None, embedder: Any | None = None) -> MetricResult:
        if not test_case.expected:
            raise ValidationError("exact_match requires an expected/reference output.")
        return MetricResult(
            metric_id=self.metric_id,
            score=exact_match_score(test_case.output, test_case.expected),
            reason="Exact normalized string match.",
        )


class LevenshteinMetric(BaseTextMetric):
    metric_id = "levenshtein"

    def score(self, test_case: TextTestCase, *, judge: Any | None = None, embedder: Any | None = None) -> MetricResult:
        if not test_case.expected:
            raise ValidationError("levenshtein requires an expected/reference output.")
        output = normalize(test_case.output)
        expected = normalize(test_case.expected)
        distance = levenshtein_distance(output, expected)
        return MetricResult(
            metric_id=self.metric_id,
            score=levenshtein_similarity(test_case.output, test_case.expected),
            reason="Normalized Levenshtein similarity.",
            details={"distance": distance},
        )


class StringPresenceMetric(BaseTextMetric):
    metric_id = "string_presence"

    def score(self, test_case: TextTestCase, *, judge: Any | None = None, embedder: Any | None = None) -> MetricResult:
        if not test_case.expected:
            raise ValidationError("string_presence requires an expected/reference output.")
        return MetricResult(
            metric_id=self.metric_id,
            score=string_presence_score(test_case.output, test_case.expected),
            reason="Expected string presence check.",
        )
