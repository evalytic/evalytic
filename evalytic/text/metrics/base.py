"""Base abstractions for text-like metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..types import MetricResult


class BaseTextMetric(ABC):
    metric_id: str = ""
    requires_judge: bool = False
    requires_embeddings: bool = False

    @abstractmethod
    def score(
        self,
        test_case: Any,
        *,
        judge: Any | None = None,
        embedder: Any | None = None,
    ) -> MetricResult:
        ...


def last_cost_of(target: Any | None) -> float:
    try:
        return float(getattr(target, "last_cost", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
