"""Shared types for metric-first eval domains."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class RetrievedChunk:
    text: str
    chunk_id: str | None = None
    source: str | None = None
    rank: int | None = None
    retrieval_score: float | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class RAGTestCase:
    query: str
    response: str
    contexts: list[RetrievedChunk]
    reference: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class TextTestCase:
    input: str
    output: str
    expected: str | None = None
    criteria: str | None = None
    context: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class MetricResult:
    metric_id: str
    score: float
    reason: str | None = None
    details: dict[str, Any] | None = None
    judge: str | None = None
    judge_scores: dict[str, float] | None = None
    agreement: str | None = None
    cost: float = 0.0
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "metric_id": self.metric_id,
            "score": round(self.score, 4),
            "cost": round(self.cost, 6),
            "duration_ms": self.duration_ms,
        }
        if self.reason is not None:
            data["reason"] = self.reason
        if self.details is not None:
            data["details"] = self.details
        if self.judge is not None:
            data["judge"] = self.judge
        if self.judge_scores is not None:
            data["judge_scores"] = self.judge_scores
        if self.agreement is not None:
            data["agreement"] = self.agreement
        return data


@dataclass
class MetricEvalResult:
    case_id: str
    test_case: Any
    metrics: list[MetricResult]
    total_cost: float
    duration_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "test_case": _serialize_case(self.test_case),
            "metrics": [metric.to_dict() for metric in self.metrics],
            "total_cost": round(self.total_cost, 6),
            "duration_ms": self.duration_ms,
        }


TextEvalResult = MetricEvalResult


def _default_version() -> str:
    from .. import __version__

    return __version__


@dataclass
class MetricEvalReport:
    eval_type: str
    judge: str
    judges: list[str] = field(default_factory=list)
    consensus_mode: bool = False
    results: list[MetricEvalResult] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = field(default_factory=_default_version)

    def metric_averages(self) -> dict[str, float]:
        buckets: dict[str, list[float]] = {}
        for result in self.results:
            for metric in result.metrics:
                buckets.setdefault(metric.metric_id, []).append(metric.score)
        return {
            metric_id: round(sum(values) / len(values), 4)
            for metric_id, values in buckets.items()
            if values
        }

    def to_dict(self) -> dict[str, Any]:
        total_cost = round(sum(result.total_cost for result in self.results), 6)
        total_duration_ms = sum(result.duration_ms for result in self.results)
        data: dict[str, Any] = {
            "evalytic_version": self.version,
            "eval_type": self.eval_type,
            "timestamp": self.created_at,
            "judge": self.judge,
            "judges": self.judges or None,
            "consensus_mode": self.consensus_mode,
            "results": [result.to_dict() for result in self.results],
            "summary": {
                "total_cases": len(self.results),
                "total_cost": total_cost,
                "total_duration_ms": total_duration_ms,
                "metric_averages": self.metric_averages(),
            },
            "metadata": self.metadata,
        }
        return data


def _serialize_case(case: Any) -> Any:
    if is_dataclass(case):
        return asdict(case)
    return case
