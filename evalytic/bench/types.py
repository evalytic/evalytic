"""Data types for bench reports.

Separate from ``sdk/evalytic/types.py`` (cloud SDK types) because bench types
include fields for local-first workflows (``human_score``, ``human_notes``)
and cost tracking that don't belong in the cloud API contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunError:
    """An error or warning captured during a bench run."""

    timestamp: str  # ISO-8601
    level: str  # "error" | "warning"
    phase: str  # "generation" | "scoring" | "metrics"
    model: str
    item_id: str  # "" for model-level warnings
    dimension: str  # "" for generation errors
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "level": self.level,
            "phase": self.phase,
            "model": self.model,
            "item_id": self.item_id,
            "dimension": self.dimension,
            "message": self.message,
        }


@dataclass
class MetricScoringConfig:
    """Configuration for how a single metric contributes to overall scoring."""

    flag_threshold: float  # Below this → flag warning, excluded from overall
    weight: float  # Weight in overall score (e.g. 0.20)
    normalize_range: tuple[float, float]  # (min, max) for linear map to 0-5


@dataclass
class ScoringConfig:
    """Configuration for metric-weighted overall scoring."""

    metric_configs: dict[str, MetricScoringConfig] = field(default_factory=dict)

    @staticmethod
    def default() -> ScoringConfig:
        """Sensible defaults for CLIP, LPIPS, and face similarity."""
        return ScoringConfig(
            metric_configs={
                "clip_score": MetricScoringConfig(
                    flag_threshold=0.18,
                    weight=0.20,
                    normalize_range=(0.18, 0.35),
                ),
                "lpips": MetricScoringConfig(
                    flag_threshold=0.40,
                    weight=0.20,
                    normalize_range=(0.40, 0.95),
                ),
                "face_similarity": MetricScoringConfig(
                    flag_threshold=0.60,
                    weight=0.20,
                    normalize_range=(0.60, 0.95),
                ),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            name: {
                "flag_threshold": cfg.flag_threshold,
                "weight": cfg.weight,
                "normalize_range": list(cfg.normalize_range),
            }
            for name, cfg in self.metric_configs.items()
        }


@dataclass
class MetricFlag:
    """A flag indicating a metric fell below its threshold."""

    metric: str
    value: float
    threshold: float
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "message": self.message,
        }


@dataclass
class DimensionResult:
    """Score for one dimension on one image."""

    dimension: str
    score: float
    confidence: float = 1.0
    explanation: str = ""
    evidence: list[str] = field(default_factory=list)
    human_score: float | None = None
    human_notes: str = ""
    judge_scores: dict[str, float] = field(default_factory=dict)
    agreement: str = ""  # "high", "disputed", "degraded", or ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "score": self.score,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "evidence": self.evidence,
        }
        if self.human_score is not None:
            d["human_score"] = self.human_score
        if self.human_notes:
            d["human_notes"] = self.human_notes
        if self.judge_scores:
            d["judge_scores"] = self.judge_scores
        if self.agreement:
            d["agreement"] = self.agreement
        return d


@dataclass
class MetricResult:
    """Local metric result for one image (CLIP / LPIPS)."""

    metric: str  # "clip_score" or "lpips"
    value: float
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "description": self.description}


@dataclass
class ImageResult:
    """Result for one model on one prompt/input."""

    model: str
    image_url: str
    image_local: str = ""
    generation_time_ms: int = 0
    generation_cost_usd: float = 0.0
    scores: list[DimensionResult] = field(default_factory=list)
    metrics: list[MetricResult] = field(default_factory=list)
    flags: list[MetricFlag] = field(default_factory=list)
    status: str = "success"
    error: str = ""

    @property
    def overall_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.score for s in self.scores) / len(self.scores)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "image_url": self.image_url,
            "image_local": self.image_local,
            "generation_time_ms": self.generation_time_ms,
            "generation_cost_usd": self.generation_cost_usd,
            "scores": {s.dimension: s.to_dict() for s in self.scores},
            "metrics": {m.metric: m.to_dict() for m in self.metrics},
            "overall_score": self.overall_score,
            "status": self.status,
        }
        if self.flags:
            d["flags"] = [f.to_dict() for f in self.flags]
        if self.error:
            d["error"] = self.error
        return d


@dataclass
class BenchItem:
    """Results for one prompt/input across all models."""

    item_id: str
    prompt: str = ""
    image_url: str = ""  # img2img input
    instruction: str = ""  # img2img instruction
    tags: list[str] = field(default_factory=list)
    results: dict[str, ImageResult] = field(default_factory=dict)  # model -> result

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"item_id": self.item_id}
        if self.prompt:
            d["prompt"] = self.prompt
        if self.image_url:
            d["image_url"] = self.image_url
        if self.instruction:
            d["instruction"] = self.instruction
        if self.tags:
            d["tags"] = self.tags
        d["results"] = {m: r.to_dict() for m, r in self.results.items()}
        return d


@dataclass
class ModelSummary:
    """Aggregated scores for one model across all items."""

    model: str
    overall_score: float
    dimension_averages: dict[str, float] = field(default_factory=dict)
    dimension_confidence: dict[str, float] = field(default_factory=dict)
    avg_confidence: float = 1.0
    metric_averages: dict[str, float] = field(default_factory=dict)
    metric_flags: dict[str, str] = field(default_factory=dict)  # "flagged" or "included"
    weighted_overall_score: float | None = None  # None → pure VLM average
    agreement_rate: float = 1.0  # fraction of dimensions with "high" agreement
    total_generation_cost_usd: float = 0.0
    total_generation_time_ms: int = 0
    item_count: int = 0
    total_items: int = 0  # includes failed items
    cost_per_image: float = 0.0
    score_per_dollar: float = 0.0  # overall_score / cost_per_image — higher is better
    # Statistical fields
    dimension_stddev: dict[str, float] = field(default_factory=dict)
    overall_stddev: float = 0.0
    sample_count: int = 0  # number of items scored (excludes failures)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "overall_score": self.overall_score,
            "dimension_averages": self.dimension_averages,
            "dimension_confidence": self.dimension_confidence,
            "avg_confidence": self.avg_confidence,
            "metric_averages": self.metric_averages,
            "total_generation_cost_usd": self.total_generation_cost_usd,
            "total_generation_time_ms": self.total_generation_time_ms,
            "cost_per_image": self.cost_per_image,
            "score_per_dollar": self.score_per_dollar,
        }
        if self.dimension_stddev:
            d["dimension_stddev"] = self.dimension_stddev
        if self.overall_stddev > 0:
            d["overall_stddev"] = self.overall_stddev
        if self.sample_count > 0:
            d["sample_count"] = self.sample_count
        if self.total_items > 0:
            d["total_items"] = self.total_items
            d["success_rate"] = round(self.item_count / self.total_items, 2) if self.total_items > 0 else 1.0
        if self.metric_flags:
            d["metric_flags"] = self.metric_flags
        if self.weighted_overall_score is not None:
            d["weighted_overall_score"] = self.weighted_overall_score
        if self.agreement_rate < 1.0:
            d["agreement_rate"] = self.agreement_rate
        return d


@dataclass
class CostBreakdown:
    """Cost tracking for the bench run."""

    generation_total_usd: float = 0.0
    generation_by_model: dict[str, float] = field(default_factory=dict)
    generation_request_count: int = 0
    judge_total_usd: float = 0.0
    judge_provider: str = ""
    judge_request_count: int = 0
    judge_providers: list[str] = field(default_factory=list)
    judge_cost_by_provider: dict[str, float] = field(default_factory=dict)
    metrics_total_usd: float = 0.0
    total_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "generation": {
                "total_usd": self.generation_total_usd,
                "by_model": self.generation_by_model,
                "request_count": self.generation_request_count,
            },
            "judge": {
                "total_usd": self.judge_total_usd,
                "provider": self.judge_provider,
                "request_count": self.judge_request_count,
            },
            "metrics": {
                "total_usd": self.metrics_total_usd,
                "note": "Local computation",
            },
            "total_usd": self.total_usd,
        }
        if self.judge_providers:
            d["judge"]["providers"] = self.judge_providers
        if self.judge_cost_by_provider:
            d["judge"]["cost_by_provider"] = self.judge_cost_by_provider
        return d


@dataclass
class CorrelationStat:
    """Correlation between VLM and objective metric."""

    metric_pair: str  # e.g. "clip_vs_prompt_adherence"
    pearson_r: float
    p_value: float
    interpretation: str  # "high_agreement", "moderate", "low_agreement"

    def to_dict(self) -> dict[str, Any]:
        return {
            "pearson_r": self.pearson_r,
            "p_value": self.p_value,
            "interpretation": self.interpretation,
        }


@dataclass
class BenchReport:
    """Complete result of an evalytic bench run."""

    name: str
    models: list[str]
    judge: str
    pipeline: str  # "text2img" or "img2img"
    judges: list[str] = field(default_factory=list)
    consensus_mode: bool = False
    dimensions: list[str] = field(default_factory=list)
    items: list[BenchItem] = field(default_factory=list)
    summary: dict[str, ModelSummary] = field(default_factory=dict)
    winner: str = ""
    ranking: list[tuple[str, float]] = field(default_factory=list)
    efficiency_ranking: list[tuple[str, float]] = field(default_factory=list)  # by score_per_dollar
    best_value: str = ""  # model with highest score_per_dollar
    cost: CostBreakdown = field(default_factory=CostBreakdown)
    correlations: list[CorrelationStat] = field(default_factory=list)
    duration_seconds: float = 0.0
    created_at: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[RunError] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        from ..report.json_report import report_to_dict

        return report_to_dict(self)

    def to_json(self, path: str) -> None:
        from ..report.json_report import write_json

        write_json(self, path)

    def to_html(self, path: str) -> None:
        from ..report.html_report import write_html

        write_html(self, path)

    def open_review(self, port: int = 3847) -> None:
        """Open a local browser review for human-in-the-loop scoring.

        Starts a local HTTP server, opens the browser, and blocks until
        the user clicks "Done". Human scores are merged into this report
        in-place.
        """
        from ..report.review_server import ReviewServer

        server = ReviewServer(self, port=port)
        server.start()
        server.wait()
        # Scores are merged in-place via get_report() -> _merge_scores()
        server.get_report()
