"""Core data types for the Evalytic SDK.

All types are plain dataclasses -- no external dependencies required.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DimensionScore:
    """Score for a single evaluation dimension (e.g. visual_quality, prompt_adherence)."""

    dimension: str
    score: float
    explanation: str = ""
    evidence: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"DimensionScore(dimension={self.dimension!r}, score={self.score})"


@dataclass
class EvalResult:
    """Result of evaluating a single image/video against one or more dimensions."""

    eval_result_id: str
    experiment_id: str
    overall_score: float
    scores: list[DimensionScore] = field(default_factory=list)
    status: str = "processing"
    raw: dict = field(default_factory=dict)

    @property
    def display_score(self) -> str:
        """Human-readable score string, e.g. '3.8/5'."""
        return f"{self.overall_score:.1f}/5"

    @property
    def is_complete(self) -> bool:
        return self.status in ("completed", "failed", "error")

    def dimension(self, name: str) -> DimensionScore | None:
        """Look up a specific dimension score by name."""
        for s in self.scores:
            if s.dimension == name:
                return s
        return None


@dataclass
class ExperimentResult:
    """Aggregated result of an experiment (many images evaluated together)."""

    experiment_id: str
    name: str
    status: str
    score: float
    total_items: int
    completed_items: int
    scores: list[EvalResult] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return self.status in ("completed", "failed", "error")

    @property
    def display_score(self) -> str:
        return f"{self.score:.1f}/5"


@dataclass
class CompareReport:
    """Side-by-side comparison of multiple models on the same dataset."""

    models: list[str]
    results: dict[str, ExperimentResult] = field(default_factory=dict)
    winner: str = ""

    @property
    def ranking(self) -> list[tuple[str, float]]:
        """Return models sorted by score, descending."""
        pairs = [(m, self.results[m].score) for m in self.models if m in self.results]
        return sorted(pairs, key=lambda p: p[1], reverse=True)

    def model_score(self, model: str) -> float | None:
        """Get the overall score for a specific model."""
        result = self.results.get(model)
        return result.score if result else None
