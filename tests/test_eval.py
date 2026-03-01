"""Basic tests for the Evalytic SDK.

These tests verify that the public API surface is importable and that the
core data types behave correctly.  They do NOT make any HTTP calls.
"""

from __future__ import annotations

import evalytic
from evalytic.types import (
    CompareReport,
    DimensionScore,
    EvalResult,
    ExperimentResult,
)
from evalytic.scorers import (
    ARTIFACT_DETECTION,
    INPUT_FIDELITY,
    PROMPT_ADHERENCE,
    TEXT_RENDERING,
    TRANSFORMATION_QUALITY,
    VISUAL_QUALITY,
    ALL_DIMENSIONS,
)
from evalytic.dataset import Dataset


# -----------------------------------------------------------------------
# Version
# -----------------------------------------------------------------------


def test_version_is_set() -> None:
    assert evalytic.__version__ == "0.3.2"


# -----------------------------------------------------------------------
# DimensionScore
# -----------------------------------------------------------------------


def test_dimension_score_creation() -> None:
    ds = DimensionScore(
        dimension="visual_quality",
        score=4.2,
        explanation="Sharp details, good composition.",
        evidence=["no artifacts detected", "high contrast"],
    )
    assert ds.dimension == "visual_quality"
    assert ds.score == 4.2
    assert len(ds.evidence) == 2


def test_dimension_score_defaults() -> None:
    ds = DimensionScore(dimension="prompt_adherence", score=3.5)
    assert ds.explanation == ""
    assert ds.evidence == []


# -----------------------------------------------------------------------
# EvalResult
# -----------------------------------------------------------------------


def test_eval_result_display_score() -> None:
    result = EvalResult(
        eval_result_id="er-001",
        experiment_id="exp-001",
        overall_score=3.85,
    )
    assert result.display_score == "3.9/5"


def test_eval_result_is_complete() -> None:
    processing = EvalResult(
        eval_result_id="er-001",
        experiment_id="exp-001",
        overall_score=0.0,
        status="processing",
    )
    assert not processing.is_complete

    completed = EvalResult(
        eval_result_id="er-002",
        experiment_id="exp-001",
        overall_score=4.0,
        status="completed",
    )
    assert completed.is_complete

    failed = EvalResult(
        eval_result_id="er-003",
        experiment_id="exp-001",
        overall_score=0.0,
        status="failed",
    )
    assert failed.is_complete


def test_eval_result_dimension_lookup() -> None:
    scores = [
        DimensionScore(dimension="visual_quality", score=4.0),
        DimensionScore(dimension="prompt_adherence", score=3.5),
    ]
    result = EvalResult(
        eval_result_id="er-001",
        experiment_id="exp-001",
        overall_score=3.75,
        scores=scores,
        status="completed",
    )
    vq = result.dimension("visual_quality")
    assert vq is not None
    assert vq.score == 4.0

    missing = result.dimension("nonexistent")
    assert missing is None


# -----------------------------------------------------------------------
# ExperimentResult
# -----------------------------------------------------------------------


def test_experiment_result_basics() -> None:
    exp = ExperimentResult(
        experiment_id="exp-001",
        name="test-experiment",
        status="completed",
        score=4.1,
        total_items=10,
        completed_items=10,
    )
    assert exp.is_complete
    assert exp.display_score == "4.1/5"


# -----------------------------------------------------------------------
# CompareReport
# -----------------------------------------------------------------------


def test_compare_report_ranking() -> None:
    results = {
        "model-a": ExperimentResult(
            experiment_id="exp-a",
            name="compare-model-a",
            status="completed",
            score=3.8,
            total_items=5,
            completed_items=5,
        ),
        "model-b": ExperimentResult(
            experiment_id="exp-b",
            name="compare-model-b",
            status="completed",
            score=4.5,
            total_items=5,
            completed_items=5,
        ),
        "model-c": ExperimentResult(
            experiment_id="exp-c",
            name="compare-model-c",
            status="completed",
            score=4.1,
            total_items=5,
            completed_items=5,
        ),
    }
    report = CompareReport(
        models=["model-a", "model-b", "model-c"],
        results=results,
        winner="model-b",
    )
    ranking = report.ranking
    assert ranking[0] == ("model-b", 4.5)
    assert ranking[-1] == ("model-a", 3.8)
    assert report.model_score("model-c") == 4.1
    assert report.model_score("nonexistent") is None


# -----------------------------------------------------------------------
# Dataset (local operations only)
# -----------------------------------------------------------------------


def test_dataset_insert_and_size() -> None:
    ds = Dataset(project_id="proj-1", name="test-set", pipeline_type="text2img")
    assert ds.size == 0
    assert len(ds) == 0

    ds.insert({"prompt": "A cat"})
    ds.insert({"prompt": "A dog"}, expected={"style": "photorealistic"})
    assert ds.size == 2
    assert len(ds) == 2
    assert not ds.is_pushed


def test_dataset_repr() -> None:
    ds = Dataset(project_id="proj-1", name="test-set")
    ds.insert({"prompt": "Hello"})
    r = repr(ds)
    assert "test-set" in r
    assert "local" in r


# -----------------------------------------------------------------------
# Scorer constants
# -----------------------------------------------------------------------


def test_scorer_constants_are_strings() -> None:
    assert VISUAL_QUALITY == "visual_quality"
    assert PROMPT_ADHERENCE == "prompt_adherence"
    assert TEXT_RENDERING == "text_rendering"
    assert INPUT_FIDELITY == "input_fidelity"
    assert TRANSFORMATION_QUALITY == "transformation_quality"
    assert ARTIFACT_DETECTION == "artifact_detection"


def test_all_dimensions_list() -> None:
    assert len(ALL_DIMENSIONS) == 7
    assert VISUAL_QUALITY in ALL_DIMENSIONS
    assert ARTIFACT_DETECTION in ALL_DIMENSIONS


# -----------------------------------------------------------------------
# Public API surface
# -----------------------------------------------------------------------


def test_public_api_exports() -> None:
    """Verify all expected names are accessible from the top-level package."""
    assert callable(evalytic.init)
    assert callable(evalytic.eval_image)
    assert callable(evalytic.compare)
    assert callable(evalytic.init_dataset)
    assert evalytic.Eval is not None
    assert evalytic.Dataset is not None
    assert evalytic.DimensionScore is not None
    assert evalytic.EvalResult is not None
    assert evalytic.ExperimentResult is not None
    assert evalytic.CompareReport is not None
