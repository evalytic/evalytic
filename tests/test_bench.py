"""Tests for the evalytic bench module.

All tests are offline -- zero network calls.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from evalytic.bench.types import (
    BenchItem,
    BenchReport,
    CostBreakdown,
    DimensionResult,
    ImageResult,
    MetricResult,
    ModelSummary,
    RunError,
)
from evalytic.bench.registry import (
    MODEL_REGISTRY,
    ModelEntry,
    UnknownModelError,
    _SCHEMA_CACHE,
    _USER_OVERRIDES,
    _COST_CACHE,
    detect_cost,
    detect_image_field,
    estimate_cost,
    list_models,
    register_model,
    resolve_model,
)
from evalytic.bench.cost import (
    build_cost_breakdown,
    estimate_run_cost,
    format_cost,
)
from evalytic.bench.runner import (
    _auto_detect_dimensions,
    _build_generation_items,
    _build_items,
    _detect_pipeline,
    _parse_pre_images,
)
from evalytic.report.terminal import score_color
from evalytic.report.json_report import report_to_dict


# -----------------------------------------------------------------------
# bench/types.py
# -----------------------------------------------------------------------


class TestDimensionResult:
    def test_creation(self) -> None:
        dr = DimensionResult(dimension="visual_quality", score=4.2, explanation="Good")
        assert dr.dimension == "visual_quality"
        assert dr.score == 4.2

    def test_to_dict(self) -> None:
        dr = DimensionResult(dimension="visual_quality", score=4.0, explanation="ok", evidence=["a"])
        d = dr.to_dict()
        assert d["score"] == 4.0
        assert "human_score" not in d  # omitted when None

    def test_to_dict_with_human_score(self) -> None:
        dr = DimensionResult(dimension="visual_quality", score=4.0, human_score=3.0, human_notes="bad shadow")
        d = dr.to_dict()
        assert d["human_score"] == 3.0
        assert d["human_notes"] == "bad shadow"

    def test_confidence_default(self) -> None:
        dr = DimensionResult(dimension="visual_quality", score=4.0)
        assert dr.confidence == 1.0

    def test_confidence_custom(self) -> None:
        dr = DimensionResult(dimension="visual_quality", score=4.0, confidence=0.85)
        assert dr.confidence == 0.85

    def test_confidence_in_to_dict(self) -> None:
        dr = DimensionResult(dimension="visual_quality", score=4.0, confidence=0.72)
        d = dr.to_dict()
        assert d["confidence"] == 0.72


class TestDimensionResultConsensus:
    def test_new_fields_default_empty(self) -> None:
        """New consensus fields default to empty (backward compat)."""
        dr = DimensionResult(dimension="visual_quality", score=4.0)
        assert dr.judge_scores == {}
        assert dr.agreement == ""

    def test_to_dict_omits_empty_consensus_fields(self) -> None:
        """Empty consensus fields are not in to_dict output."""
        dr = DimensionResult(dimension="visual_quality", score=4.0, explanation="ok")
        d = dr.to_dict()
        assert "judge_scores" not in d
        assert "agreement" not in d

    def test_to_dict_includes_consensus_fields(self) -> None:
        """Non-empty consensus fields appear in to_dict output."""
        dr = DimensionResult(
            dimension="visual_quality", score=4.1,
            judge_scores={"gemini-3-flash": 4.0, "gpt-4o-mini": 4.2},
            agreement="high",
        )
        d = dr.to_dict()
        assert d["judge_scores"] == {"gemini-3-flash": 4.0, "gpt-4o-mini": 4.2}
        assert d["agreement"] == "high"


class TestImageResult:
    def test_overall_score(self) -> None:
        ir = ImageResult(
            model="flux-schnell",
            image_url="https://example.com/img.jpg",
            scores=[
                DimensionResult(dimension="visual_quality", score=4.0),
                DimensionResult(dimension="prompt_adherence", score=3.0),
            ],
        )
        assert ir.overall_score == 3.5

    def test_overall_score_empty(self) -> None:
        ir = ImageResult(model="flux-schnell", image_url="")
        assert ir.overall_score == 0.0

    def test_to_dict(self) -> None:
        ir = ImageResult(
            model="flux-schnell",
            image_url="https://example.com/img.jpg",
            scores=[DimensionResult(dimension="visual_quality", score=4.0)],
        )
        d = ir.to_dict()
        assert "visual_quality" in d["scores"]
        assert d["overall_score"] == 4.0


class TestBenchReport:
    def _make_report(self) -> BenchReport:
        return BenchReport(
            name="test-bench",
            models=["flux-schnell", "flux-pro"],
            judge="gemini-2.0-flash",
            pipeline="text2img",
            dimensions=["visual_quality", "prompt_adherence"],
            items=[
                BenchItem(
                    item_id="item-001",
                    prompt="A cat",
                    results={
                        "flux-schnell": ImageResult(
                            model="flux-schnell",
                            image_url="https://example.com/1.jpg",
                            scores=[DimensionResult(dimension="visual_quality", score=3.5)],
                        ),
                    },
                )
            ],
            summary={
                "flux-schnell": ModelSummary(
                    model="flux-schnell",
                    overall_score=3.5,
                    dimension_averages={"visual_quality": 3.5},
                ),
                "flux-pro": ModelSummary(
                    model="flux-pro",
                    overall_score=4.5,
                    dimension_averages={"visual_quality": 4.5},
                ),
            },
            winner="flux-pro",
            ranking=[("flux-pro", 4.5), ("flux-schnell", 3.5)],
            cost=CostBreakdown(generation_total_usd=0.05, total_usd=0.05, judge_provider="gemini-2.0-flash"),
            created_at="2026-02-25T14:00:00Z",
            config={"models": ["flux-schnell", "flux-pro"]},
            metadata={"evalytic_version": "0.2.0"},
        )

    def test_to_dict(self) -> None:
        report = self._make_report()
        d = report.to_dict()
        assert d["$schema"] == "https://evalytic.ai/schemas/bench-report-v1.json"
        assert d["name"] == "test-bench"
        assert d["winner"] == "flux-pro"
        assert len(d["ranking"]) == 2
        assert d["ranking"][0] == ["flux-pro", 4.5]

    def test_to_json(self) -> None:
        report = self._make_report()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        report.to_json(path)
        data = json.loads(Path(path).read_text())
        assert data["version"] == "1.0"
        assert data["name"] == "test-bench"

    def test_open_review_calls_server(self) -> None:
        from unittest.mock import patch

        report = self._make_report()
        with patch("evalytic.report.review_server.ReviewServer") as MockServer:
            instance = MockServer.return_value
            instance.get_report.return_value = report
            report.open_review(port=9999)
            MockServer.assert_called_once_with(report, port=9999)


# -----------------------------------------------------------------------
# bench/registry.py
# -----------------------------------------------------------------------


class TestModelSummaryConfidence:
    def test_confidence_defaults(self) -> None:
        ms = ModelSummary(model="flux-schnell", overall_score=4.0)
        assert ms.avg_confidence == 1.0
        assert ms.dimension_confidence == {}

    def test_confidence_fields(self) -> None:
        ms = ModelSummary(
            model="flux-schnell",
            overall_score=4.0,
            dimension_confidence={"visual_quality": 0.85, "prompt_adherence": 0.90},
            avg_confidence=0.875,
        )
        assert ms.avg_confidence == 0.875
        assert ms.dimension_confidence["visual_quality"] == 0.85

    def test_confidence_in_to_dict(self) -> None:
        ms = ModelSummary(
            model="flux-schnell",
            overall_score=4.0,
            dimension_confidence={"visual_quality": 0.85},
            avg_confidence=0.85,
        )
        d = ms.to_dict()
        assert d["avg_confidence"] == 0.85
        assert d["dimension_confidence"] == {"visual_quality": 0.85}


class TestRegistry:
    def test_resolve_known_model(self) -> None:
        entry = resolve_model("flux-schnell")
        assert entry.endpoint == "fal-ai/flux/schnell"
        assert entry.pipeline == "text2img"

    def test_resolve_custom_endpoint(self) -> None:
        entry = resolve_model("fal-ai/custom/model")
        assert entry.endpoint == "fal-ai/custom/model"
        assert entry.short_name == "fal-ai/custom/model"

    def test_unknown_model_error_with_suggestion(self) -> None:
        with pytest.raises(UnknownModelError) as exc_info:
            resolve_model("flux-schnall")
        assert exc_info.value.suggestion == "flux-schnell"
        assert "Did you mean" in str(exc_info.value)

    def test_unknown_model_no_suggestion(self) -> None:
        with pytest.raises(UnknownModelError) as exc_info:
            resolve_model("zzz-nonexistent")
        assert exc_info.value.suggestion is None

    def test_list_models_all(self) -> None:
        all_models = list_models()
        assert len(all_models) == len(MODEL_REGISTRY)

    def test_list_models_filter(self) -> None:
        t2i = list_models(pipeline="text2img")
        assert all(e.pipeline == "text2img" for e in t2i)
        assert len(t2i) == 17

        i2i = list_models(pipeline="img2img")
        assert len(i2i) == 7

    def test_estimate_cost(self) -> None:
        cost = estimate_cost("flux-schnell", 10)
        assert cost == pytest.approx(0.03)


# -----------------------------------------------------------------------
# bench/registry.py -- schema detection
# -----------------------------------------------------------------------


class TestSchemaDetection:
    def test_detect_image_field_cached(self) -> None:
        """Cache hit returns cached value."""
        _SCHEMA_CACHE["fal-ai/test-cached"] = "image_urls"
        try:
            assert detect_image_field("fal-ai/test-cached") == "image_urls"
        finally:
            _SCHEMA_CACHE.pop("fal-ai/test-cached", None)

    def test_detect_image_field_fallback_default(self) -> None:
        """API failure falls back to 'image_url' when no registry default."""
        _SCHEMA_CACHE.pop("nonexistent/model", None)
        assert detect_image_field("nonexistent/model") == "image_url"

    def test_detect_image_field_fallback_registry(self) -> None:
        """API failure falls back to registry_default when provided."""
        _SCHEMA_CACHE.pop("nonexistent/model2", None)
        result = detect_image_field("nonexistent/model2", registry_default="image_urls")
        assert result == "image_urls"

    def test_build_gen_items_image_urls(self) -> None:
        """image_urls field should be sent as array."""
        entry = ModelEntry("test", "fal-ai/test", "img2img", 0.03, image_field="image_urls")
        items = [{"item_id": "item-001", "image_url": "https://example.com/img.jpg", "instruction": "edit it"}]
        result = _build_generation_items(items, entry, "img2img", None, None, None)
        assert result[0]["arguments"]["image_urls"] == ["https://example.com/img.jpg"]
        assert "image_url" not in result[0]["arguments"]

    def test_build_gen_items_image_url(self) -> None:
        """Default image_url field should be sent as string."""
        entry = ModelEntry("test", "fal-ai/test", "img2img", 0.03)
        items = [{"item_id": "item-001", "image_url": "https://example.com/img.jpg", "instruction": "edit it"}]
        result = _build_generation_items(items, entry, "img2img", None, None, None)
        assert result[0]["arguments"]["image_url"] == "https://example.com/img.jpg"
        assert "image_urls" not in result[0]["arguments"]


# -----------------------------------------------------------------------
# bench/registry.py -- cost detection
# -----------------------------------------------------------------------


class TestCostDetection:
    def test_detect_cost_cached(self) -> None:
        """Cache hit returns cached value."""
        _COST_CACHE["fal-ai/test-cost-cached"] = 0.042
        try:
            assert detect_cost("fal-ai/test-cost-cached") == 0.042
        finally:
            _COST_CACHE.pop("fal-ai/test-cost-cached", None)

    def test_detect_cost_cached_none(self) -> None:
        """Cache stores None for previously failed lookups."""
        _COST_CACHE["fal-ai/test-cost-none"] = None
        try:
            assert detect_cost("fal-ai/test-cost-none") is None
        finally:
            _COST_CACHE.pop("fal-ai/test-cost-none", None)

    def test_detect_cost_fallback_default(self) -> None:
        """Scrape failure falls back to registry_default."""
        _COST_CACHE.pop("nonexistent/cost-model", None)
        result = detect_cost("nonexistent/cost-model", registry_default=0.05)
        assert result == 0.05

    def test_detect_cost_fallback_none(self) -> None:
        """Scrape failure with no default returns None."""
        _COST_CACHE.pop("nonexistent/cost-model2", None)
        result = detect_cost("nonexistent/cost-model2")
        assert result is None

    def test_register_model_overrides_cost(self) -> None:
        """register_model cost takes priority over any auto-detect."""
        register_model("test-cost-override", endpoint="fal-ai/test-co", cost_per_image=0.099)
        try:
            entry = resolve_model("test-cost-override")
            assert entry.cost_per_image == 0.099
            assert "test-cost-override" in _USER_OVERRIDES
        finally:
            MODEL_REGISTRY.pop("test-cost-override", None)
            _USER_OVERRIDES.discard("test-cost-override")


# -----------------------------------------------------------------------
# register_model()
# -----------------------------------------------------------------------


class TestRegisterModel:
    def test_register_new_model(self) -> None:
        """Register a brand-new model with all fields."""
        try:
            entry = register_model(
                "test-custom",
                endpoint="fal-ai/test-custom/v1",
                pipeline="img2img",
                image_field="image_urls",
                cost_per_image=0.04,
            )
            assert entry.short_name == "test-custom"
            assert entry.endpoint == "fal-ai/test-custom/v1"
            assert entry.pipeline == "img2img"
            assert entry.image_field == "image_urls"
            assert entry.cost_per_image == 0.04
            assert MODEL_REGISTRY["test-custom"] is entry
        finally:
            MODEL_REGISTRY.pop("test-custom", None)

    def test_override_existing_field(self) -> None:
        """Override a single field of an existing model."""
        original = MODEL_REGISTRY["seedream-edit"]
        try:
            entry = register_model("seedream-edit", image_field="image_urls")
            assert entry.image_field == "image_urls"
            assert entry.endpoint == original.endpoint
            assert entry.pipeline == original.pipeline
            assert entry.cost_per_image == original.cost_per_image
        finally:
            MODEL_REGISTRY["seedream-edit"] = original
            _USER_OVERRIDES.discard("seedream-edit")

    def test_override_multiple_fields(self) -> None:
        """Override pipeline and cost together."""
        original = MODEL_REGISTRY.get("flux-schnell")
        try:
            entry = register_model(
                "flux-schnell", pipeline="img2img", cost_per_image=0.10
            )
            assert entry.pipeline == "img2img"
            assert entry.cost_per_image == 0.10
            assert entry.endpoint == original.endpoint
        finally:
            MODEL_REGISTRY["flux-schnell"] = original
            _USER_OVERRIDES.discard("flux-schnell")

    def test_resolve_finds_registered_by_endpoint(self) -> None:
        """resolve_model with a '/' path finds registered models by endpoint."""
        try:
            register_model(
                "my-edit",
                endpoint="fal-ai/my-custom/edit",
                pipeline="img2img",
                image_field="image_urls",
            )
            entry = resolve_model("fal-ai/my-custom/edit")
            assert entry.short_name == "my-edit"
            assert entry.image_field == "image_urls"
        finally:
            MODEL_REGISTRY.pop("my-edit", None)
            _USER_OVERRIDES.discard("my-edit")

    def test_user_override_tracked(self) -> None:
        """register_model() marks model as user-overridden."""
        try:
            register_model("test-tracked", endpoint="fal-ai/test", pipeline="img2img")
            assert "test-tracked" in _USER_OVERRIDES
        finally:
            MODEL_REGISTRY.pop("test-tracked", None)
            _USER_OVERRIDES.discard("test-tracked")

    def test_builtin_not_in_overrides(self) -> None:
        """Built-in registry models are not in _USER_OVERRIDES."""
        assert "flux-schnell" not in _USER_OVERRIDES

    def test_register_via_public_api(self) -> None:
        """register_model is importable from top-level evalytic."""
        import evalytic

        assert hasattr(evalytic, "register_model")
        assert evalytic.register_model is register_model


# -----------------------------------------------------------------------
# bench/cost.py
# -----------------------------------------------------------------------


class TestCostBreakdownConsensus:
    def test_new_fields_default_empty(self) -> None:
        """New consensus cost fields default to empty."""
        cb = CostBreakdown()
        assert cb.judge_providers == []
        assert cb.judge_cost_by_provider == {}

    def test_to_dict_omits_empty_consensus_fields(self) -> None:
        """Empty consensus fields not in to_dict."""
        cb = CostBreakdown(judge_provider="gemini-2.5-flash")
        d = cb.to_dict()
        assert "providers" not in d["judge"]
        assert "cost_by_provider" not in d["judge"]

    def test_to_dict_includes_consensus_fields(self) -> None:
        """Consensus fields present in to_dict when populated."""
        cb = CostBreakdown(
            judge_provider="consensus:j1,j2",
            judge_providers=["j1", "j2"],
            judge_cost_by_provider={"j1": 0.01, "j2": 0.02},
        )
        d = cb.to_dict()
        assert d["judge"]["providers"] == ["j1", "j2"]
        assert d["judge"]["cost_by_provider"] == {"j1": 0.01, "j2": 0.02}


class TestBenchReportConsensus:
    def test_judges_and_consensus_mode_defaults(self) -> None:
        """New fields have sensible defaults."""
        report = BenchReport(
            name="test", models=["m1"], judge="gemini-2.5-flash",
            pipeline="text2img", ranking=[("m1", 4.0)], winner="m1",
        )
        assert report.judges == []
        assert report.consensus_mode is False

    def test_consensus_mode_in_json(self) -> None:
        """consensus_mode appears in JSON when True."""
        from evalytic.report.json_report import report_to_dict

        report = BenchReport(
            name="test", models=["m1"], judge="gemini-3-flash",
            pipeline="text2img", ranking=[("m1", 4.0)], winner="m1",
            judges=["gemini-3-flash", "gpt-4o-mini"],
            consensus_mode=True,
        )
        d = report_to_dict(report)
        assert d["consensus_mode"] is True
        assert d["judges"] == ["gemini-3-flash", "gpt-4o-mini"]

    def test_non_consensus_no_extra_json_fields(self) -> None:
        """Non-consensus report omits consensus fields from JSON."""
        from evalytic.report.json_report import report_to_dict

        report = BenchReport(
            name="test", models=["m1"], judge="gemini-2.5-flash",
            pipeline="text2img", ranking=[("m1", 4.0)], winner="m1",
        )
        d = report_to_dict(report)
        assert "consensus_mode" not in d
        assert "judges" not in d


class TestModelSummaryAgreement:
    def test_agreement_rate_default(self) -> None:
        ms = ModelSummary(model="m1", overall_score=4.0)
        assert ms.agreement_rate == 1.0

    def test_agreement_rate_in_to_dict(self) -> None:
        """agreement_rate only in to_dict when < 1.0."""
        ms = ModelSummary(model="m1", overall_score=4.0, agreement_rate=0.75)
        d = ms.to_dict()
        assert d["agreement_rate"] == 0.75

    def test_agreement_rate_omitted_when_1(self) -> None:
        ms = ModelSummary(model="m1", overall_score=4.0, agreement_rate=1.0)
        d = ms.to_dict()
        assert "agreement_rate" not in d


class TestCost:
    def test_estimate_run_cost(self) -> None:
        cb = estimate_run_cost(["flux-schnell", "flux-pro"], 5, "gemini-2.0-flash", 2)
        assert cb.generation_request_count == 10
        assert cb.generation_total_usd > 0
        assert cb.judge_request_count == 20  # 10 images * 2 dimensions

    def test_format_cost_zero(self) -> None:
        assert format_cost(0.0) == "$0.00"

    def test_format_cost_small(self) -> None:
        assert format_cost(0.003) == "$0.0030"

    def test_format_cost_normal(self) -> None:
        assert format_cost(1.50) == "$1.50"

    def test_build_cost_breakdown(self) -> None:
        cb = build_cost_breakdown(
            {"flux-schnell": 0.015, "flux-pro": 0.25},
            10,
            20,
            "gemini-2.0-flash",
        )
        assert cb.generation_total_usd == 0.265
        assert cb.generation_request_count == 10
        assert cb.total_usd >= cb.generation_total_usd


# -----------------------------------------------------------------------
# bench/runner.py (helpers)
# -----------------------------------------------------------------------


class TestPipelineDetection:
    def test_text2img(self) -> None:
        assert _detect_pipeline([{"prompt": "A cat"}], None) == "text2img"

    def test_img2img_via_inputs(self) -> None:
        assert _detect_pipeline(None, [{"image_url": "x.jpg"}]) == "img2img"

    def test_img2img_via_image_url_in_prompts(self) -> None:
        assert _detect_pipeline([{"image_url": "x.jpg", "instruction": "enhance"}], None) == "img2img"

    def test_missing_raises(self) -> None:
        with pytest.raises(ValueError, match="prompts or inputs"):
            _detect_pipeline(None, None)


class TestDimensionAutoDetect:
    def test_text2img_basic(self) -> None:
        dims = _auto_detect_dimensions("text2img", [{"prompt": "A cat"}])
        assert "visual_quality" in dims
        assert "prompt_adherence" in dims
        assert "text_rendering" not in dims

    def test_text2img_with_text_keyword(self) -> None:
        dims = _auto_detect_dimensions("text2img", [{"prompt": "Write the word hello"}])
        assert "text_rendering" in dims

    def test_img2img(self) -> None:
        dims = _auto_detect_dimensions("img2img", [{"image_url": "x.jpg"}])
        assert "visual_quality" in dims
        assert "input_fidelity" in dims
        assert "transformation_quality" in dims
        assert "artifact_detection" in dims


class TestBuildItems:
    def test_text2img(self) -> None:
        items = _build_items([{"prompt": "A cat"}, {"prompt": "A dog"}], None, "text2img")
        assert len(items) == 2
        assert items[0]["prompt"] == "A cat"
        assert items[0]["item_id"] == "item-001"

    def test_img2img(self) -> None:
        items = _build_items(
            None,
            [{"image_url": "x.jpg", "instruction": "enhance"}],
            "img2img",
        )
        assert len(items) == 1
        assert items[0]["image_url"] == "x.jpg"
        assert items[0]["instruction"] == "enhance"


# -----------------------------------------------------------------------
# report/terminal.py
# -----------------------------------------------------------------------


class TestTerminal:
    def test_score_color_green(self) -> None:
        assert score_color(4.5) == "green"
        assert score_color(4.0) == "green"

    def test_score_color_yellow(self) -> None:
        assert score_color(3.5) == "yellow"
        assert score_color(3.0) == "yellow"

    def test_score_color_red(self) -> None:
        assert score_color(2.9) == "red"
        assert score_color(1.0) == "red"


# -----------------------------------------------------------------------
# report/json_report.py
# -----------------------------------------------------------------------


class TestJsonReport:
    def test_schema_keys(self) -> None:
        report = BenchReport(
            name="test",
            models=["m1"],
            judge="gemini-2.0-flash",
            pipeline="text2img",
            ranking=[("m1", 4.0)],
            winner="m1",
        )
        d = report_to_dict(report)
        assert "$schema" in d
        assert "version" in d
        assert "items" in d
        assert "summary" in d
        assert "ranking" in d
        assert "cost" in d

    def test_ranking_format(self) -> None:
        report = BenchReport(
            name="test",
            models=["m1", "m2"],
            judge="gemini-2.0-flash",
            pipeline="text2img",
            ranking=[("m1", 4.5), ("m2", 3.5)],
            winner="m1",
        )
        d = report_to_dict(report)
        assert d["ranking"] == [["m1", 4.5], ["m2", 3.5]]


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------


# -----------------------------------------------------------------------
# bench/runner.py -- run_bench integration (mock generator + judge)
# -----------------------------------------------------------------------


class TestRunBench:
    def test_dimension_validation_in_runner(self) -> None:
        from evalytic.bench.runner import run_bench

        with pytest.raises(ValueError, match="Unknown dimensions"):
            run_bench(
                models=["flux-schnell"],
                prompts=[{"prompt": "A cat"}],
                dimensions=["invalid_dim"],
            )

    def test_run_bench_with_mocks(self) -> None:
        """Full run_bench with mocked generator and judge."""
        from unittest.mock import MagicMock, patch

        from evalytic.bench.generator import GenerationResult
        from evalytic.bench.runner import run_bench
        from evalytic.bench.types import DimensionResult

        mock_gen_result = GenerationResult(
            image_url="https://example.com/img.jpg",
            generation_time_ms=500,
            generation_cost_usd=0.003,
            model="flux-schnell",
            item_id="item-001",
            status="success",
            local_path="",
        )

        mock_dim_results = [
            DimensionResult(dimension="visual_quality", score=4.0, explanation="Good"),
            DimensionResult(dimension="prompt_adherence", score=3.5, explanation="Decent"),
        ]

        with patch("evalytic.bench.runner.generate_single", return_value=mock_gen_result), \
             patch("evalytic.bench.runner.Judge") as MockJudge:
            mock_judge = MockJudge.return_value
            mock_judge.score.return_value = mock_dim_results
            mock_judge.close = MagicMock()

            report = run_bench(
                models=["flux-schnell"],
                prompts=[{"prompt": "A cat"}],
                dimensions=["visual_quality", "prompt_adherence"],
            )

        assert report.pipeline == "text2img"
        assert len(report.items) == 1
        assert "flux-schnell" in report.summary
        assert report.summary["flux-schnell"].overall_score > 0
        assert report.winner == "flux-schnell"

    def test_run_bench_failed_generation(self) -> None:
        """Failed generation should produce partial report without crash."""
        from unittest.mock import MagicMock, patch

        from evalytic.bench.generator import GenerationResult
        from evalytic.bench.runner import run_bench

        mock_gen_result = GenerationResult(
            model="flux-schnell",
            item_id="item-001",
            status="failed",
            error="API timeout",
        )

        with patch("evalytic.bench.runner.generate_single", return_value=mock_gen_result), \
             patch("evalytic.bench.runner.Judge") as MockJudge:
            mock_judge = MockJudge.return_value
            mock_judge.close = MagicMock()

            report = run_bench(
                models=["flux-schnell"],
                prompts=[{"prompt": "A cat"}],
                dimensions=["visual_quality"],
            )

        assert len(report.items) == 1
        item = report.items[0]
        assert item.results["flux-schnell"].status == "failed"
        assert report.summary["flux-schnell"].overall_score == 0.0


# -----------------------------------------------------------------------
# Pre-images mode (--images)
# -----------------------------------------------------------------------


class TestParsePreImages:
    def test_basic_text2img(self) -> None:
        raw = [
            {
                "prompt": "A cat",
                "images": {"model-a": "https://example.com/a.png", "model-b": "https://example.com/b.png"},
            },
        ]
        items, model_names, pipeline = _parse_pre_images(raw)
        assert len(items) == 1
        assert model_names == ["model-a", "model-b"]
        assert pipeline == "text2img"
        assert items[0]["prompt"] == "A cat"
        assert items[0]["_images"]["model-a"] == "https://example.com/a.png"

    def test_img2img_with_input_image(self) -> None:
        raw = [
            {
                "prompt": "Place on counter",
                "input_image": "https://example.com/product.jpg",
                "images": {"model-a": "out-a.png"},
            },
        ]
        items, model_names, pipeline = _parse_pre_images(raw)
        assert pipeline == "img2img"
        assert items[0]["image_url"] == "https://example.com/product.jpg"
        assert items[0]["instruction"] == "Place on counter"

    def test_multiple_items_collects_all_models(self) -> None:
        raw = [
            {"prompt": "P1", "images": {"m1": "a.png", "m2": "b.png"}},
            {"prompt": "P2", "images": {"m2": "c.png", "m3": "d.png"}},
        ]
        items, model_names, pipeline = _parse_pre_images(raw)
        assert len(items) == 2
        assert model_names == ["m1", "m2", "m3"]  # ordered, deduplicated

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            _parse_pre_images([])

    def test_missing_images_key_raises(self) -> None:
        with pytest.raises(ValueError, match="images.*required"):
            _parse_pre_images([{"prompt": "A cat"}])

    def test_item_id_generation(self) -> None:
        raw = [
            {"prompt": "P1", "images": {"m1": "a.png"}},
            {"prompt": "P2", "images": {"m1": "b.png"}},
        ]
        items, _, _ = _parse_pre_images(raw)
        assert items[0]["item_id"] == "item-001"
        assert items[1]["item_id"] == "item-002"

    def test_custom_item_id(self) -> None:
        raw = [{"prompt": "P1", "item_id": "my-custom-id", "images": {"m1": "a.png"}}]
        items, _, _ = _parse_pre_images(raw)
        assert items[0]["item_id"] == "my-custom-id"

    def test_tags_preserved(self) -> None:
        raw = [{"prompt": "P1", "tags": ["portrait", "hd"], "images": {"m1": "a.png"}}]
        items, _, _ = _parse_pre_images(raw)
        assert items[0]["tags"] == ["portrait", "hd"]


class TestPreImagesBench:
    def test_run_bench_with_pre_images(self) -> None:
        """Full run_bench with pre_images and mocked judge."""
        from unittest.mock import MagicMock, patch

        from evalytic.bench.runner import run_bench
        from evalytic.bench.types import DimensionResult

        mock_dim_results = [
            DimensionResult(dimension="visual_quality", score=4.5, explanation="Great"),
            DimensionResult(dimension="prompt_adherence", score=4.0, explanation="Good"),
        ]

        pre_images = [
            {
                "prompt": "A cat wearing a top hat",
                "images": {
                    "midjourney": "https://example.com/mj.png",
                    "dalle3": "https://example.com/dalle.png",
                },
            },
        ]

        with patch("evalytic.bench.runner.Judge") as MockJudge:
            mock_judge = MockJudge.return_value
            mock_judge.score.return_value = mock_dim_results
            mock_judge.close = MagicMock()

            report = run_bench(
                pre_images=pre_images,
                dimensions=["visual_quality", "prompt_adherence"],
            )

        assert report.pipeline == "text2img"
        assert len(report.items) == 1
        assert set(report.models) == {"midjourney", "dalle3"}
        assert "midjourney" in report.summary
        assert "dalle3" in report.summary
        assert report.summary["midjourney"].overall_score > 0
        assert report.winner in ("midjourney", "dalle3")

    def test_run_bench_pre_images_no_generation(self) -> None:
        """Ensure generate_single is never called in pre_images mode."""
        from unittest.mock import MagicMock, patch

        from evalytic.bench.runner import run_bench

        mock_dim_results = [
            DimensionResult(dimension="visual_quality", score=3.0, explanation="Ok"),
        ]

        pre_images = [
            {"prompt": "Test", "images": {"m1": "img.png"}},
        ]

        with patch("evalytic.bench.runner.generate_single") as mock_gen, \
             patch("evalytic.bench.runner.Judge") as MockJudge:
            mock_judge = MockJudge.return_value
            mock_judge.score.return_value = mock_dim_results
            mock_judge.close = MagicMock()

            run_bench(
                pre_images=pre_images,
                dimensions=["visual_quality"],
            )

        mock_gen.assert_not_called()

    def test_pre_images_image_urls_passed_to_judge(self) -> None:
        """Verify the correct image URL is passed to the judge for scoring."""
        from unittest.mock import MagicMock, patch

        from evalytic.bench.runner import run_bench

        mock_dim_results = [
            DimensionResult(dimension="visual_quality", score=4.0, explanation="Good"),
        ]

        pre_images = [
            {"prompt": "A dog", "images": {"m1": "https://example.com/dog.png"}},
        ]

        with patch("evalytic.bench.runner.Judge") as MockJudge:
            mock_judge = MockJudge.return_value
            mock_judge.score.return_value = mock_dim_results
            mock_judge.close = MagicMock()

            run_bench(
                pre_images=pre_images,
                dimensions=["visual_quality"],
            )

        # Judge.score should have been called with the pre-existing image URL
        call_args = mock_judge.score.call_args
        assert call_args.kwargs["image_url"] == "https://example.com/dog.png"


class TestImagesCliParsing:
    def test_parse_images_file_object_format(self, tmp_path: Path) -> None:
        from evalytic.cli.bench import _parse_images_file

        data = {
            "judge": "openai/gpt-4o",
            "dimensions": ["visual_quality"],
            "items": [
                {"prompt": "A cat", "images": {"m1": "img.png"}},
            ],
        }
        f = tmp_path / "images.json"
        f.write_text(json.dumps(data))

        items, config = _parse_images_file(str(f))
        assert len(items) == 1
        assert config["judge"] == "openai/gpt-4o"
        assert config["dimensions"] == ["visual_quality"]

    def test_parse_images_file_list_format(self, tmp_path: Path) -> None:
        from evalytic.cli.bench import _parse_images_file

        data = [
            {"prompt": "A cat", "images": {"m1": "img.png"}},
            {"prompt": "A dog", "images": {"m2": "img2.png"}},
        ]
        f = tmp_path / "images.json"
        f.write_text(json.dumps(data))

        items, config = _parse_images_file(str(f))
        assert len(items) == 2
        assert config == {}

    def test_parse_images_file_name_override(self, tmp_path: Path) -> None:
        from evalytic.cli.bench import _parse_images_file

        data = {
            "name": "my-bench",
            "items": [{"prompt": "X", "images": {"m1": "x.png"}}],
        }
        f = tmp_path / "images.json"
        f.write_text(json.dumps(data))

        _, config = _parse_images_file(str(f))
        assert config["name"] == "my-bench"


# -----------------------------------------------------------------------
# scorers/__init__.py
# -----------------------------------------------------------------------


class TestScorers:
    def test_all_dimensions_are_implemented(self) -> None:
        from evalytic.scorers import ALL_DIMENSIONS
        from evalytic.bench.judge import DIMENSION_CONFIG

        # Every exported dimension must have a judge config
        for dim in ALL_DIMENSIONS:
            assert dim in DIMENSION_CONFIG, f"{dim} exported but not implemented in judge"

    def test_no_unimplemented_exports(self) -> None:
        from evalytic.scorers import ALL_DIMENSIONS

        assert len(ALL_DIMENSIONS) == 7
        assert "anatomical_correctness" not in ALL_DIMENSIONS
        assert "temporal_coherence" not in ALL_DIMENSIONS


# -----------------------------------------------------------------------
# config.py
# -----------------------------------------------------------------------


class TestConfig:
    def test_load_config_no_file(self) -> None:
        from evalytic.config import load_config

        config = load_config()
        # May return empty dict if no evalytic.toml in cwd
        assert isinstance(config, dict)

    def test_load_config_missing_explicit_path(self) -> None:
        from evalytic.config import load_config
        from evalytic.exceptions import ConfigError

        with pytest.raises(ConfigError):
            load_config("/nonexistent/evalytic.toml")

    def test_load_config_from_file(self, tmp_path: Path) -> None:
        from evalytic.config import load_config

        toml_file = tmp_path / "evalytic.toml"
        toml_file.write_text('[bench]\njudge = "gemini-2.0-flash"\nconcurrency = 8\n')

        config = load_config(str(toml_file))
        assert config["bench"]["judge"] == "gemini-2.0-flash"
        assert config["bench"]["concurrency"] == 8

    def test_apply_keys_sets_env(self, tmp_path: Path) -> None:
        import os
        from evalytic.config import apply_keys

        # Clear env var if set
        old = os.environ.pop("GEMINI_API_KEY", None)
        try:
            apply_keys({"keys": {"gemini": "test-key-123"}})
            assert os.environ["GEMINI_API_KEY"] == "test-key-123"
        finally:
            if old is not None:
                os.environ["GEMINI_API_KEY"] = old
            else:
                os.environ.pop("GEMINI_API_KEY", None)

    def test_apply_keys_does_not_override(self) -> None:
        import os
        from evalytic.config import apply_keys

        os.environ["GEMINI_API_KEY"] = "existing-key"
        try:
            apply_keys({"keys": {"gemini": "new-key"}})
            assert os.environ["GEMINI_API_KEY"] == "existing-key"
        finally:
            os.environ.pop("GEMINI_API_KEY", None)


# -----------------------------------------------------------------------
# Sharpness scorer (no torch required)
# -----------------------------------------------------------------------


class TestSharpnessScorer:
    def test_score_returns_float_in_range(self, tmp_path: Path) -> None:
        """SharpnessScorer returns a float in [0, 1]."""
        from PIL import Image
        from evalytic.bench.metrics import SharpnessScorer

        # Create a sharp test image with edges
        img = Image.new("RGB", (256, 256), "white")
        pixels = img.load()
        for x in range(128):
            for y in range(256):
                pixels[x, y] = (0, 0, 0)
        path = str(tmp_path / "sharp.png")
        img.save(path)

        scorer = SharpnessScorer()
        val = scorer.score(path)
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0

    def test_sharp_vs_blurry(self, tmp_path: Path) -> None:
        """Sharp image scores higher than blurry image."""
        from PIL import Image, ImageFilter
        from evalytic.bench.metrics import SharpnessScorer

        # Sharp: high-frequency checkerboard
        img = Image.new("RGB", (256, 256))
        pixels = img.load()
        for x in range(256):
            for y in range(256):
                c = 255 if (x // 4 + y // 4) % 2 == 0 else 0
                pixels[x, y] = (c, c, c)
        sharp_path = str(tmp_path / "sharp.png")
        img.save(sharp_path)

        # Blurry: heavily gaussian-blurred version
        blurry = img.filter(ImageFilter.GaussianBlur(radius=10))
        blurry_path = str(tmp_path / "blurry.png")
        blurry.save(blurry_path)

        scorer = SharpnessScorer()
        sharp_score = scorer.score(sharp_path)
        blurry_score = scorer.score(blurry_path)
        assert sharp_score > blurry_score, f"sharp={sharp_score} should be > blurry={blurry_score}"

    def test_uniform_image_low_score(self, tmp_path: Path) -> None:
        """Uniform (no edges) image gets low sharpness score."""
        from PIL import Image
        from evalytic.bench.metrics import SharpnessScorer

        img = Image.new("RGB", (256, 256), (128, 128, 128))
        path = str(tmp_path / "uniform.png")
        img.save(path)

        scorer = SharpnessScorer()
        val = scorer.score(path)
        assert val < 0.2, f"Uniform image should have low sharpness, got {val}"

    def test_compute_metrics_sharpness(self, tmp_path: Path) -> None:
        """compute_metrics includes sharpness without METRICS_AVAILABLE."""
        from PIL import Image
        from evalytic.bench.metrics import compute_metrics
        from evalytic.bench.types import BenchItem, ImageResult

        img = Image.new("RGB", (64, 64), "red")
        img_path = str(tmp_path / "test.png")
        img.save(img_path)

        item = BenchItem(
            item_id="t1",
            prompt="test",
            results={
                "model-a": ImageResult(
                    model="model-a",
                    image_url="http://example.com/img.png",
                    image_local=img_path,
                    status="success",
                ),
            },
        )
        compute_metrics([item], ["sharpness"], "text2img", [{"item_id": "t1", "prompt": "test"}])
        metrics = item.results["model-a"].metrics
        assert len(metrics) == 1
        assert metrics[0].metric == "sharpness"
        assert 0.0 <= metrics[0].value <= 1.0


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------


class TestPublicApi:
    def test_bench_importable(self) -> None:
        import evalytic

        assert callable(evalytic.bench)
        assert evalytic.__version__ == "0.3.2"


# -----------------------------------------------------------------------
# CLI: Quick Start (bare evalytic)
# -----------------------------------------------------------------------


class TestBareEvalytic:
    def test_quick_start_output(self) -> None:
        from click.testing import CliRunner
        from evalytic.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, [])
        assert result.exit_code == 0
        assert "Quick Start" in result.output
        assert "evalytic bench" in result.output
        assert "FAL_KEY" in result.output
        assert "GEMINI_API_KEY" in result.output
        assert "evalytic.ai" in result.output

    def test_help_still_works(self) -> None:
        from click.testing import CliRunner
        from evalytic.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "bench" in result.output
        assert "eval" in result.output


# -----------------------------------------------------------------------
# CLI: Grouped bench --help
# -----------------------------------------------------------------------


class TestBenchGroupedHelp:
    def test_essential_options_section(self) -> None:
        from click.testing import CliRunner
        from evalytic.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["bench", "--help"])
        assert result.exit_code == 0
        assert "Essential Options" in result.output
        assert "Advanced Options" in result.output

    def test_essential_contains_key_options(self) -> None:
        from click.testing import CliRunner
        from evalytic.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["bench", "--help"])
        output = result.output
        # --models should be in Essential
        essential_start = output.index("Essential Options")
        advanced_start = output.index("Advanced Options")
        essential_section = output[essential_start:advanced_start]
        assert "--models" in essential_section
        assert "--prompts" in essential_section
        assert "--yes" in essential_section
        assert "--review" in essential_section

    def test_advanced_contains_detail_options(self) -> None:
        from click.testing import CliRunner
        from evalytic.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["bench", "--help"])
        output = result.output
        advanced_start = output.index("Advanced Options")
        advanced_section = output[advanced_start:]
        assert "--concurrency" in advanced_section
        assert "--seed" in advanced_section
        assert "--timeout" in advanced_section

    def test_output_dir_in_essential_help(self) -> None:
        from click.testing import CliRunner
        from evalytic.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["bench", "--help"])
        output = result.output
        essential_start = output.index("Essential Options")
        advanced_start = output.index("Advanced Options")
        essential_section = output[essential_start:advanced_start]
        assert "--output-dir" in essential_section


# ── RunError ─────────────────────────────────────────────────────────


class TestRunError:
    """Tests for the RunError dataclass."""

    def test_create_and_to_dict(self) -> None:
        err = RunError(
            timestamp="2026-02-27T14:30:01+00:00",
            level="error",
            phase="generation",
            model="flux-schnell",
            item_id="item-001",
            dimension="",
            message="content_policy_violation: Image rejected",
        )
        d = err.to_dict()
        assert d["level"] == "error"
        assert d["phase"] == "generation"
        assert d["model"] == "flux-schnell"
        assert d["item_id"] == "item-001"
        assert d["dimension"] == ""
        assert "content_policy_violation" in d["message"]

    def test_bench_report_errors_default_empty(self) -> None:
        report = BenchReport(
            name="test",
            models=["m1"],
            judge="gemini-2.5-flash",
            pipeline="text2img",
        )
        assert report.errors == []

    def test_bench_report_errors_in_json(self) -> None:
        err = RunError(
            timestamp="2026-02-27T14:30:01+00:00",
            level="error",
            phase="scoring",
            model="flux-pro",
            item_id="item-002",
            dimension="visual_quality",
            message="Judge failed after 3 retries",
        )
        report = BenchReport(
            name="test",
            models=["flux-pro"],
            judge="gemini-2.5-flash",
            pipeline="text2img",
            errors=[err],
        )
        d = report_to_dict(report)
        assert "errors" in d
        assert len(d["errors"]) == 1
        assert d["errors"][0]["phase"] == "scoring"

    def test_bench_report_no_errors_in_json(self) -> None:
        report = BenchReport(
            name="test",
            models=["m1"],
            judge="gemini-2.5-flash",
            pipeline="text2img",
        )
        d = report_to_dict(report)
        assert "errors" not in d
