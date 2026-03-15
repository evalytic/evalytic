"""Tests for Phase 10: metrics (CLIP/LPIPS) + browser review.

All tests are offline -- zero network calls, zero GPU required.
"""

from __future__ import annotations

import io
from unittest.mock import patch

import pytest

from evalytic.bench.metrics import (
    METRICS_AVAILABLE,
    _approx_t_pvalue,
    _interpret_r,
    compute_correlation,
)
from evalytic.bench.types import (
    BenchItem,
    BenchReport,
    CorrelationStat,
    CostBreakdown,
    DimensionResult,
    ImageResult,
    MetricFlag,
    MetricResult,
    MetricScoringConfig,
    ModelSummary,
    ScoringConfig,
)
from evalytic.report.terminal import print_comparison_table, print_correlation, print_full_report


# -----------------------------------------------------------------------
# metrics.py — graceful fallback
# -----------------------------------------------------------------------


class TestMetricsAvailability:
    def test_metrics_available_is_bool(self) -> None:
        assert isinstance(METRICS_AVAILABLE, bool)

    def test_clip_scorer_importable(self) -> None:
        from evalytic.bench.metrics import CLIPScorer
        assert CLIPScorer is not None

    def test_lpips_scorer_importable(self) -> None:
        from evalytic.bench.metrics import LPIPSScorer
        assert LPIPSScorer is not None

    def test_face_scorer_importable(self) -> None:
        from evalytic.bench.metrics import FaceScorer
        assert FaceScorer is not None

    def test_compute_metrics_importable(self) -> None:
        from evalytic.bench.metrics import compute_metrics
        assert callable(compute_metrics)


# -----------------------------------------------------------------------
# metrics.py — correlation math
# -----------------------------------------------------------------------


def _make_items_with_metrics(
    metric_vals: list[float],
    dim_vals: list[float],
    metric: str = "clip_score",
    dimension: str = "prompt_adherence",
) -> list[BenchItem]:
    """Create bench items with paired metric/dimension values for testing."""
    items: list[BenchItem] = []
    for i, (mv, dv) in enumerate(zip(metric_vals, dim_vals)):
        item = BenchItem(
            item_id=f"item-{i:03d}",
            prompt=f"test prompt {i}",
            results={
                "model-a": ImageResult(
                    model="model-a",
                    image_url=f"https://example.com/{i}.jpg",
                    scores=[DimensionResult(dimension=dimension, score=dv)],
                    metrics=[MetricResult(metric=metric, value=mv)],
                ),
            },
        )
        items.append(item)
    return items


class TestCorrelation:
    def test_perfect_positive(self) -> None:
        items = _make_items_with_metrics(
            [0.5, 0.6, 0.7, 0.8, 0.9],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        )
        corr = compute_correlation(items, "clip_score", "prompt_adherence")
        assert corr is not None
        assert corr.pearson_r == pytest.approx(1.0, abs=0.01)
        assert corr.interpretation == "high_agreement"
        assert corr.metric_pair == "clip_score_vs_prompt_adherence"

    def test_negative_correlation(self) -> None:
        items = _make_items_with_metrics(
            [0.9, 0.8, 0.7, 0.6, 0.5],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        )
        corr = compute_correlation(items, "clip_score", "prompt_adherence")
        assert corr is not None
        assert corr.pearson_r == pytest.approx(-1.0, abs=0.01)
        assert corr.interpretation == "high_agreement"

    def test_too_few_points_returns_none(self) -> None:
        items = _make_items_with_metrics([0.5, 0.6], [3.0, 4.0])
        corr = compute_correlation(items, "clip_score", "prompt_adherence")
        assert corr is None

    def test_no_matching_metric_returns_none(self) -> None:
        items = _make_items_with_metrics([0.5, 0.6, 0.7], [3.0, 4.0, 5.0])
        corr = compute_correlation(items, "nonexistent_metric", "prompt_adherence")
        assert corr is None

    def test_moderate_correlation(self) -> None:
        items = _make_items_with_metrics(
            [0.5, 0.6, 0.7, 0.8, 0.3],
            [3.0, 4.0, 3.5, 4.5, 4.0],
        )
        corr = compute_correlation(items, "clip_score", "prompt_adherence")
        assert corr is not None
        assert -1.0 <= corr.pearson_r <= 1.0
        assert corr.p_value >= 0.0

    def test_face_vs_identity_correlation(self) -> None:
        items = _make_items_with_metrics(
            [0.7, 0.8, 0.9, 0.95, 0.6],
            [3.0, 4.0, 4.5, 5.0, 2.5],
            metric="face_similarity",
            dimension="identity_preservation",
        )
        corr = compute_correlation(items, "face_similarity", "identity_preservation")
        assert corr is not None
        assert corr.metric_pair == "face_similarity_vs_identity_preservation"
        assert corr.pearson_r > 0.5


class TestInterpretation:
    def test_high(self) -> None:
        assert _interpret_r(0.85) == "high_agreement"
        assert _interpret_r(-0.75) == "high_agreement"

    def test_moderate(self) -> None:
        assert _interpret_r(0.5) == "moderate"
        assert _interpret_r(-0.45) == "moderate"

    def test_low(self) -> None:
        assert _interpret_r(0.2) == "low_agreement"
        assert _interpret_r(-0.1) == "low_agreement"


class TestPValueApprox:
    def test_zero_t_gives_high_pvalue(self) -> None:
        p = _approx_t_pvalue(0.0, 10)
        assert p >= 0.5

    def test_large_t_gives_small_pvalue(self) -> None:
        p = _approx_t_pvalue(10.0, 30)
        assert p < 0.01

    def test_invalid_df(self) -> None:
        assert _approx_t_pvalue(1.0, 0) == 1.0


# -----------------------------------------------------------------------
# runner.py integration
# -----------------------------------------------------------------------


class TestRunnerMetricsParam:
    def test_run_bench_accepts_metrics_param(self) -> None:
        """Verify run_bench() signature includes the metrics parameter."""
        import inspect
        from evalytic.bench.runner import run_bench

        sig = inspect.signature(run_bench)
        assert "metrics" in sig.parameters

    def test_config_includes_metrics(self) -> None:
        """A report built with metrics should store them in config."""
        report = BenchReport(
            name="test",
            models=["m1"],
            judge="gemini-2.0-flash",
            pipeline="text2img",
            config={"metrics": ["clip"]},
        )
        assert report.config["metrics"] == ["clip"]


# -----------------------------------------------------------------------
# runner.py — aggregate metric_averages
# -----------------------------------------------------------------------


class TestMetricAverages:
    def test_aggregate_metric_averages(self) -> None:
        from evalytic.bench.runner import _aggregate_model_summary

        items = [
            BenchItem(
                item_id="item-001",
                prompt="A cat",
                results={
                    "model-a": ImageResult(
                        model="model-a",
                        image_url="https://example.com/1.jpg",
                        scores=[DimensionResult(dimension="visual_quality", score=4.0)],
                        metrics=[MetricResult(metric="clip_score", value=0.82)],
                    ),
                },
            ),
            BenchItem(
                item_id="item-002",
                prompt="A dog",
                results={
                    "model-a": ImageResult(
                        model="model-a",
                        image_url="https://example.com/2.jpg",
                        scores=[DimensionResult(dimension="visual_quality", score=3.0)],
                        metrics=[MetricResult(metric="clip_score", value=0.78)],
                    ),
                },
            ),
        ]
        ms = _aggregate_model_summary("model-a", items, ["visual_quality"])
        assert "clip_score" in ms.metric_averages
        assert ms.metric_averages["clip_score"] == pytest.approx(0.80, abs=0.01)


# -----------------------------------------------------------------------
# terminal.py — metric columns + correlation display
# -----------------------------------------------------------------------


class TestTerminalMetrics:
    def _make_report_with_metrics(self) -> BenchReport:
        return BenchReport(
            name="test-bench",
            models=["model-a", "model-b"],
            judge="gemini-2.0-flash",
            pipeline="text2img",
            dimensions=["visual_quality", "prompt_adherence"],
            items=[],
            summary={
                "model-a": ModelSummary(
                    model="model-a",
                    overall_score=4.0,
                    dimension_averages={"visual_quality": 4.0, "prompt_adherence": 4.0},
                    metric_averages={"clip_score": 0.82},
                ),
                "model-b": ModelSummary(
                    model="model-b",
                    overall_score=3.5,
                    dimension_averages={"visual_quality": 3.5, "prompt_adherence": 3.5},
                    metric_averages={"clip_score": 0.75},
                ),
            },
            winner="model-a",
            ranking=[("model-a", 4.0), ("model-b", 3.5)],
            cost=CostBreakdown(judge_provider="gemini-2.0-flash"),
            correlations=[
                CorrelationStat(
                    metric_pair="clip_score_vs_prompt_adherence",
                    pearson_r=0.84,
                    p_value=0.002,
                    interpretation="high_agreement",
                ),
            ],
        )

    def test_comparison_table_includes_clip_column(self, capsys: pytest.CaptureFixture[str]) -> None:
        report = self._make_report_with_metrics()
        print_comparison_table(report)
        output = capsys.readouterr().out
        assert "CLIP" in output
        assert "0.82" in output or "0.820" in output

    def test_correlation_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        report = self._make_report_with_metrics()
        print_correlation(report.correlations)
        output = capsys.readouterr().out
        assert "0.84" in output
        assert "high agreement" in output

    def test_full_report_includes_correlation(self, capsys: pytest.CaptureFixture[str]) -> None:
        report = self._make_report_with_metrics()
        print_full_report(report)
        output = capsys.readouterr().out
        assert "CLIP" in output
        assert "0.84" in output


# -----------------------------------------------------------------------
# review_server.py — basic import/structure tests
# -----------------------------------------------------------------------


class TestReviewServer:
    def test_importable(self) -> None:
        from evalytic.report.review_server import ReviewServer
        assert ReviewServer is not None

    def test_constructor(self) -> None:
        from evalytic.report.review_server import ReviewServer

        report = BenchReport(
            name="test",
            models=["m1"],
            judge="gemini-2.0-flash",
            pipeline="text2img",
        )
        server = ReviewServer(report, port=9999)
        assert server._port == 9999
        assert server._report is report

    def test_merge_scores(self) -> None:
        from evalytic.report.review_server import ReviewServer

        report = BenchReport(
            name="test",
            models=["model-a"],
            judge="gemini-2.0-flash",
            pipeline="text2img",
            items=[
                BenchItem(
                    item_id="item-001",
                    prompt="A cat",
                    results={
                        "model-a": ImageResult(
                            model="model-a",
                            image_url="https://example.com/1.jpg",
                            scores=[
                                DimensionResult(dimension="visual_quality", score=4.0),
                            ],
                        ),
                    },
                ),
            ],
        )
        server = ReviewServer(report, port=9999)
        server._human_scores = {
            "item-001/model-a/0": {"score": 3, "notes": "shadow issue"},
        }
        updated = server.get_report()
        dim = updated.items[0].results["model-a"].scores[0]
        assert dim.human_score == 3.0
        assert dim.human_notes == "shadow issue"


# -----------------------------------------------------------------------
# types.py — open_review no longer raises
# -----------------------------------------------------------------------


class TestOpenReview:
    def test_open_review_calls_server(self) -> None:
        """open_review() should no longer raise NotImplementedError."""
        report = BenchReport(
            name="test",
            models=["m1"],
            judge="gemini-2.0-flash",
            pipeline="text2img",
        )
        # Patch ReviewServer to avoid starting a real server
        with patch("evalytic.report.review_server.ReviewServer") as MockServer:
            instance = MockServer.return_value
            instance.get_report.return_value = report
            report.open_review(port=9999)
            MockServer.assert_called_once_with(report, port=9999)
            instance.start.assert_called_once()
            instance.wait.assert_called_once()
            instance.get_report.assert_called_once()


# -----------------------------------------------------------------------
# types.py — ScoringConfig
# -----------------------------------------------------------------------


class TestScoringConfig:
    def test_default_has_clip_lpips_and_face(self) -> None:
        sc = ScoringConfig.default()
        assert "clip_score" in sc.metric_configs
        assert "lpips" in sc.metric_configs
        assert "face_similarity" in sc.metric_configs

    def test_default_clip_values(self) -> None:
        cfg = ScoringConfig.default().metric_configs["clip_score"]
        assert cfg.flag_threshold == 0.18
        assert cfg.weight == 0.20
        assert cfg.normalize_range == (0.18, 0.35)

    def test_default_face_values(self) -> None:
        cfg = ScoringConfig.default().metric_configs["face_similarity"]
        assert cfg.flag_threshold == 0.60
        assert cfg.weight == 0.20
        assert cfg.normalize_range == (0.60, 0.95)

    def test_empty_config(self) -> None:
        sc = ScoringConfig()
        assert sc.metric_configs == {}

    def test_to_dict(self) -> None:
        sc = ScoringConfig.default()
        d = sc.to_dict()
        assert "clip_score" in d
        assert d["clip_score"]["weight"] == 0.20
        assert d["clip_score"]["normalize_range"] == [0.18, 0.35]
        assert "face_similarity" in d


# -----------------------------------------------------------------------
# metrics.py — normalization
# -----------------------------------------------------------------------


class TestNormalization:
    def test_midpoint_maps_to_2_5(self) -> None:
        from evalytic.bench.metrics import normalize_metric

        # Midpoint of (0.18, 0.35) = 0.265
        result = normalize_metric(0.265, 0.18, 0.35)
        assert result == pytest.approx(2.5, abs=0.01)

    def test_min_maps_to_0(self) -> None:
        from evalytic.bench.metrics import normalize_metric

        assert normalize_metric(0.18, 0.18, 0.35) == pytest.approx(0.0)

    def test_max_maps_to_5(self) -> None:
        from evalytic.bench.metrics import normalize_metric

        assert normalize_metric(0.35, 0.18, 0.35) == pytest.approx(5.0)

    def test_clamp_below(self) -> None:
        from evalytic.bench.metrics import normalize_metric

        assert normalize_metric(0.10, 0.18, 0.35) == 0.0

    def test_clamp_above(self) -> None:
        from evalytic.bench.metrics import normalize_metric

        assert normalize_metric(0.50, 0.18, 0.35) == 5.0

    def test_invalid_range(self) -> None:
        from evalytic.bench.metrics import normalize_metric

        assert normalize_metric(0.5, 0.5, 0.5) == 0.0


# -----------------------------------------------------------------------
# metrics.py — threshold check
# -----------------------------------------------------------------------


class TestThresholdCheck:
    def test_below_threshold_returns_flag(self) -> None:
        from evalytic.bench.metrics import check_metric_threshold

        cfg = MetricScoringConfig(flag_threshold=0.18, weight=0.20, normalize_range=(0.18, 0.35))
        flag = check_metric_threshold("clip_score", 0.12, cfg)
        assert flag is not None
        assert flag.metric == "clip_score"
        assert flag.value == 0.12
        assert flag.threshold == 0.18
        assert "below threshold" in flag.message

    def test_above_threshold_returns_none(self) -> None:
        from evalytic.bench.metrics import check_metric_threshold

        cfg = MetricScoringConfig(flag_threshold=0.18, weight=0.20, normalize_range=(0.18, 0.35))
        assert check_metric_threshold("clip_score", 0.25, cfg) is None

    def test_at_boundary_returns_flag(self) -> None:
        from evalytic.bench.metrics import check_metric_threshold

        cfg = MetricScoringConfig(flag_threshold=0.18, weight=0.20, normalize_range=(0.18, 0.35))
        flag = check_metric_threshold("clip_score", 0.18, cfg)
        assert flag is not None
        assert flag.value == 0.18

    def test_just_above_boundary_returns_none(self) -> None:
        from evalytic.bench.metrics import check_metric_threshold

        cfg = MetricScoringConfig(flag_threshold=0.18, weight=0.20, normalize_range=(0.18, 0.35))
        assert check_metric_threshold("clip_score", 0.1801, cfg) is None

    def test_face_below_threshold_returns_flag(self) -> None:
        from evalytic.bench.metrics import check_metric_threshold

        cfg = MetricScoringConfig(flag_threshold=0.60, weight=0.20, normalize_range=(0.60, 0.95))
        flag = check_metric_threshold("face_similarity", 0.45, cfg)
        assert flag is not None
        assert flag.metric == "face_similarity"
        assert "Face similarity" in flag.message

    def test_face_above_threshold_returns_none(self) -> None:
        from evalytic.bench.metrics import check_metric_threshold

        cfg = MetricScoringConfig(flag_threshold=0.60, weight=0.20, normalize_range=(0.60, 0.95))
        assert check_metric_threshold("face_similarity", 0.85, cfg) is None


# -----------------------------------------------------------------------
# runner.py — weighted overall scoring
# -----------------------------------------------------------------------


class TestWeightedOverall:
    def _make_items(self, clip_values: list[float], dim_scores: list[float]) -> list[BenchItem]:
        items = []
        for i, (cv, ds) in enumerate(zip(clip_values, dim_scores)):
            items.append(
                BenchItem(
                    item_id=f"item-{i:03d}",
                    prompt=f"prompt {i}",
                    results={
                        "model-a": ImageResult(
                            model="model-a",
                            image_url=f"https://example.com/{i}.jpg",
                            scores=[
                                DimensionResult(dimension="visual_quality", score=ds),
                                DimensionResult(dimension="prompt_adherence", score=ds),
                            ],
                            metrics=[MetricResult(metric="clip_score", value=cv)],
                        ),
                    },
                )
            )
        return items

    def test_clip_above_threshold_weighted_overall(self) -> None:
        from evalytic.bench.runner import _aggregate_model_summary

        items = self._make_items([0.28, 0.30], [4.0, 4.0])
        sc = ScoringConfig.default()
        ms = _aggregate_model_summary("model-a", items, ["visual_quality", "prompt_adherence"], sc)

        assert ms.metric_flags.get("clip_score") == "included"
        assert ms.weighted_overall_score is not None
        # VLM avg = 4.0, CLIP normalized = roughly midpoint-ish, weight = 0.20
        # weighted = 4.0 * 0.80 + normalized * 0.20
        assert ms.overall_score != 4.0  # Should differ from pure VLM
        assert ms.weighted_overall_score == ms.overall_score

    def test_clip_below_threshold_flagged(self) -> None:
        from evalytic.bench.runner import _aggregate_model_summary

        items = self._make_items([0.10, 0.12], [4.0, 4.0])
        sc = ScoringConfig.default()
        ms = _aggregate_model_summary("model-a", items, ["visual_quality", "prompt_adherence"], sc)

        assert ms.metric_flags.get("clip_score") == "flagged"
        # Below threshold → excluded → pure VLM average
        assert ms.overall_score == 4.0
        assert ms.weighted_overall_score is None

    def test_no_config_backward_compatible(self) -> None:
        from evalytic.bench.runner import _aggregate_model_summary

        items = self._make_items([0.28, 0.30], [4.0, 3.5])
        ms = _aggregate_model_summary("model-a", items, ["visual_quality", "prompt_adherence"])

        assert ms.metric_flags == {}
        assert ms.weighted_overall_score is None
        expected_avg = (4.0 + 3.5) / 2  # = 3.75
        assert ms.overall_score == pytest.approx(expected_avg, abs=0.01)

    def test_excessive_weight_clamped(self) -> None:
        """Weight total > 1.0 should be clamped, not produce negative VLM contribution."""
        from evalytic.bench.runner import _aggregate_model_summary

        items = self._make_items([0.28, 0.30], [4.0, 4.0])
        sc = ScoringConfig(
            metric_configs={
                "clip_score": MetricScoringConfig(
                    flag_threshold=0.18,
                    weight=0.90,  # Excessive weight
                    normalize_range=(0.18, 0.35),
                ),
            }
        )
        ms = _aggregate_model_summary("model-a", items, ["visual_quality", "prompt_adherence"], sc)

        # Overall should still be positive and reasonable
        assert ms.overall_score > 0
        assert ms.weighted_overall_score is not None
        assert ms.weighted_overall_score > 0

    def test_image_flags_attached_when_below_threshold(self) -> None:
        from evalytic.bench.runner import _aggregate_model_summary

        items = self._make_items([0.10, 0.12], [4.0, 4.0])
        sc = ScoringConfig.default()
        _aggregate_model_summary("model-a", items, ["visual_quality", "prompt_adherence"], sc)

        # Per-image flags should be attached
        for item in items:
            img = item.results["model-a"]
            assert len(img.flags) == 1
            assert img.flags[0].metric == "clip_score"


# -----------------------------------------------------------------------
# runner.py — cost efficiency metrics
# -----------------------------------------------------------------------


class TestCostEfficiency:
    def _make_items_for_model(
        self, model: str, score: float, cost: float, count: int = 2,
    ) -> list[BenchItem]:
        items = []
        for i in range(count):
            items.append(
                BenchItem(
                    item_id=f"item-{i:03d}",
                    prompt=f"prompt {i}",
                    results={
                        model: ImageResult(
                            model=model,
                            image_url=f"https://example.com/{model}/{i}.jpg",
                            generation_cost_usd=cost,
                            scores=[
                                DimensionResult(dimension="visual_quality", score=score),
                            ],
                        ),
                    },
                )
            )
        return items

    def test_score_per_dollar_computed(self) -> None:
        from evalytic.bench.runner import _aggregate_model_summary

        items = self._make_items_for_model("cheap-model", score=4.0, cost=0.003)
        ms = _aggregate_model_summary("cheap-model", items, ["visual_quality"])

        assert ms.cost_per_image == pytest.approx(0.003)
        assert ms.score_per_dollar == pytest.approx(4.0 / 0.003, rel=0.01)

    def test_cheap_model_higher_efficiency(self) -> None:
        from evalytic.bench.runner import _aggregate_model_summary

        items_cheap = self._make_items_for_model("cheap", score=3.8, cost=0.003)
        items_expensive = self._make_items_for_model("expensive", score=4.2, cost=0.050)

        ms_cheap = _aggregate_model_summary("cheap", items_cheap, ["visual_quality"])
        ms_expensive = _aggregate_model_summary("expensive", items_expensive, ["visual_quality"])

        # Cheap model: 3.8 / 0.003 ≈ 1267 score/$
        # Expensive model: 4.2 / 0.050 = 84 score/$
        assert ms_cheap.score_per_dollar > ms_expensive.score_per_dollar

    def test_zero_cost_gives_zero_efficiency(self) -> None:
        from evalytic.bench.runner import _aggregate_model_summary

        items = self._make_items_for_model("free-model", score=4.0, cost=0.0)
        ms = _aggregate_model_summary("free-model", items, ["visual_quality"])

        assert ms.cost_per_image == 0.0
        assert ms.score_per_dollar == 0.0

    def test_efficiency_ranking_in_report(self) -> None:
        report = BenchReport(
            name="test",
            models=["cheap", "expensive"],
            judge="gemini-2.0-flash",
            pipeline="text2img",
            summary={
                "cheap": ModelSummary(
                    model="cheap", overall_score=3.8,
                    cost_per_image=0.003, score_per_dollar=1266.7,
                ),
                "expensive": ModelSummary(
                    model="expensive", overall_score=4.2,
                    cost_per_image=0.050, score_per_dollar=84.0,
                ),
            },
            efficiency_ranking=[("cheap", 1266.7), ("expensive", 84.0)],
            best_value="cheap",
            winner="expensive",
            ranking=[("expensive", 4.2), ("cheap", 3.8)],
        )
        assert report.best_value == "cheap"
        assert report.winner == "expensive"
        assert report.efficiency_ranking[0][0] == "cheap"

    def test_efficiency_table_prints(self, capsys: pytest.CaptureFixture[str]) -> None:
        from evalytic.report.terminal import print_efficiency_table

        report = BenchReport(
            name="test",
            models=["cheap", "expensive"],
            judge="gemini-2.0-flash",
            pipeline="text2img",
            dimensions=["visual_quality"],
            summary={
                "cheap": ModelSummary(
                    model="cheap", overall_score=3.8,
                    cost_per_image=0.003, score_per_dollar=1266.7,
                ),
                "expensive": ModelSummary(
                    model="expensive", overall_score=4.2,
                    cost_per_image=0.050, score_per_dollar=84.0,
                ),
            },
            efficiency_ranking=[("cheap", 1266.7), ("expensive", 84.0)],
            best_value="cheap",
            winner="expensive",
            ranking=[("expensive", 4.2), ("cheap", 3.8)],
            cost=CostBreakdown(judge_provider="gemini-2.0-flash"),
        )
        print_efficiency_table(report)
        output = capsys.readouterr().out
        assert "BEST VALUE" in output
        assert "cheap" in output
        assert "cheaper" in output


# -----------------------------------------------------------------------
# Dimension weights
# -----------------------------------------------------------------------


class TestDimensionWeights:
    def test_equal_weights_default(self) -> None:
        """Empty dimension_weights = equal weight (1/n)."""
        sc = ScoringConfig()
        weights = sc.resolve_dimension_weights(["visual_quality", "prompt_adherence"])
        assert weights["visual_quality"] == pytest.approx(0.5)
        assert weights["prompt_adherence"] == pytest.approx(0.5)

    def test_custom_weights(self) -> None:
        """Weighted average is computed correctly."""
        from evalytic.bench.runner import _aggregate_model_summary

        items = [
            BenchItem(
                item_id="item-001",
                prompt="A cat",
                results={
                    "model-a": ImageResult(
                        model="model-a",
                        image_url="https://example.com/1.jpg",
                        scores=[
                            DimensionResult(dimension="visual_quality", score=5.0),
                            DimensionResult(dimension="prompt_adherence", score=1.0),
                        ],
                    ),
                },
            ),
        ]
        sc = ScoringConfig(dimension_weights={"visual_quality": 0.8, "prompt_adherence": 0.2})
        ms = _aggregate_model_summary("model-a", items, ["visual_quality", "prompt_adherence"], sc)
        # 5.0 * 0.8 + 1.0 * 0.2 = 4.2
        assert ms.overall_score == pytest.approx(4.2, abs=0.01)

    def test_partial_weights(self) -> None:
        """Unspecified dimensions get equal share of remaining weight."""
        sc = ScoringConfig(dimension_weights={"visual_quality": 0.6})
        weights = sc.resolve_dimension_weights(["visual_quality", "prompt_adherence", "text_rendering"])
        assert weights["visual_quality"] == pytest.approx(0.6)
        # Remaining 0.4 split equally between 2 unspecified
        assert weights["prompt_adherence"] == pytest.approx(0.2)
        assert weights["text_rendering"] == pytest.approx(0.2)

    def test_weights_normalize(self) -> None:
        """Weights that don't sum to 1.0 are normalized."""
        sc = ScoringConfig(dimension_weights={"visual_quality": 2.0, "prompt_adherence": 3.0})
        weights = sc.resolve_dimension_weights(["visual_quality", "prompt_adherence"])
        assert sum(weights.values()) == pytest.approx(1.0)
        assert weights["visual_quality"] == pytest.approx(0.4)
        assert weights["prompt_adherence"] == pytest.approx(0.6)

    def test_dim_weights_in_report_config(self) -> None:
        """Dimension weights should be stored in report config."""
        report = BenchReport(
            name="test",
            models=["m1"],
            judge="gemini-2.0-flash",
            pipeline="text2img",
            config={"dimension_weights": {"visual_quality": 0.8, "prompt_adherence": 0.2}},
        )
        assert report.config["dimension_weights"]["visual_quality"] == 0.8

    def test_dim_weights_with_metrics(self) -> None:
        """Dimension weights work alongside metric weights."""
        from evalytic.bench.runner import _aggregate_model_summary

        items = [
            BenchItem(
                item_id="item-001",
                prompt="A cat",
                results={
                    "model-a": ImageResult(
                        model="model-a",
                        image_url="https://example.com/1.jpg",
                        scores=[
                            DimensionResult(dimension="visual_quality", score=5.0),
                            DimensionResult(dimension="prompt_adherence", score=1.0),
                        ],
                        metrics=[MetricResult(metric="clip_score", value=0.28)],
                    ),
                },
            ),
        ]
        sc = ScoringConfig.default()
        sc.dimension_weights = {"visual_quality": 0.8, "prompt_adherence": 0.2}
        ms = _aggregate_model_summary("model-a", items, ["visual_quality", "prompt_adherence"], sc)
        # VLM weighted avg = 5.0 * 0.8 + 1.0 * 0.2 = 4.2
        # Then metric-weighted overall adjusts further
        assert ms.overall_score > 0

    def test_backward_compat_no_weights(self) -> None:
        """Empty dimension_weights produces same result as old behavior."""
        from evalytic.bench.runner import _aggregate_model_summary

        items = [
            BenchItem(
                item_id="item-001",
                prompt="A cat",
                results={
                    "model-a": ImageResult(
                        model="model-a",
                        image_url="https://example.com/1.jpg",
                        scores=[
                            DimensionResult(dimension="visual_quality", score=4.0),
                            DimensionResult(dimension="prompt_adherence", score=3.0),
                        ],
                    ),
                },
            ),
        ]
        ms_no_config = _aggregate_model_summary("model-a", items, ["visual_quality", "prompt_adherence"])
        ms_empty = _aggregate_model_summary("model-a", items, ["visual_quality", "prompt_adherence"], ScoringConfig())
        assert ms_no_config.overall_score == ms_empty.overall_score
        assert ms_no_config.overall_score == pytest.approx(3.5)

    def test_resolve_dimension_weights(self) -> None:
        """resolve_dimension_weights unit test."""
        sc = ScoringConfig(dimension_weights={"a": 0.5, "b": 0.3})
        weights = sc.resolve_dimension_weights(["a", "b", "c"])
        assert sum(weights.values()) == pytest.approx(1.0)
        assert weights["a"] > weights["b"] > weights["c"]

    def test_resolve_partial(self) -> None:
        """Partial weights: unspecified dims share remainder."""
        sc = ScoringConfig(dimension_weights={"x": 0.5})
        weights = sc.resolve_dimension_weights(["x", "y"])
        assert sum(weights.values()) == pytest.approx(1.0)
        assert weights["x"] == pytest.approx(0.5)
        assert weights["y"] == pytest.approx(0.5)

    def test_resolve_normalize(self) -> None:
        """Non-1.0 sum gets normalized."""
        sc = ScoringConfig(dimension_weights={"a": 1.0, "b": 1.0})
        weights = sc.resolve_dimension_weights(["a", "b"])
        assert weights["a"] == pytest.approx(0.5)
        assert weights["b"] == pytest.approx(0.5)


# -----------------------------------------------------------------------
# Normalize range overrides
# -----------------------------------------------------------------------


class TestNormalizeRangeOverrides:
    def test_custom_clip_range(self) -> None:
        """Custom normalize range produces different normalized value."""
        from evalytic.bench.metrics import normalize_metric

        # Default range (0.18, 0.35): 0.265 → 2.5
        default_result = normalize_metric(0.265, 0.18, 0.35)
        # Custom range (0.20, 0.40): 0.265 → different value
        custom_result = normalize_metric(0.265, 0.20, 0.40)
        assert default_result != custom_result
        assert 0 <= custom_result <= 5

    def test_range_in_scoring_config(self) -> None:
        """ScoringConfig stores custom normalize_range."""
        sc = ScoringConfig(
            metric_configs={
                "clip_score": MetricScoringConfig(
                    flag_threshold=0.18,
                    weight=0.20,
                    normalize_range=(0.20, 0.40),
                ),
            }
        )
        assert sc.metric_configs["clip_score"].normalize_range == (0.20, 0.40)

    def test_custom_range_in_to_dict(self) -> None:
        """to_dict() includes custom normalize_range."""
        sc = ScoringConfig(
            metric_configs={
                "clip_score": MetricScoringConfig(
                    flag_threshold=0.18,
                    weight=0.20,
                    normalize_range=(0.20, 0.40),
                ),
            }
        )
        d = sc.to_dict()
        assert d["clip_score"]["normalize_range"] == [0.20, 0.40]
