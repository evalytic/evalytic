"""Tests for the ConsensusJudge (adaptive 2+1 multi-judge scoring).

All tests are offline -- zero network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from evalytic.bench.consensus import CONSENSUS_THRESHOLD, ConsensusJudge
from evalytic.bench.types import DimensionResult
from evalytic.exceptions import ValidationError


def _make_dim(dim: str, score: float, confidence: float = 0.9) -> DimensionResult:
    return DimensionResult(
        dimension=dim,
        score=score,
        confidence=confidence,
        explanation=f"Score {score} from judge",
        evidence=[f"evidence-{score}"],
    )


class TestConsensusHighAgreement:
    def test_high_agreement_averages(self) -> None:
        """Two judges within threshold → average score, agreement='high'."""
        results_a = [_make_dim("visual_quality", 4.0)]
        results_b = [_make_dim("visual_quality", 4.5)]

        with patch("evalytic.bench.consensus.Judge") as MockJudge:
            instances = [MagicMock(), MagicMock(), MagicMock()]
            instances[0].score.return_value = results_a
            instances[1].score.return_value = results_b
            MockJudge.side_effect = instances

            cj = ConsensusJudge(["judge-a", "judge-b"])
            results = cj.score(
                image_url="https://example.com/img.png",
                dimensions=["visual_quality"],
                prompt="A cat",
            )
            cj.close()

        assert len(results) == 1
        assert results[0].score == 4.25  # average of 4.0 and 4.5
        assert results[0].agreement == "high"
        assert "judge-a" in results[0].judge_scores
        assert "judge-b" in results[0].judge_scores


class TestConsensusDisputed:
    def test_disputed_triggers_tiebreaker(self) -> None:
        """Difference > threshold with 3 judges → median, agreement='disputed'."""
        results_a = [_make_dim("visual_quality", 2.0)]
        results_b = [_make_dim("visual_quality", 4.0)]
        results_c = [_make_dim("visual_quality", 3.5)]

        with patch("evalytic.bench.consensus.Judge") as MockJudge:
            instances = [MagicMock(), MagicMock(), MagicMock()]
            instances[0].score.return_value = results_a
            instances[1].score.return_value = results_b
            instances[2].score.return_value = results_c
            MockJudge.side_effect = instances

            cj = ConsensusJudge(["judge-a", "judge-b", "judge-c"])
            results = cj.score(
                image_url="https://example.com/img.png",
                dimensions=["visual_quality"],
            )
            cj.close()

        assert len(results) == 1
        assert results[0].score == 3.5  # median of 2.0, 4.0, 3.5
        assert results[0].agreement == "disputed"
        assert len(results[0].judge_scores) == 3

    def test_two_judges_no_tiebreaker(self) -> None:
        """Dispute with only 2 judges → average (no tiebreaker available)."""
        results_a = [_make_dim("visual_quality", 2.0)]
        results_b = [_make_dim("visual_quality", 4.0)]

        with patch("evalytic.bench.consensus.Judge") as MockJudge:
            instances = [MagicMock(), MagicMock()]
            instances[0].score.return_value = results_a
            instances[1].score.return_value = results_b
            MockJudge.side_effect = instances

            cj = ConsensusJudge(["judge-a", "judge-b"])
            results = cj.score(
                image_url="https://example.com/img.png",
                dimensions=["visual_quality"],
            )
            cj.close()

        assert len(results) == 1
        assert results[0].score == 3.0  # average of 2.0 and 4.0
        assert results[0].agreement == "disputed"


class TestConsensusDegraded:
    def test_degraded_one_judge_fails(self) -> None:
        """One judge fails → use the other, agreement='degraded'."""
        results_b = [_make_dim("visual_quality", 4.0)]

        with patch("evalytic.bench.consensus.Judge") as MockJudge:
            instances = [MagicMock(), MagicMock()]
            instances[0].score.side_effect = Exception("API error")
            instances[1].score.return_value = results_b
            MockJudge.side_effect = instances

            cj = ConsensusJudge(["judge-a", "judge-b"])
            results = cj.score(
                image_url="https://example.com/img.png",
                dimensions=["visual_quality"],
            )
            cj.close()

        assert len(results) == 1
        assert results[0].score == 4.0
        assert results[0].agreement == "degraded"
        assert "judge-b" in results[0].judge_scores
        assert "judge-a" not in results[0].judge_scores

    def test_both_fail_raises_judge_error(self) -> None:
        """Both judges fail → JudgeError raised (not silent zero)."""
        from evalytic.exceptions import JudgeError

        with patch("evalytic.bench.consensus.Judge") as MockJudge:
            instances = [MagicMock(), MagicMock()]
            instances[0].score.side_effect = Exception("API error")
            instances[1].score.side_effect = Exception("API error")
            MockJudge.side_effect = instances

            cj = ConsensusJudge(["judge-a", "judge-b"])
            with pytest.raises(JudgeError, match="All judges failed"):
                cj.score(
                    image_url="https://example.com/img.png",
                    dimensions=["visual_quality"],
                )
            cj.close()


class TestConsensusValidation:
    def test_min_two_judges(self) -> None:
        """Less than 2 judges → ValidationError."""
        with pytest.raises(ValidationError, match="at least 2"):
            ConsensusJudge(["only-one"])

    def test_max_three_judges(self) -> None:
        """More than 3 judges → ValidationError."""
        with pytest.raises(ValidationError, match="at most 3"):
            ConsensusJudge(["a", "b", "c", "d"])


class TestConsensusMetadata:
    def test_judge_scores_populated(self) -> None:
        """DimensionResult contains per-judge scores."""
        results_a = [_make_dim("visual_quality", 4.2)]
        results_b = [_make_dim("visual_quality", 4.0)]

        with patch("evalytic.bench.consensus.Judge") as MockJudge:
            instances = [MagicMock(), MagicMock()]
            instances[0].score.return_value = results_a
            instances[1].score.return_value = results_b
            MockJudge.side_effect = instances

            cj = ConsensusJudge(["gemini-3-flash", "gpt-4o-mini"])
            results = cj.score(
                image_url="https://example.com/img.png",
                dimensions=["visual_quality"],
            )
            cj.close()

        assert results[0].judge_scores == {"gemini-3-flash": 4.2, "gpt-4o-mini": 4.0}

    def test_agreement_field_set(self) -> None:
        """Agreement field is correctly populated."""
        results_a = [_make_dim("visual_quality", 4.5)]
        results_b = [_make_dim("visual_quality", 4.5)]

        with patch("evalytic.bench.consensus.Judge") as MockJudge:
            instances = [MagicMock(), MagicMock()]
            instances[0].score.return_value = results_a
            instances[1].score.return_value = results_b
            MockJudge.side_effect = instances

            cj = ConsensusJudge(["j1", "j2"])
            results = cj.score(
                image_url="https://example.com/img.png",
                dimensions=["visual_quality"],
            )
            cj.close()

        assert results[0].agreement == "high"

    def test_explanation_highest_confidence(self) -> None:
        """Explanation comes from the judge with highest confidence."""
        results_a = [_make_dim("visual_quality", 4.0, confidence=0.7)]
        results_a[0] = DimensionResult(
            dimension="visual_quality", score=4.0, confidence=0.7,
            explanation="Low confidence explanation", evidence=["a"],
        )
        results_b = [DimensionResult(
            dimension="visual_quality", score=4.2, confidence=0.95,
            explanation="High confidence explanation", evidence=["b"],
        )]

        with patch("evalytic.bench.consensus.Judge") as MockJudge:
            instances = [MagicMock(), MagicMock()]
            instances[0].score.return_value = results_a
            instances[1].score.return_value = results_b
            MockJudge.side_effect = instances

            cj = ConsensusJudge(["j1", "j2"])
            results = cj.score(
                image_url="https://example.com/img.png",
                dimensions=["visual_quality"],
            )
            cj.close()

        assert results[0].explanation == "High confidence explanation"

    def test_context_manager(self) -> None:
        """ConsensusJudge works as a context manager."""
        with patch("evalytic.bench.consensus.Judge") as MockJudge:
            instances = [MagicMock(), MagicMock()]
            MockJudge.side_effect = instances

            with ConsensusJudge(["j1", "j2"]) as cj:
                assert cj is not None

            instances[0].close.assert_called_once()
            instances[1].close.assert_called_once()

    def test_judge_string_property(self) -> None:
        """judge_string returns 'consensus:a,b' format."""
        with patch("evalytic.bench.consensus.Judge"):
            cj = ConsensusJudge(["gemini-3-flash", "gpt-4o-mini"])
            assert cj.judge_string == "consensus:gemini-3-flash,gpt-4o-mini"
            cj.close()


class TestConsensusMixedDimensions:
    def test_mixed_dimensions(self) -> None:
        """Some dimensions high agreement, some disputed."""
        results_a = [
            _make_dim("visual_quality", 4.0),
            _make_dim("prompt_adherence", 2.0),
        ]
        results_b = [
            _make_dim("visual_quality", 4.3),
            _make_dim("prompt_adherence", 4.0),
        ]
        results_c = [
            _make_dim("prompt_adherence", 3.5),
        ]

        with patch("evalytic.bench.consensus.Judge") as MockJudge:
            instances = [MagicMock(), MagicMock(), MagicMock()]
            instances[0].score.return_value = results_a
            instances[1].score.return_value = results_b
            # Tiebreaker only called for disputed dimensions
            instances[2].score.return_value = results_c
            MockJudge.side_effect = instances

            cj = ConsensusJudge(["j1", "j2", "j3"])
            results = cj.score(
                image_url="https://example.com/img.png",
                dimensions=["visual_quality", "prompt_adherence"],
            )
            cj.close()

        result_map = {r.dimension: r for r in results}
        assert result_map["visual_quality"].agreement == "high"
        assert result_map["visual_quality"].score == 4.15  # avg(4.0, 4.3)
        assert result_map["prompt_adherence"].agreement == "disputed"
        assert result_map["prompt_adherence"].score == 3.5  # median(2.0, 4.0, 3.5)
