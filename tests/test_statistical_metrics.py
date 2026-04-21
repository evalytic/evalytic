"""Deterministic (no-LLM) text metric tests.

Covers BLEU / ROUGE / exact_match / levenshtein / string_presence via their
pure-function source in ``evalytic.text.metrics._deterministic`` and the
metric wrappers in ``evalytic.text.metrics.statistical``.
"""

from __future__ import annotations

import pytest

from evalytic.exceptions import ValidationError
from evalytic.text.metrics._deterministic import (
    bleu_score,
    exact_match_score,
    levenshtein_similarity,
    rouge_l_f1,
    string_presence_score,
    tokens,
)
from evalytic.text.metrics.statistical import (
    BleuMetric,
    ExactMatchMetric,
    LevenshteinMetric,
    RougeMetric,
    StringPresenceMetric,
)
from evalytic.text.types import TextTestCase


def _case(output: str, expected: str | None = "") -> TextTestCase:
    return TextTestCase(input="ignored", output=output, expected=expected)


class TestBleu:
    def test_perfect_match_returns_1_0(self) -> None:
        score = bleu_score(
            "the cat sat on the mat quickly",
            "the cat sat on the mat quickly",
        )
        assert score == pytest.approx(1.0, rel=1e-6)

    def test_no_overlap_returns_0_0(self) -> None:
        score = bleu_score("alpha beta gamma delta", "one two three four")
        assert score == 0.0

    def test_brevity_penalty_applied_when_candidate_shorter(self) -> None:
        long_ref = "the quick brown fox jumps over the lazy dog"
        short_cand = "the quick brown fox"
        assert bleu_score(short_cand, long_ref) < 1.0

    def test_missing_4gram_yields_zero(self) -> None:
        # Fewer than 4 tokens -> 4-gram precision = 0 -> BLEU = 0
        score = bleu_score("only three tokens", "only three tokens")
        assert score == 0.0

    def test_bleu_metric_wrapper_requires_expected(self) -> None:
        with pytest.raises(ValidationError):
            BleuMetric().score(_case("hello", None))

    def test_bleu_metric_wrapper_empty_candidate_is_zero(self) -> None:
        result = BleuMetric().score(_case("", "some reference"))
        assert result.score == 0.0


class TestRouge:
    def test_rouge_l_identical_texts_is_1(self) -> None:
        assert rouge_l_f1("one two three", "one two three") == pytest.approx(1.0)

    def test_rouge_l_empty_either_side_is_zero(self) -> None:
        assert rouge_l_f1("", "anything") == 0.0
        assert rouge_l_f1("anything", "") == 0.0

    def test_rouge_metric_wrapper(self) -> None:
        result = RougeMetric().score(_case("cat sat mat", "the cat sat on the mat"))
        assert 0.0 < result.score <= 1.0


class TestExactMatch:
    def test_normalizes_whitespace(self) -> None:
        assert exact_match_score("  hello   world  ", "hello world") == 1.0

    def test_case_insensitive(self) -> None:
        assert exact_match_score("Hello World", "hello world") == 1.0

    def test_turkish_characters_preserved(self) -> None:
        # Turkish-specific characters should survive normalization (no stripping).
        # Note: Python casefold() does not dotless-i-fold "İ" -> "i"; that is a
        # locale-specific rule we intentionally do NOT emulate to keep behavior
        # deterministic. Test covers the characters that *do* fold cleanly.
        assert exact_match_score("Çığlık Şoför", "çığlık şoför") == 1.0
        assert exact_match_score("GÜZEL günler", "güzel günler") == 1.0
        assert exact_match_score("Ankara", "istanbul") == 0.0


class TestLevenshtein:
    def test_identical_returns_1_0(self) -> None:
        assert levenshtein_similarity("abc", "abc") == pytest.approx(1.0)

    def test_completely_different_positive_clamped(self) -> None:
        score = levenshtein_similarity("abc", "xyz")
        assert score >= 0.0 and score < 1.0

    def test_lev_wrapper_stores_distance(self) -> None:
        result = LevenshteinMetric().score(_case("abc", "abd"))
        assert result.details is not None
        assert result.details["distance"] == 1


class TestStringPresence:
    def test_substring_match(self) -> None:
        assert string_presence_score("the cat sat on the mat", "cat sat") == 1.0

    def test_missing_substring(self) -> None:
        assert string_presence_score("dog and bird", "cat") == 0.0

    def test_empty_expected_returns_0(self) -> None:
        assert string_presence_score("anything", "") == 0.0

    def test_wrapper_raises_without_expected(self) -> None:
        with pytest.raises(ValidationError):
            StringPresenceMetric().score(_case("hi", None))


def test_tokens_splits_on_punctuation() -> None:
    assert tokens("Hello, world! Foo-bar.") == ["hello", "world", "foo", "bar"]


class TestExactMatchWrapper:
    def test_exact_match_wrapper_ok(self) -> None:
        result = ExactMatchMetric().score(_case("hi", "HI"))
        assert result.score == 1.0

    def test_exact_match_wrapper_mismatch(self) -> None:
        result = ExactMatchMetric().score(_case("hello", "world"))
        assert result.score == 0.0
