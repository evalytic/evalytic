"""Pure-function deterministic text scoring helpers.

This module is the single source of truth for BLEU / ROUGE / exact_match /
levenshtein / string_presence logic used by both the SDK ``statistical.py``
metrics and the Judge Lambda worker. Keep it dependency-free (stdlib only)
so it can be copied verbatim into the Lambda bundle by build.sh.
"""

from __future__ import annotations

import math
import re
from collections import Counter


def normalize(text: str | None) -> str:
    return " ".join(str(text or "").strip().casefold().split())


def tokens(text: str | None) -> list[str]:
    return re.findall(r"\w+", normalize(text))


def ngrams(tokens_list: list[str], n: int) -> Counter[tuple[str, ...]]:
    if len(tokens_list) < n:
        return Counter()
    return Counter(tuple(tokens_list[idx : idx + n]) for idx in range(len(tokens_list) - n + 1))


def lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, token_a in enumerate(a, start=1):
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insertions = previous[j] + 1
            deletions = current[j - 1] + 1
            substitutions = previous[j - 1] + (char_a != char_b)
            current.append(min(insertions, deletions, substitutions))
        previous = current
    return previous[-1]


def bleu_score(candidate_text: str, reference_text: str) -> float:
    """BLEU-4 with smoothed brevity penalty."""
    candidate = tokens(candidate_text)
    reference = tokens(reference_text)
    if not candidate or not reference:
        return 0.0

    precisions: list[float] = []
    for n in range(1, 5):
        cand_ngrams = ngrams(candidate, n)
        ref_ngrams = ngrams(reference, n)
        total = sum(cand_ngrams.values())
        if total == 0:
            precisions.append(0.0)
            continue
        overlap = sum(min(count, ref_ngrams[ngram]) for ngram, count in cand_ngrams.items())
        precisions.append(overlap / total)

    if any(p == 0.0 for p in precisions):
        return 0.0
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / 4)
    brevity_penalty = (
        1.0
        if len(candidate) > len(reference)
        else math.exp(1 - len(reference) / max(1, len(candidate)))
    )
    return geo_mean * brevity_penalty


def rouge_l_f1(candidate_text: str, reference_text: str) -> float:
    candidate = tokens(candidate_text)
    reference = tokens(reference_text)
    if not candidate or not reference:
        return 0.0
    lcs = lcs_length(candidate, reference)
    recall = lcs / len(reference)
    precision = lcs / len(candidate)
    if not (precision + recall):
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exact_match_score(candidate_text: str, reference_text: str) -> float:
    candidate = normalize(candidate_text)
    reference = normalize(reference_text)
    return 1.0 if candidate and candidate == reference else 0.0


def levenshtein_similarity(candidate_text: str, reference_text: str) -> float:
    candidate = normalize(candidate_text)
    reference = normalize(reference_text)
    max_len = max(len(candidate), len(reference), 1)
    distance = levenshtein_distance(candidate, reference)
    return max(0.0, 1 - (distance / max_len))


def string_presence_score(candidate_text: str, reference_text: str) -> float:
    candidate = normalize(candidate_text)
    reference = normalize(reference_text)
    return 1.0 if reference and reference in candidate else 0.0
