"""Consensus judge -- adaptive 2+1 multi-judge scoring.

Algorithm:
    For each (image, dimension) pair:
    1. Judge A + Judge B score in parallel
    2. |A - B| <= threshold → average, agreement="high"
    3. |A - B| > threshold  → Judge C (tiebreaker) → median, agreement="disputed"
    4. One judge fails → use the other, agreement="degraded"
"""

from __future__ import annotations

import statistics
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..exceptions import ValidationError
from .judge import Judge
from .types import DimensionResult

CONSENSUS_THRESHOLD = 0.5
MAX_JUDGES = 3
MIN_JUDGES = 2


class ConsensusJudge:
    """Score images using multiple VLM judges with adaptive consensus."""

    def __init__(
        self,
        judges: list[str],
        api_keys: dict[str, str] | None = None,
        base_url: str | None = None,
    ) -> None:
        if len(judges) < MIN_JUDGES:
            raise ValidationError(
                f"Consensus mode requires at least {MIN_JUDGES} judges, got {len(judges)}."
            )
        if len(judges) > MAX_JUDGES:
            raise ValidationError(
                f"Consensus mode supports at most {MAX_JUDGES} judges, got {len(judges)}."
            )

        self._judge_names = list(judges)
        api_keys = api_keys or {}

        self._judges: list[Judge] = []
        for j in judges:
            self._judges.append(
                Judge(
                    judge=j,
                    api_key=api_keys.get(j),
                    base_url=base_url,
                )
            )

        self._request_counts: dict[str, int] = {j: 0 for j in judges}

    @property
    def judge_string(self) -> str:
        return "consensus:" + ",".join(self._judge_names)

    @property
    def request_counts(self) -> dict[str, int]:
        return dict(self._request_counts)

    def score(
        self,
        image_url: str,
        dimensions: list[str],
        prompt: str | None = None,
        input_image_url: str | None = None,
    ) -> list[DimensionResult]:
        """Score an image using consensus of multiple judges.

        Same interface as ``Judge.score()``.
        """
        # Step 1: Two primary judges score in parallel
        primary_a, primary_b = self._judges[0], self._judges[1]
        results_a, results_b = self._parallel_score(
            [primary_a, primary_b], image_url, dimensions, prompt, input_image_url,
        )

        # Build dimension lookup
        map_a = {d.dimension: d for d in results_a} if results_a else {}
        map_b = {d.dimension: d for d in results_b} if results_b else {}

        # Track which dimensions need tiebreaker
        needs_tiebreaker: list[str] = []
        consensus_results: dict[str, DimensionResult] = {}

        for dim in dimensions:
            da = map_a.get(dim)
            db = map_b.get(dim)

            if da is not None and db is not None:
                diff = abs(da.score - db.score)
                if diff <= CONSENSUS_THRESHOLD:
                    # High agreement → average
                    consensus_results[dim] = self._merge_results(
                        dim, [da, db],
                        [self._judge_names[0], self._judge_names[1]],
                        agreement="high",
                    )
                else:
                    needs_tiebreaker.append(dim)
            elif da is not None:
                # B failed → use A
                consensus_results[dim] = self._single_result(
                    da, self._judge_names[0], agreement="degraded",
                )
            elif db is not None:
                # A failed → use B
                consensus_results[dim] = self._single_result(
                    db, self._judge_names[1], agreement="degraded",
                )
            else:
                # Both failed → zero score
                consensus_results[dim] = DimensionResult(
                    dimension=dim,
                    score=0.0,
                    confidence=0.0,
                    explanation="All judges failed for this dimension.",
                    agreement="degraded",
                    judge_scores={},
                )

        # Step 2: Tiebreaker for disputed dimensions
        if needs_tiebreaker and len(self._judges) >= 3:
            tiebreaker = self._judges[2]
            results_c = self._safe_score(
                tiebreaker, image_url, needs_tiebreaker, prompt, input_image_url,
            )
            if results_c is not None:
                self._request_counts[self._judge_names[2]] += len(needs_tiebreaker)
            map_c = {d.dimension: d for d in results_c} if results_c else {}

            for dim in needs_tiebreaker:
                da = map_a.get(dim)
                db = map_b.get(dim)
                dc = map_c.get(dim)

                all_results = [r for r in [da, db, dc] if r is not None]
                all_names = []
                if da is not None:
                    all_names.append(self._judge_names[0])
                if db is not None:
                    all_names.append(self._judge_names[1])
                if dc is not None:
                    all_names.append(self._judge_names[2])

                if all_results:
                    consensus_results[dim] = self._merge_results(
                        dim, all_results, all_names, agreement="disputed",
                    )
                else:
                    consensus_results[dim] = DimensionResult(
                        dimension=dim, score=0.0, confidence=0.0,
                        explanation="All judges failed.", agreement="degraded",
                        judge_scores={},
                    )
        elif needs_tiebreaker:
            # No tiebreaker judge available — average the two disputed scores
            for dim in needs_tiebreaker:
                da = map_a.get(dim)
                db = map_b.get(dim)
                all_results = [r for r in [da, db] if r is not None]
                all_names = []
                if da is not None:
                    all_names.append(self._judge_names[0])
                if db is not None:
                    all_names.append(self._judge_names[1])
                consensus_results[dim] = self._merge_results(
                    dim, all_results, all_names, agreement="disputed",
                )

        return [consensus_results[dim] for dim in dimensions if dim in consensus_results]

    def _parallel_score(
        self,
        judges: list[Judge],
        image_url: str,
        dimensions: list[str],
        prompt: str | None,
        input_image_url: str | None,
    ) -> list[list[DimensionResult] | None]:
        """Score with multiple judges in parallel."""
        results: list[list[DimensionResult] | None] = [None] * len(judges)
        with ThreadPoolExecutor(max_workers=len(judges)) as pool:
            futures = {
                pool.submit(
                    self._safe_score, j, image_url, dimensions, prompt, input_image_url,
                ): i
                for i, j in enumerate(judges)
            }
            for future in futures:
                idx = futures[future]
                result = future.result()
                results[idx] = result
                if result is not None:
                    self._request_counts[self._judge_names[idx]] += len(dimensions)
        return results

    @staticmethod
    def _safe_score(
        judge: Judge,
        image_url: str,
        dimensions: list[str],
        prompt: str | None,
        input_image_url: str | None,
    ) -> list[DimensionResult] | None:
        """Score and catch exceptions, returning None on failure."""
        try:
            return judge.score(
                image_url=image_url,
                dimensions=dimensions,
                prompt=prompt,
                input_image_url=input_image_url,
            )
        except Exception:
            return None

    @staticmethod
    def _merge_results(
        dimension: str,
        results: list[DimensionResult],
        judge_names: list[str],
        agreement: str,
    ) -> DimensionResult:
        """Merge multiple judge results into one consensus result."""
        scores = [r.score for r in results]
        judge_scores = {name: r.score for name, r in zip(judge_names, results)}

        if len(scores) >= 3:
            final_score = statistics.median(scores)
        else:
            final_score = sum(scores) / len(scores)

        # Pick explanation from highest-confidence judge
        best = max(results, key=lambda r: r.confidence)
        # Merge evidence from all judges (deduplicated)
        seen: set[str] = set()
        merged_evidence: list[str] = []
        for r in results:
            for e in r.evidence:
                if e not in seen:
                    seen.add(e)
                    merged_evidence.append(e)

        return DimensionResult(
            dimension=dimension,
            score=round(final_score, 2),
            confidence=round(best.confidence, 2),
            explanation=best.explanation,
            evidence=merged_evidence,
            judge_scores=judge_scores,
            agreement=agreement,
        )

    @staticmethod
    def _single_result(
        result: DimensionResult,
        judge_name: str,
        agreement: str,
    ) -> DimensionResult:
        """Wrap a single judge's result with consensus metadata."""
        return DimensionResult(
            dimension=result.dimension,
            score=result.score,
            confidence=result.confidence,
            explanation=result.explanation,
            evidence=result.evidence,
            judge_scores={judge_name: result.score},
            agreement=agreement,
        )

    def close(self) -> None:
        for j in self._judges:
            j.close()

    def __enter__(self) -> ConsensusJudge:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
