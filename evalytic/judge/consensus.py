"""Shared consensus helpers for metric-style judge results."""

from __future__ import annotations

import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

from ..exceptions import ValidationError

T = TypeVar("T")

MIN_JUDGES = 2
MAX_JUDGES = 3
DEFAULT_THRESHOLD = 0.15


def merge_median_or_average(
    values: list[Any],
    judge_names: list[str],
    agreement: str,
    *,
    metric_result_cls: Any,
    score_of: Callable[[Any], float] = lambda item: item.score,
    extra_details_builder: Callable[[list[Any], list[str], Any], dict[str, Any] | None] | None = None,
) -> Any:
    """Merge per-judge MetricResult-like values into one.

    3+ judges: median of scores; reason/details taken from the judge whose
    score is at the median position. 2 judges: arithmetic mean; reason/details
    from the judge closest to the mean. 1 judge: passthrough.
    """
    if not values:
        raise ValidationError("Cannot merge zero metric results.")

    scores = [float(score_of(value)) for value in values]
    if len(values) >= 3:
        ordered_pairs = sorted(zip(scores, values, judge_names), key=lambda pair: pair[0])
        median_idx = len(ordered_pairs) // 2
        final_score = ordered_pairs[median_idx][0]
        chosen = ordered_pairs[median_idx][1]
    elif len(values) == 2:
        final_score = sum(scores) / len(scores)
        chosen = min(zip(scores, values), key=lambda pair: abs(pair[0] - final_score))[1]
    else:
        final_score = scores[0]
        chosen = values[0]

    details = dict(getattr(chosen, "details", None) or {})
    if extra_details_builder is not None:
        extra = extra_details_builder(values, judge_names, chosen)
        if extra:
            details.update(extra)

    return metric_result_cls(
        metric_id=chosen.metric_id,
        score=round(final_score, 4),
        reason=getattr(chosen, "reason", None),
        details=details or None,
        judge=f"consensus({','.join(judge_names)})",
        judge_scores={name: round(float(score_of(value)), 4) for name, value in zip(judge_names, values)},
        agreement=agreement,
        cost=sum(float(getattr(value, "cost", 0.0)) for value in values),
        duration_ms=max(int(getattr(value, "duration_ms", 0)) for value in values),
    )


@dataclass
class _Outcome(Generic[T]):
    judge_name: str
    value: T | None = None
    error: Exception | None = None


def resolve_metric_consensus(
    judges: list[str],
    scorer: Callable[[str], T],
    score_of: Callable[[T], float],
    merge: Callable[[list[T], list[str], str], T],
    threshold: float = DEFAULT_THRESHOLD,
) -> T:
    """Resolve one metric value across 2 or 3 judges.

    The scorer runs once per judge name and returns a metric result. The merge
    callback is responsible for combining successful results into the final
    value and annotating agreement metadata if needed.
    """
    if len(judges) < MIN_JUDGES:
        raise ValidationError(
            f"Consensus mode requires at least {MIN_JUDGES} judges, got {len(judges)}."
        )
    if len(judges) > MAX_JUDGES:
        raise ValidationError(
            f"Consensus mode supports at most {MAX_JUDGES} judges, got {len(judges)}."
        )

    with ThreadPoolExecutor(max_workers=min(2, len(judges))) as pool:
        futures = {pool.submit(_run_scorer, scorer, judge): judge for judge in judges[:2]}
        first_two = [future.result() for future in futures]

    successful = [o for o in first_two if o.value is not None]
    if len(successful) == 2:
        diff = abs(score_of(successful[0].value) - score_of(successful[1].value))
        if diff <= threshold:
            return merge(
                [successful[0].value, successful[1].value],
                [successful[0].judge_name, successful[1].judge_name],
                "high",
            )

    if len(judges) == 2:
        if successful:
            return merge(
                [o.value for o in successful if o.value is not None],
                [o.judge_name for o in successful if o.value is not None],
                "disputed" if len(successful) == 2 else "degraded",
            )
        errors = ", ".join(str(o.error) for o in first_two if o.error)
        raise RuntimeError(f"All consensus judges failed: {errors}")

    third = _run_scorer(scorer, judges[2])
    successful = [o for o in [*first_two, third] if o.value is not None]
    if not successful:
        errors = ", ".join(str(o.error) for o in [*first_two, third] if o.error)
        raise RuntimeError(f"All consensus judges failed: {errors}")

    values = [o.value for o in successful if o.value is not None]
    names = [o.judge_name for o in successful if o.value is not None]
    if len(values) == 1:
        agreement = "degraded"
    elif len(values) == 2:
        diff = abs(score_of(values[0]) - score_of(values[1]))
        agreement = "high" if diff <= threshold else "disputed"
    else:
        median = statistics.median(score_of(v) for v in values)
        agreement = "high" if all(abs(score_of(v) - median) <= threshold for v in values) else "disputed"
    return merge(values, names, agreement)


def _run_scorer(scorer: Callable[[str], T], judge_name: str) -> _Outcome[T]:
    try:
        return _Outcome(judge_name=judge_name, value=scorer(judge_name))
    except Exception as exc:  # pragma: no cover - exercised through callers
        return _Outcome(judge_name=judge_name, error=exc)
