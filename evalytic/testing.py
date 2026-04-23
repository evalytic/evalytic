"""Pytest-native assertions for Evalytic metrics.

Use inside any pytest test function:

    from evalytic.testing import assert_test
    from evalytic.text.types import RAGTestCase, RetrievedChunk

    def test_rag_quality():
        case = RAGTestCase(
            query="What is the capital of France?",
            response="Paris.",
            contexts=[RetrievedChunk(text="Paris is the capital of France.")],
        )
        assert_test(case, metrics={"faithfulness": 0.8, "hallucination": 0.9})

The helper runs the appropriate Evalytic runner for the given test case, reads the
per-metric averages, and raises ``AssertionError`` when any threshold is not met.
All failures are reported together so one test run surfaces every regression.
"""

from __future__ import annotations

from typing import Any, Mapping

from .agent.runner import evaluate_agent
from .agent.types import AgentTestCase
from .text.runner import evaluate_rag, evaluate_text
from .text.types import MetricEvalReport, RAGTestCase, TextTestCase

TestCase = RAGTestCase | TextTestCase | AgentTestCase

__all__ = ["assert_test", "assert_metric", "evaluate_case"]


def assert_test(
    case: TestCase,
    metrics: Mapping[str, float],
    *,
    judge: str | None = None,
    judges: list[str] | None = None,
    base_url: str | None = None,
    config: dict[str, Any] | None = None,
) -> MetricEvalReport:
    """Evaluate ``case`` and assert every metric meets its threshold.

    Args:
        case: a ``RAGTestCase``, ``TextTestCase`` or ``AgentTestCase``.
        metrics: mapping from metric id to the minimum acceptable score (inclusive).
        judge: optional judge model override (ignored when ``judges`` is provided).
        judges: optional list of judges for consensus mode.
        base_url: optional custom judge API base URL.
        config: optional SDK config dict (same shape used by the CLI).

    Returns:
        The underlying ``MetricEvalReport`` so tests can inspect individual scores.

    Raises:
        AssertionError: when any requested metric is missing or below its threshold.
        TypeError: when ``case`` is not a supported test case type.
    """
    if not metrics:
        raise ValueError("assert_test requires at least one metric threshold.")

    report = evaluate_case(
        case,
        metric_ids=list(metrics.keys()),
        judge=judge,
        judges=judges,
        base_url=base_url,
        config=config,
    )
    averages = report.metric_averages()

    failures: list[str] = []
    for metric_id, threshold in metrics.items():
        actual = averages.get(metric_id)
        if actual is None:
            failures.append(f"  - {metric_id}: not produced by runner (got: {sorted(averages)})")
        elif actual < threshold:
            failures.append(f"  - {metric_id}: {actual:.4f} < threshold {threshold:.4f}")

    if failures:
        raise AssertionError(
            "Evalytic metric thresholds failed:\n" + "\n".join(failures)
        )

    return report


def assert_metric(
    case: TestCase,
    metric_id: str,
    threshold: float,
    *,
    judge: str | None = None,
    judges: list[str] | None = None,
    base_url: str | None = None,
    config: dict[str, Any] | None = None,
) -> MetricEvalReport:
    """Convenience wrapper around ``assert_test`` for a single metric."""
    return assert_test(
        case,
        {metric_id: threshold},
        judge=judge,
        judges=judges,
        base_url=base_url,
        config=config,
    )


def evaluate_case(
    case: TestCase,
    *,
    metric_ids: list[str] | None = None,
    judge: str | None = None,
    judges: list[str] | None = None,
    base_url: str | None = None,
    config: dict[str, Any] | None = None,
) -> MetricEvalReport:
    """Dispatch to the correct Evalytic runner based on the test case type."""
    kwargs: dict[str, Any] = {
        "judges": judges,
        "base_url": base_url,
        "config": config,
    }
    if judge is not None:
        kwargs["judge"] = judge

    if isinstance(case, RAGTestCase):
        return evaluate_rag([case], metric_ids=metric_ids, **kwargs)
    if isinstance(case, TextTestCase):
        return evaluate_text([case], metric_ids=metric_ids, **kwargs)
    if isinstance(case, AgentTestCase):
        return evaluate_agent([case], **kwargs)
    raise TypeError(
        f"Unsupported test case type for assert_test: {type(case).__name__}. "
        "Expected RAGTestCase, TextTestCase, or AgentTestCase."
    )
