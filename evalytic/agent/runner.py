"""Agent evaluation runner."""

from __future__ import annotations

import hashlib
import time
from typing import Any

from ..exceptions import ValidationError
from ..judge.consensus import merge_median_or_average, resolve_metric_consensus
from ..text.embeddings import cosine_similarity, resolve_embedder
from ..text.judge import TextJudge
from ..text.metrics.base import last_cost_of
from ..text.types import MetricEvalReport, MetricEvalResult, MetricResult
from .types import AgentTestCase


def evaluate_agent(
    cases: list[AgentTestCase],
    *,
    judge: str = "gemini-2.5-flash",
    judges: list[str] | None = None,
    base_url: str | None = None,
    config: dict[str, Any] | None = None,
    force_embedding: bool = False,
) -> MetricEvalReport:
    judge_names = [name.strip() for name in (judges or []) if name.strip()]
    consensus_mode = bool(judge_names)
    primary_judge = judge_names[0] if judge_names else judge
    judge_cache: dict[str, TextJudge] = {}

    embedder = None
    embedder_resolved = False

    def get_embedder() -> Any:
        nonlocal embedder, embedder_resolved
        if embedder_resolved:
            return embedder
        embedder_resolved = True
        try:
            embedder = resolve_embedder(config)
        except Exception:
            if force_embedding:
                raise
            embedder = None
        return embedder

    def get_judge(judge_name: str) -> TextJudge:
        if judge_name not in judge_cache:
            judge_cache[judge_name] = TextJudge(judge=judge_name, base_url=base_url)
        return judge_cache[judge_name]

    try:
        results: list[MetricEvalResult] = []
        for case in cases:
            started = time.perf_counter()
            metrics: list[MetricResult] = []
            for metric_name in ("tool_call_accuracy", "goal_accuracy", "step_efficiency"):
                if consensus_mode and metric_name == "goal_accuracy":
                    metric = resolve_metric_consensus(
                        judge_names,
                        scorer=lambda judge_name: _goal_accuracy(
                            case, get_judge(judge_name), get_embedder()
                        ),
                        score_of=lambda item: item.score,
                        merge=_merge_metric_results,
                    )
                elif metric_name == "tool_call_accuracy":
                    metric = _tool_call_accuracy(case)
                elif metric_name == "goal_accuracy":
                    metric = _goal_accuracy(case, get_judge(primary_judge), get_embedder())
                else:
                    metric = _step_efficiency(case)
                metrics.append(metric)
            results.append(
                MetricEvalResult(
                    case_id=_case_id(case),
                    test_case=case,
                    metrics=metrics,
                    total_cost=sum(metric.cost for metric in metrics),
                    duration_ms=int((time.perf_counter() - started) * 1000),
                )
            )
        return MetricEvalReport(
            eval_type="agent",
            judge=(
                f"consensus({','.join(judge_names)})"
                if consensus_mode
                else primary_judge
            ),
            judges=judge_names,
            consensus_mode=consensus_mode,
            results=results,
        )
    finally:
        if embedder is not None:
            embedder.close()
        for judge_instance in judge_cache.values():
            judge_instance.close()


def _tool_call_accuracy(case: AgentTestCase) -> MetricResult:
    expected = ((case.metadata or {}).get("expected_tool_calls") or [])
    actual = [call.name for call in (case.tool_calls or [])]
    if not expected:
        score = 1.0 if not actual else max(0.0, 1.0 - (len(actual) - 1) * 0.1)
        return MetricResult(
            metric_id="tool_call_accuracy",
            score=score,
            reason="No expected tool calls provided; applied lightweight heuristic.",
            details={"actual_tool_calls": actual},
        )
    expected_set = set(expected)
    actual_set = set(actual)
    if not expected_set and not actual_set:
        score = 1.0
    elif not actual_set:
        score = 0.0
    else:
        true_positive = len(expected_set & actual_set)
        precision = true_positive / len(actual_set) if actual_set else 0.0
        recall = true_positive / len(expected_set) if expected_set else 0.0
        score = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return MetricResult(
        metric_id="tool_call_accuracy",
        score=score,
        reason="Compared executed tool names against expected tool names.",
        details={"expected_tool_calls": expected, "actual_tool_calls": actual},
    )


def _goal_accuracy(case: AgentTestCase, judge: TextJudge, embedder: Any) -> MetricResult:
    if case.expected_output and embedder is not None:
        vectors = embedder.embed_texts([case.final_output, case.expected_output])
        score = cosine_similarity(vectors[0], vectors[1])
        return MetricResult(
            metric_id="goal_accuracy",
            score=score,
            reason="Compared final output to expected output using embedding similarity.",
            judge=judge.judge_string,
            cost=last_cost_of(embedder),
        )

    expected_block = (
        f"\n\nExpected final output:\n{case.expected_output}" if case.expected_output else ""
    )
    prompt = (
        "Evaluate whether the agent achieved the user's goal.\n"
        'Return JSON like {"score": 0.0, "reason":"..."}.\n\n'
        f"User goal:\n{case.input}\n\n"
        f"Agent final output:\n{case.final_output}"
        f"{expected_block}\n\n"
        f"Tool calls:\n{[call.name for call in (case.tool_calls or [])]}"
    )
    raw = judge.complete_json(prompt)
    return MetricResult(
        metric_id="goal_accuracy",
        score=max(0.0, min(1.0, float(raw.get("score", 0.0)))),
        reason=raw.get("reason") or "LLM-judged goal accuracy.",
        judge=judge.judge_string,
        cost=last_cost_of(judge),
    )


def _step_efficiency(case: AgentTestCase) -> MetricResult:
    expected_max_steps = (case.metadata or {}).get("expected_max_steps")
    actual_steps = len(case.tool_calls or [])
    if actual_steps == 0:
        score = 1.0
    elif expected_max_steps:
        score = min(float(expected_max_steps) / actual_steps, 1.0)
    else:
        score = max(0.0, 1.0 - max(0, actual_steps - 3) * 0.15)
    return MetricResult(
        metric_id="step_efficiency",
        score=score,
        reason="Measured tool-step efficiency from the observed tool call count.",
        details={"actual_steps": actual_steps, "expected_max_steps": expected_max_steps},
    )


def _merge_metric_results(values: list[MetricResult], judge_names: list[str], agreement: str) -> MetricResult:
    return merge_median_or_average(
        values,
        judge_names,
        agreement,
        metric_result_cls=MetricResult,
    )


def _case_id(case: AgentTestCase) -> str:
    payload = f"agent::{case.input}::{case.final_output}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
