"""Runners for text and RAG evaluation."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from ..exceptions import ValidationError
from ..judge.consensus import merge_median_or_average, resolve_metric_consensus
from .embeddings import resolve_embedder
from .judge import TextJudge
from .metrics import METRIC_REGISTRY
from .types import (
    MetricEvalReport,
    MetricEvalResult,
    MetricResult,
    RAGTestCase,
    RetrievedChunk,
    TextTestCase,
)

DEFAULT_RAG_METRICS = ["faithfulness", "answer_relevancy"]
DEFAULT_TEXT_METRICS = ["factual_correctness", "semantic_similarity"]


def load_cases_from_dataset(path: str, expected_type: str) -> list[RAGTestCase | TextTestCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    dataset_type = detect_dataset_type(data)
    if dataset_type != expected_type:
        raise ValidationError(
            f"Dataset type mismatch: expected {expected_type!r}, found {dataset_type!r}."
        )
    items = data.get("items", [])
    if expected_type == "rag":
        return [_rag_case_from_item(item) for item in items]
    if expected_type == "text":
        return [_text_case_from_item(item) for item in items]
    raise ValidationError(f"Unsupported dataset type for text runner: {expected_type!r}")


def detect_dataset_type(data: Any) -> str:
    if isinstance(data, dict):
        if data.get("type"):
            return str(data["type"])
        if data.get("pipeline"):
            return str(data["pipeline"])
        if "inputs" in data:
            return "img2img"
        if "prompts" in data:
            return "text2img"
        items = data.get("items")
        sample = items[0] if isinstance(items, list) and items else data
    elif isinstance(data, list) and data:
        sample = data[0]
    else:
        return "text2img"

    if isinstance(sample, dict):
        if "query" in sample and "response" in sample:
            return "rag"
        if "input" in sample and "final_output" in sample:
            return "agent"
        if "input" in sample and "output" in sample:
            return "text"
        if "image_url" in sample:
            return "img2img"
        if "prompt" in sample:
            return "text2img"
    if isinstance(sample, str):
        return "text2img"
    return "text2img"


def evaluate_rag(
    cases: list[RAGTestCase],
    *,
    metric_ids: list[str] | None = None,
    judge: str = "gemini-2.5-flash",
    judges: list[str] | None = None,
    base_url: str | None = None,
    config: dict[str, Any] | None = None,
) -> MetricEvalReport:
    metric_ids = metric_ids or DEFAULT_RAG_METRICS
    return _evaluate_cases(
        cases=cases,
        eval_type="rag",
        metric_ids=metric_ids,
        judge=judge,
        judges=judges,
        base_url=base_url,
        config=config,
    )


def evaluate_text(
    cases: list[TextTestCase],
    *,
    metric_ids: list[str] | None = None,
    judge: str = "gemini-2.5-flash",
    judges: list[str] | None = None,
    base_url: str | None = None,
    config: dict[str, Any] | None = None,
) -> MetricEvalReport:
    metric_ids = metric_ids or DEFAULT_TEXT_METRICS
    return _evaluate_cases(
        cases=cases,
        eval_type="text",
        metric_ids=metric_ids,
        judge=judge,
        judges=judges,
        base_url=base_url,
        config=config,
    )


def _evaluate_cases(
    *,
    cases: list[Any],
    eval_type: str,
    metric_ids: list[str],
    judge: str,
    judges: list[str] | None,
    base_url: str | None,
    config: dict[str, Any] | None,
) -> MetricEvalReport:
    judge_names = [name.strip() for name in (judges or []) if name.strip()]
    consensus_mode = bool(judge_names)
    primary_judge = judge_names[0] if judge_names else judge
    requested_metrics = [_create_metric(metric_id) for metric_id in metric_ids]
    embedder = None
    if any(metric.requires_embeddings for metric in requested_metrics):
        embedder = resolve_embedder(config)

    judge_cache: dict[str, TextJudge] = {}

    def get_judge(judge_name: str) -> TextJudge:
        if judge_name not in judge_cache:
            judge_cache[judge_name] = TextJudge(judge=judge_name, base_url=base_url)
        return judge_cache[judge_name]

    try:
        results: list[MetricEvalResult] = []
        for case in cases:
            started = time.perf_counter()
            metrics: list[MetricResult] = []
            for metric in requested_metrics:
                if consensus_mode:
                    result = resolve_metric_consensus(
                        judge_names,
                        scorer=lambda judge_name, metric=metric: metric.score(
                            case,
                            judge=get_judge(judge_name) if metric.requires_judge else None,
                            embedder=embedder,
                        ),
                        score_of=lambda item: item.score,
                        merge=_merge_metric_results,
                    )
                else:
                    result = metric.score(
                        case,
                        judge=get_judge(primary_judge) if metric.requires_judge else None,
                        embedder=embedder,
                    )
                metrics.append(result)
            case_id = _case_id(case)
            results.append(
                MetricEvalResult(
                    case_id=case_id,
                    test_case=case,
                    metrics=metrics,
                    total_cost=sum(metric.cost for metric in metrics),
                    duration_ms=int((time.perf_counter() - started) * 1000),
                )
            )
        return MetricEvalReport(
            eval_type=eval_type,
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


def _merge_metric_results(
    values: list[MetricResult],
    judge_names: list[str],
    agreement: str,
) -> MetricResult:
    def build_judge_reasons(
        values: list[MetricResult],
        judge_names: list[str],
        chosen: MetricResult,
    ) -> dict[str, Any] | None:
        reasons = {
            judge_name: value.reason
            for judge_name, value in zip(judge_names, values)
            if value.reason
        }
        return {"judge_reasons": reasons} if reasons else None

    return merge_median_or_average(
        values,
        judge_names,
        agreement,
        metric_result_cls=MetricResult,
        extra_details_builder=build_judge_reasons,
    )


def _create_metric(metric_id: str):
    metric_cls = METRIC_REGISTRY.get(metric_id)
    if metric_cls is None:
        valid = ", ".join(sorted(METRIC_REGISTRY))
        raise ValidationError(f"Unknown metric {metric_id!r}. Valid: {valid}")
    return metric_cls()


def _rag_case_from_item(item: dict[str, Any]) -> RAGTestCase:
    contexts = [
        RetrievedChunk(
            text=context["text"] if isinstance(context, dict) else str(context),
            chunk_id=context.get("chunk_id") if isinstance(context, dict) else None,
            source=context.get("source") if isinstance(context, dict) else None,
            rank=context.get("rank") if isinstance(context, dict) else None,
            retrieval_score=context.get("retrieval_score") if isinstance(context, dict) else None,
            metadata=context.get("metadata") if isinstance(context, dict) else None,
        )
        for context in item.get("contexts", [])
    ]
    return RAGTestCase(
        query=item["query"],
        response=item["response"],
        contexts=contexts,
        reference=item.get("reference"),
        metadata=item.get("metadata"),
    )


def _text_case_from_item(item: dict[str, Any]) -> TextTestCase:
    return TextTestCase(
        input=item["input"],
        output=item["output"],
        expected=item.get("expected"),
        criteria=item.get("criteria"),
        context=item.get("context"),
        metadata=item.get("metadata"),
    )


def _case_id(case: Any) -> str:
    if isinstance(case, RAGTestCase):
        payload = f"rag::{case.query}::{case.response}"
    elif isinstance(case, TextTestCase):
        payload = f"text::{case.input}::{case.output}"
    else:
        payload = repr(case)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
