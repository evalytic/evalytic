"""Main bench orchestrator: generate -> score -> aggregate -> report."""

from __future__ import annotations

import platform
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import evalytic

from concurrent.futures import ThreadPoolExecutor, as_completed

from ..exceptions import ValidationError
from .consensus import ConsensusJudge
from .cost import build_cost_breakdown
from .generator import GenerationResult, generate_single
from .judge import DIMENSION_CONFIG, Judge
from .registry import ModelEntry, _USER_OVERRIDES, detect_cost, detect_image_field, resolve_model
from .types import (
    BenchItem,
    BenchReport,
    CorrelationStat,
    DimensionResult,
    ImageResult,
    ModelSummary,
    RunError,
    ScoringConfig,
)

# Keywords that trigger text_rendering auto-detection
_TEXT_KEYWORDS = {"text", "word", "letter", "write", "say", "font", "type", "sign"}


def run_bench(
    models: list[str] | None = None,
    prompts: list[dict[str, Any]] | None = None,
    inputs: list[dict[str, Any]] | None = None,
    pre_images: list[dict[str, Any]] | None = None,
    judge: str = "gemini-2.5-flash",
    judges: list[str] | None = None,
    dimensions: list[str] | None = None,
    metrics: list[str] | None = None,
    concurrency: int = 4,
    seed: int | None = None,
    image_size: str | None = None,
    fal_params: dict[str, Any] | None = None,
    cache_dir: str | None = None,
    timeout: int = 600,
    name: str | None = None,
    scoring_config: ScoringConfig | None = None,
    judge_url: str | None = None,
    no_judge: bool = False,
    on_generation_progress: Any = None,
    on_scoring_progress: Any = None,
    verbose: bool = False,
) -> BenchReport:
    """Run a bench: generate (or use pre-existing) images, score, aggregate, return report.

    When *pre_images* is provided, the generation phase is skipped entirely.
    Each item in *pre_images* maps model labels to image URLs/paths::

        pre_images=[
            {
                "prompt": "A cat wearing a top hat",
                "images": {
                    "midjourney-v6": "https://cdn.midjourney.com/abc.png",
                    "dalle3": "./images/dalle3/cat.png",
                }
            },
        ]

    For img2img scoring, include ``input_image`` in each item.
    """
    import sys

    def _log(msg: str) -> None:
        if verbose:
            sys.stderr.write(f"[evalytic] {msg}\n")

    start_time = time.monotonic()
    run_id = uuid.uuid4().hex[:8]
    run_name = name or f"bench-{run_id}"
    errors: list[RunError] = []

    # ---- PRE-IMAGES MODE (score-only, no generation) ----
    skip_generation = pre_images is not None

    if skip_generation:
        items, model_names, pipeline = _parse_pre_images(pre_images)  # type: ignore[arg-type]
        models = models or model_names
    else:
        if not models:
            raise ValidationError("Either models or pre_images must be provided.")
        # Determine pipeline
        pipeline = _detect_pipeline(prompts, inputs)
        # Build bench items
        items = _build_items(prompts, inputs, pipeline)

    _log(f"Pipeline: {pipeline}")

    # Auto-detect dimensions
    if dimensions is None:
        dimensions = _auto_detect_dimensions(pipeline, items)

    # Validate dimensions
    unknown = set(dimensions) - set(DIMENSION_CONFIG)
    if unknown:
        raise ValidationError(
            f"Unknown dimensions: {sorted(unknown)}. "
            f"Valid: {list(DIMENSION_CONFIG)}"
        )

    _log(f"Dimensions: {dimensions}")

    if skip_generation:
        # Build GenerationResult objects from pre-existing images
        entries = [
            ModelEntry(short_name=m, endpoint="", pipeline=pipeline, cost_per_image=0.0)
            for m in models
        ]
        all_gen_results: dict[str, list[GenerationResult]] = {m: [] for m in models}
        for item in items:
            for model_name, image_path in item.get("_images", {}).items():
                all_gen_results[model_name].append(
                    GenerationResult(
                        image_url=image_path,
                        model=model_name,
                        item_id=item["item_id"],
                    )
                )
        _log(f"Pre-images mode: {len(models)} models, {len(items)} items")
    else:
        # Resolve models
        entries = [resolve_model(m) for m in models]

        # Auto-detect image field and cost for models via fal.ai.
        # Skip models that the user explicitly configured via register_model().
        updated: list[ModelEntry] = []
        for entry in entries:
            if entry.short_name in _USER_OVERRIDES:
                updated.append(entry)
                continue
            changes: dict[str, object] = {}
            # Image field detection (img2img only)
            if pipeline == "img2img":
                detected_field = detect_image_field(
                    entry.endpoint, registry_default=entry.image_field
                )
                if detected_field != entry.image_field:
                    changes["image_field"] = detected_field
            # Cost detection (all pipelines)
            detected_cost = detect_cost(
                entry.endpoint, registry_default=entry.cost_per_image
            )
            if detected_cost is not None and detected_cost != entry.cost_per_image:
                changes["cost_per_image"] = detected_cost
            if changes:
                entry = ModelEntry(
                    short_name=entry.short_name,
                    endpoint=entry.endpoint,
                    pipeline=entry.pipeline,
                    cost_per_image=changes.get("cost_per_image", entry.cost_per_image),  # type: ignore[arg-type]
                    image_field=changes.get("image_field", entry.image_field),  # type: ignore[arg-type]
                )
            updated.append(entry)
        entries = updated

        _log(f"Models: {[e.short_name for e in entries]}")

        # Set up cache dir
        cache_root = Path(cache_dir or "~/.evalytic/cache").expanduser()
        cache_path = cache_root / run_name
        try:
            cache_path.mkdir(parents=True, exist_ok=True)
        except OSError:
            fallback_root = Path(tempfile.gettempdir()) / "evalytic-cache"
            cache_path = fallback_root / run_name
            cache_path.mkdir(parents=True, exist_ok=True)

        # ---- GENERATE (cross-model parallel) ----
        all_tasks: list[tuple[Any, dict[str, Any]]] = []
        for entry in entries:
            gen_items = _build_generation_items(items, entry, pipeline, seed, image_size, fal_params)
            for gi in gen_items:
                all_tasks.append((entry, gi))

        all_gen_results = {e.short_name: [] for e in entries}
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(
                    generate_single, entry, gi["arguments"], gi["item_id"], cache_path, timeout,
                ): entry.short_name
                for entry, gi in all_tasks
            }
            for future in as_completed(futures):
                model_name = futures[future]
                result = future.result()
                all_gen_results[model_name].append(result)
                retry_tag = " [retried]" if getattr(result, "retried", False) else ""
                time_str = f"{result.generation_time_ms / 1000:.1f}s"
                if result.status == "failed":
                    _log(f"Generated {model_name}/{result.item_id}: FAILED ({time_str}){retry_tag} — {result.error[:120]}")
                else:
                    _log(f"Generated {model_name}/{result.item_id}: OK ({time_str}){retry_tag}")
                if result.status == "failed" and result.error:
                    errors.append(RunError(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        level="error",
                        phase="generation",
                        model=model_name,
                        item_id=result.item_id,
                        dimension="",
                        message=result.error,
                    ))
                if on_generation_progress:
                    on_generation_progress(result)

    # ---- SCORE (parallel) ----
    bench_items: list[BenchItem] = []
    judge_request_count = 0
    is_consensus = bool(judges and len(judges) >= 2)

    score_results: dict[tuple[str, str], list[DimensionResult]] = {}

    if no_judge:
        # Skip VLM judge entirely — metrics-only mode
        dimensions = []
        judge = ""
        judges = None
        is_consensus = False
        _log("No-judge mode: skipping VLM scoring")

        # Build BenchItems without scores
        for item in items:
            bench_item = BenchItem(
                item_id=item["item_id"],
                prompt=item.get("prompt", ""),
                image_url=item.get("image_url", ""),
                instruction=item.get("instruction", ""),
                tags=item.get("tags", []),
            )
            for entry in entries:
                model_name = entry.short_name
                gen = _find_gen_result(all_gen_results[model_name], item["item_id"])
                if gen is None or gen.status == "failed":
                    bench_item.results[model_name] = ImageResult(
                        model=model_name,
                        image_url="",
                        status="failed",
                        error=gen.error if gen else "Generation result not found",
                    )
                else:
                    bench_item.results[model_name] = ImageResult(
                        model=model_name,
                        image_url=gen.image_url,
                        image_local=gen.local_path,
                        generation_time_ms=gen.generation_time_ms,
                        generation_cost_usd=gen.generation_cost_usd,
                        scores=[],
                    )
            bench_items.append(bench_item)
    else:
        # Collect scorable (gen, item) pairs
        score_tasks: list[tuple[GenerationResult, dict[str, Any]]] = []
        for item in items:
            for entry in entries:
                gen = _find_gen_result(all_gen_results[entry.short_name], item["item_id"])
                if gen and gen.status != "failed":
                    score_tasks.append((gen, item))

        # Create judge instance (single or consensus)
        if is_consensus:
            scorer: Judge | ConsensusJudge = ConsensusJudge(judges, base_url=judge_url)
        else:
            scorer = Judge(judge=judge, base_url=judge_url)

        # Score in parallel
        try:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {
                    pool.submit(
                        _score_single, scorer, gen, item, dimensions,
                    ): (gen.model, gen.item_id)
                    for gen, item in score_tasks
                }
                for future in as_completed(futures):
                    model_name, item_id = futures[future]
                    try:
                        _, _, dim_results = future.result()
                        score_results[(model_name, item_id)] = dim_results
                        judge_request_count += len(dimensions)
                        _log(f"Scored {model_name}/{item_id}: "
                             + ", ".join(f"{d.dimension}={d.score}" for d in dim_results))
                    except Exception as exc:
                        score_results[(model_name, item_id)] = []
                        _log(f"Scoring failed {model_name}/{item_id}: {exc}")
                        errors.append(RunError(
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            level="error",
                            phase="scoring",
                            model=model_name,
                            item_id=item_id,
                            dimension="",
                            message=str(exc),
                        ))
                        if verbose:
                            import traceback
                            traceback.print_exc(file=sys.stderr)
                    if on_scoring_progress:
                        on_scoring_progress(model_name, item_id)

            # Build BenchItems from results
            for item in items:
                bench_item = BenchItem(
                    item_id=item["item_id"],
                    prompt=item.get("prompt", ""),
                    image_url=item.get("image_url", ""),
                    instruction=item.get("instruction", ""),
                    tags=item.get("tags", []),
                )
                for entry in entries:
                    model_name = entry.short_name
                    gen = _find_gen_result(all_gen_results[model_name], item["item_id"])
                    if gen is None or gen.status == "failed":
                        bench_item.results[model_name] = ImageResult(
                            model=model_name,
                            image_url="",
                            status="failed",
                            error=gen.error if gen else "Generation result not found",
                        )
                    else:
                        bench_item.results[model_name] = ImageResult(
                            model=model_name,
                            image_url=gen.image_url,
                            image_local=gen.local_path,
                            generation_time_ms=gen.generation_time_ms,
                            generation_cost_usd=gen.generation_cost_usd,
                            scores=score_results.get((model_name, gen.item_id), []),
                        )
                bench_items.append(bench_item)
        finally:
            scorer.close()

    # ---- METRICS (optional) ----
    metrics = metrics or []
    if metrics:
        from .metrics import METRICS_AVAILABLE, compute_metrics

        # Sharpness needs no torch — always available.  Others need METRICS_AVAILABLE.
        heavy_metrics = [m for m in metrics if m != "sharpness"]
        has_sharpness = "sharpness" in metrics
        run_metrics: list[str] = []
        if METRICS_AVAILABLE:
            run_metrics = list(metrics)
        elif has_sharpness:
            run_metrics = ["sharpness"]

        if run_metrics:
            source_items = inputs if inputs else (prompts or [])
            compute_metrics(bench_items, run_metrics, pipeline, source_items, cache_dir=cache_dir)

    # ---- AGGREGATE ----
    # Use default scoring config when metrics are active but no config given
    effective_scoring_config = scoring_config
    if effective_scoring_config is None and metrics:
        effective_scoring_config = ScoringConfig.default()

    summary: dict[str, ModelSummary] = {}
    gen_costs: dict[str, float] = {}
    gen_count = 0

    for entry in entries:
        model_name = entry.short_name
        ms = _aggregate_model_summary(
            model_name, bench_items, dimensions, effective_scoring_config,
        )
        summary[model_name] = ms
        gen_costs[model_name] = ms.total_generation_cost_usd
        gen_count += ms.item_count

    # ---- CORRELATION (metrics vs VLM dimensions) ----
    correlations: list[CorrelationStat] = []
    if metrics:
        from .metrics import METRICS_AVAILABLE, compute_correlation

        if METRICS_AVAILABLE:
            metric_dim_pairs = []
            if "clip" in metrics and pipeline == "text2img":
                metric_dim_pairs.append(("clip_score", "prompt_adherence"))
            if "lpips" in metrics and pipeline == "img2img":
                metric_dim_pairs.append(("lpips", "input_fidelity"))
            if "face" in metrics and pipeline == "img2img":
                metric_dim_pairs.append(("face_similarity", "identity_preservation"))

            for m_name, d_name in metric_dim_pairs:
                if d_name in dimensions:
                    corr = compute_correlation(bench_items, m_name, d_name)
                    if corr is not None:
                        correlations.append(corr)

    ranking = sorted(
        [(m, s.overall_score) for m, s in summary.items()],
        key=lambda p: p[1],
        reverse=True,
    )
    winner = ranking[0][0] if ranking else ""

    # Efficiency ranking (score per dollar — higher is better)
    efficiency_ranking = sorted(
        [(m, s.score_per_dollar) for m, s in summary.items() if s.score_per_dollar > 0],
        key=lambda p: p[1],
        reverse=True,
    )
    best_value = efficiency_ranking[0][0] if efficiency_ranking else ""

    # Get per-provider request counts for consensus mode
    consensus_request_counts: dict[str, int] | None = None
    if is_consensus and isinstance(scorer, ConsensusJudge):
        consensus_request_counts = scorer.request_counts
        judge_request_count = sum(consensus_request_counts.values())

    cost = build_cost_breakdown(
        gen_costs, gen_count, judge_request_count, judge,
        judges=judges if is_consensus else None,
        judge_request_counts=consensus_request_counts,
    )
    duration = time.monotonic() - start_time

    return BenchReport(
        name=run_name,
        models=models,
        judge=judge,
        pipeline=pipeline,
        judges=judges or ([judge] if judge else []),
        consensus_mode=is_consensus,
        dimensions=dimensions,
        items=bench_items,
        summary=summary,
        winner=winner,
        ranking=ranking,
        efficiency_ranking=efficiency_ranking,
        best_value=best_value,
        cost=cost,
        correlations=correlations,
        duration_seconds=round(duration, 1),
        created_at=datetime.now(timezone.utc).isoformat(),
        config={
            "models": models,
            "judge": judge,
            **({"judges": judges} if judges else {}),
            "dimensions": dimensions,
            "metrics": metrics,
            "pipeline": pipeline,
            "seed": seed,
            "image_size": image_size,
            "concurrency": concurrency,
            **({"scoring_config": effective_scoring_config.to_dict()} if effective_scoring_config else {}),
            **({"dimension_weights": dict(effective_scoring_config.dimension_weights)} if effective_scoring_config and effective_scoring_config.dimension_weights else {}),
        },
        metadata={
            "evalytic_version": evalytic.__version__,
            "python_version": platform.python_version(),
            "platform": f"{platform.system().lower()}-{platform.machine()}",
        },
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_pre_images(
    pre_images: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], str]:
    """Parse pre_images input into (items, model_names, pipeline).

    Each entry in *pre_images* must have an ``images`` dict mapping model
    labels to image URLs or local paths.  Optional fields: ``prompt``,
    ``input_image`` (triggers img2img), ``instruction``, ``tags``.
    """
    if not pre_images:
        raise ValidationError("pre_images list is empty.")

    all_models: dict[str, None] = {}  # ordered set
    items: list[dict[str, Any]] = []
    has_input_image = False

    for i, raw in enumerate(pre_images):
        images = raw.get("images")
        if not images or not isinstance(images, dict):
            raise ValidationError(
                f"Item {i + 1}: 'images' dict is required. "
                "Each key is a model label, each value an image URL or path."
            )
        item_id = raw.get("item_id", f"item-{i + 1:03d}")
        item: dict[str, Any] = {
            "item_id": item_id,
            "_images": images,  # preserved for GenerationResult construction
        }
        if raw.get("input_image"):
            item["image_url"] = raw["input_image"]
            has_input_image = True
        if raw.get("prompt"):
            item["prompt"] = raw["prompt"]
        if raw.get("instruction"):
            item["instruction"] = raw["instruction"]
        elif raw.get("prompt") and has_input_image:
            item["instruction"] = raw["prompt"]
        item["tags"] = raw.get("tags", [])
        items.append(item)

        for model_name in images:
            all_models[model_name] = None

    pipeline = "img2img" if has_input_image else "text2img"
    return items, list(all_models), pipeline


def _detect_pipeline(
    prompts: list[dict[str, Any]] | None,
    inputs: list[dict[str, Any]] | None,
) -> str:
    """Detect whether this is a text2img or img2img bench."""
    if inputs:
        return "img2img"
    if prompts:
        # Check if items have image_url -> img2img
        for item in prompts:
            if "image_url" in item:
                return "img2img"
        return "text2img"
    raise ValidationError("Either prompts or inputs must be provided.")


def _auto_detect_dimensions(pipeline: str, items: list[dict[str, Any]]) -> list[str]:
    """Choose dimensions based on pipeline type and prompt content."""
    if pipeline == "img2img":
        return ["visual_quality", "input_fidelity", "transformation_quality", "artifact_detection"]

    dims = ["visual_quality", "prompt_adherence"]
    # Check if any prompt mentions text-related keywords
    for item in items:
        prompt_text = item.get("prompt", "").lower()
        if any(kw in prompt_text for kw in _TEXT_KEYWORDS):
            dims.append("text_rendering")
            break
    return dims


def _build_items(
    prompts: list[dict[str, Any]] | None,
    inputs: list[dict[str, Any]] | None,
    pipeline: str,
) -> list[dict[str, Any]]:
    """Normalize prompts/inputs into a unified item list."""
    source = inputs if inputs else (prompts or [])
    items: list[dict[str, Any]] = []
    for i, raw in enumerate(source):
        item_id = raw.get("item_id", f"item-{i + 1:03d}")
        item: dict[str, Any] = {"item_id": item_id}
        if pipeline == "text2img":
            item["prompt"] = raw.get("prompt", "")
        else:
            item["image_url"] = raw.get("image_url", "")
            item["instruction"] = raw.get("instruction", raw.get("prompt", ""))
        item["tags"] = raw.get("tags", [])
        items.append(item)
    return items


def _build_generation_items(
    items: list[dict[str, Any]],
    entry: Any,
    pipeline: str,
    seed: int | None,
    image_size: str | None,
    fal_params: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Build the list of generation request dicts for one model."""
    gen_items: list[dict[str, Any]] = []
    for item in items:
        args: dict[str, Any] = {}
        if pipeline == "text2img":
            args["prompt"] = item["prompt"]
        else:
            url = item["image_url"]
            if entry.image_field == "image_urls":
                args["image_urls"] = [url]
            else:
                args["image_url"] = url
            args["prompt"] = item.get("instruction", "")
        if seed is not None:
            args["seed"] = seed
        if image_size:
            args["image_size"] = image_size
        if fal_params:
            # Per-model overrides
            model_params = fal_params.get(entry.short_name, fal_params)
            args.update(model_params)
        gen_items.append({"item_id": item["item_id"], "arguments": args})
    return gen_items


def _score_single(
    scorer: Judge | ConsensusJudge,
    gen: GenerationResult,
    item: dict[str, Any],
    dimensions: list[str],
) -> tuple[str, str, list[DimensionResult]]:
    """Score one (model, item) pair. Returns (model_name, item_id, dim_results)."""
    dim_results = scorer.score(
        image_url=gen.image_url,
        dimensions=dimensions,
        prompt=item.get("prompt"),
        input_image_url=item.get("image_url"),
    )
    return (gen.model, gen.item_id, dim_results)


def _find_gen_result(
    results: list[GenerationResult], item_id: str
) -> GenerationResult | None:
    for r in results:
        if r.item_id == item_id:
            return r
    return None


def _aggregate_model_summary(
    model: str,
    items: list[BenchItem],
    dimensions: list[str],
    scoring_config: ScoringConfig | None = None,
) -> ModelSummary:
    """Compute average scores for a model across all items."""
    dim_scores: dict[str, list[float]] = {d: [] for d in dimensions}
    dim_confidences: dict[str, list[float]] = {d: [] for d in dimensions}
    metric_scores: dict[str, list[float]] = {}
    agreement_counts: list[str] = []  # track agreement values
    total_cost = 0.0
    total_time = 0
    count = 0
    total_items = 0

    for item in items:
        img_result = item.results.get(model)
        if img_result is None:
            continue
        total_items += 1
        if img_result.status == "failed":
            # Penalize failures: count as 0.0 for all dimensions
            for d in dimensions:
                dim_scores[d].append(0.0)
                dim_confidences[d].append(1.0)
            continue
        count += 1
        total_cost += img_result.generation_cost_usd
        total_time += img_result.generation_time_ms
        for dr in img_result.scores:
            if dr.dimension in dim_scores:
                dim_scores[dr.dimension].append(dr.score)
                dim_confidences[dr.dimension].append(dr.confidence)
            if dr.agreement:
                agreement_counts.append(dr.agreement)
        for mr in img_result.metrics:
            metric_scores.setdefault(mr.metric, []).append(mr.value)

    dim_avgs = {
        d: round(sum(scores) / len(scores), 2) if scores else 0.0
        for d, scores in dim_scores.items()
    }
    # Standard deviation per dimension (population stddev for small N)
    dim_stddevs: dict[str, float] = {}
    for d, scores in dim_scores.items():
        if len(scores) >= 2:
            mean = sum(scores) / len(scores)
            variance = sum((s - mean) ** 2 for s in scores) / len(scores)
            dim_stddevs[d] = round(variance ** 0.5, 2)
        else:
            dim_stddevs[d] = 0.0
    # Overall stddev across all dimension scores
    all_dim_scores = [s for scores in dim_scores.values() for s in scores]
    if len(all_dim_scores) >= 2:
        mean_all = sum(all_dim_scores) / len(all_dim_scores)
        var_all = sum((s - mean_all) ** 2 for s in all_dim_scores) / len(all_dim_scores)
        overall_stddev = round(var_all ** 0.5, 2)
    else:
        overall_stddev = 0.0

    dim_conf_avgs = {
        d: round(sum(confs) / len(confs), 2) if confs else 1.0
        for d, confs in dim_confidences.items()
    }
    all_confs = [c for confs in dim_confidences.values() for c in confs]
    avg_conf = round(sum(all_confs) / len(all_confs), 2) if all_confs else 1.0
    metric_avgs = {
        m: round(sum(vals) / len(vals), 4) if vals else 0.0
        for m, vals in metric_scores.items()
    }
    if scoring_config and scoring_config.dimension_weights:
        weights = scoring_config.resolve_dimension_weights(dimensions)
        vlm_avg = round(sum(dim_avgs.get(d, 0) * weights.get(d, 0) for d in dimensions), 2)
    else:
        all_avgs = [v for v in dim_avgs.values() if v > 0]
        vlm_avg = round(sum(all_avgs) / len(all_avgs), 2) if all_avgs else 0.0

    # Weighted scoring: incorporate metrics into overall
    metric_flags: dict[str, str] = {}
    weighted_overall: float | None = None

    if scoring_config and scoring_config.metric_configs and metric_avgs:
        from .metrics import check_metric_threshold, normalize_metric

        weighted_sum = 0.0
        total_weight = 0.0

        for m_name, m_avg in metric_avgs.items():
            cfg = scoring_config.metric_configs.get(m_name)
            if cfg is None:
                continue

            flag = check_metric_threshold(m_name, m_avg, cfg)
            if flag is not None:
                metric_flags[m_name] = "flagged"
                _attach_flags_to_images(model, items, m_name, cfg)
            else:
                metric_flags[m_name] = "included"
                normalized = normalize_metric(m_avg, *cfg.normalize_range)
                weighted_sum += normalized * cfg.weight
                total_weight += cfg.weight

        # Clamp total weight to avoid negative VLM contribution
        if total_weight > 1.0:
            total_weight = 1.0

        if total_weight > 0:
            if vlm_avg > 0:
                weighted_overall = round(
                    vlm_avg * (1.0 - total_weight) + weighted_sum, 2,
                )
            else:
                # Metrics-only: use weighted metrics as overall (scale to 0-5)
                weighted_overall = round(weighted_sum / total_weight, 2) if total_weight > 0 else 0.0

    overall = weighted_overall if weighted_overall is not None else vlm_avg

    # Metrics-only fallback: when no VLM scores, use raw metric average as overall
    if overall == 0.0 and metric_avgs and not dimensions:
        # Scale normalized metrics to 0-5 range for consistency
        vals = list(metric_avgs.values())
        raw_avg = sum(vals) / len(vals)
        # Metrics like sharpness are already 0-1, scale to 0-5
        overall = round(raw_avg * 5.0, 2) if raw_avg <= 1.0 else round(raw_avg, 2)

    # Efficiency metrics
    cost_per_img = round(total_cost / count, 4) if count > 0 else 0.0
    spd = round(overall / cost_per_img, 1) if cost_per_img > 0 else 0.0

    # Agreement rate (consensus mode): fraction of dimensions with "high" agreement
    agree_rate = 1.0
    if agreement_counts:
        high_count = sum(1 for a in agreement_counts if a == "high")
        agree_rate = round(high_count / len(agreement_counts), 2)

    return ModelSummary(
        model=model,
        overall_score=overall,
        dimension_averages=dim_avgs,
        dimension_confidence=dim_conf_avgs,
        dimension_stddev=dim_stddevs,
        overall_stddev=overall_stddev,
        sample_count=count,
        avg_confidence=avg_conf,
        metric_averages=metric_avgs,
        metric_flags=metric_flags,
        weighted_overall_score=weighted_overall,
        agreement_rate=agree_rate,
        total_generation_cost_usd=round(total_cost, 4),
        total_generation_time_ms=total_time,
        item_count=count,
        total_items=total_items,
        cost_per_image=cost_per_img,
        score_per_dollar=spd,
    )


def _attach_flags_to_images(
    model: str,
    items: list[BenchItem],
    metric_name: str,
    cfg: Any,
) -> None:
    """Attach per-image MetricFlag to ImageResults when metric is below threshold."""
    from .metrics import check_metric_threshold

    for item in items:
        img_result = item.results.get(model)
        if img_result is None or img_result.status != "success":
            continue
        for mr in img_result.metrics:
            if mr.metric == metric_name:
                flag = check_metric_threshold(metric_name, mr.value, cfg)
                if flag is not None:
                    # Avoid duplicates
                    if not any(f.metric == metric_name for f in img_result.flags):
                        img_result.flags.append(flag)
