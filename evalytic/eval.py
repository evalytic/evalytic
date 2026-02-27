"""Single-image evaluation entry point.

Example:
    import evalytic
    evalytic.init(api_key="ek-...")
    result = evalytic.eval_image(
        image="https://example.com/output.png",
        prompt="A sunset over the ocean",
        dimensions=["visual_quality", "prompt_adherence"],
    )
    print(result.display_score)
"""

from __future__ import annotations

import time
from typing import Any

from .client import _get, _post
from .types import DimensionScore, EvalResult


def _parse_eval_result(data: dict[str, Any]) -> EvalResult:
    """Convert a raw API response dict into an ``EvalResult``."""
    dimension_scores: list[DimensionScore] = []
    for s in data.get("scores", []):
        dimension_scores.append(
            DimensionScore(
                dimension=s.get("dimension", ""),
                score=float(s.get("score", 0.0)),
                explanation=s.get("explanation", ""),
                evidence=s.get("evidence", []),
            )
        )

    return EvalResult(
        eval_result_id=data.get("eval_result_id", ""),
        experiment_id=data.get("experiment_id", ""),
        overall_score=float(data.get("overall_score", 0.0)),
        scores=dimension_scores,
        status=data.get("status", "processing"),
        raw=data,
    )


def _poll_eval_result(
    experiment_id: str,
    eval_result_id: str,
    timeout: int,
    interval: float = 2.0,
) -> EvalResult:
    """Poll the API until the evaluation result reaches a terminal state."""
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        data = _get(f"/v1/experiments/{experiment_id}")
        # The experiment response may contain a list of individual results.
        # Try to find ours, or fall back to treating the whole response as the result.
        results_list: list[dict[str, Any]] = data.get("results", [])
        matched: dict[str, Any] | None = None
        for r in results_list:
            if r.get("eval_result_id") == eval_result_id:
                matched = r
                break

        result_data = matched if matched else data
        result = _parse_eval_result(result_data)

        # Propagate top-level status if we fell back to the experiment envelope.
        if matched is None:
            result.eval_result_id = eval_result_id
            result.experiment_id = experiment_id
            result.status = data.get("status", result.status)

        if result.is_complete:
            return result

        time.sleep(interval)

    # Timed out -- return whatever we last had.
    return _parse_eval_result(
        {
            "eval_result_id": eval_result_id,
            "experiment_id": experiment_id,
            "overall_score": 0.0,
            "status": "timeout",
        }
    )


def eval_image(
    image: str,
    prompt: str | None = None,
    input_image: str | None = None,
    baseline_image: str | None = None,
    pipeline_type: str | None = None,
    dimensions: list[str] | None = None,
    project_id: str = "default",
    product_id: str | None = None,
    wait: bool = True,
    timeout: int = 300,
) -> EvalResult:
    """Evaluate a single generated image.

    Parameters
    ----------
    image:
        URL or base-64 data URI of the generated image to evaluate.
    prompt:
        The text prompt used to generate the image (text2img) or the
        transformation instruction (img2img).
    input_image:
        URL of the original input image (img2img pipelines only).
    baseline_image:
        URL of an existing baseline output to compare against.
    pipeline_type:
        ``"text2img"`` or ``"img2img"``.  When *None* the API infers the
        type from the presence of ``input_image``.
    dimensions:
        List of evaluation dimensions to score.  When *None* the API
        uses its default dimension set for the pipeline type.
    project_id:
        Project identifier.  Defaults to ``"default"``.
    product_id:
        Optional product identifier for portfolio-level tracking.
    wait:
        When *True* (default) the call blocks until the evaluation
        completes or *timeout* seconds elapse.
    timeout:
        Maximum seconds to wait when *wait* is *True*.

    Returns
    -------
    EvalResult
        The evaluation result.  If *wait* is *False* the result will have
        ``status="processing"`` and the caller should poll manually.
    """
    payload: dict[str, Any] = {
        "image": image,
        "project_id": project_id,
    }
    if prompt is not None:
        payload["prompt"] = prompt
    if input_image is not None:
        payload["input_image"] = input_image
    if baseline_image is not None:
        payload["baseline_image"] = baseline_image
    if pipeline_type is not None:
        payload["pipeline_type"] = pipeline_type
    if dimensions is not None:
        payload["dimensions"] = dimensions
    if product_id is not None:
        payload["product_id"] = product_id

    data = _post("/v1/evaluate", json=payload)
    result = _parse_eval_result(data)

    if wait and not result.is_complete:
        result = _poll_eval_result(
            experiment_id=result.experiment_id,
            eval_result_id=result.eval_result_id,
            timeout=timeout,
        )

    return result
