"""Shared utilities for judge infrastructure."""

from __future__ import annotations

import json
import time
from typing import Any

from ..exceptions import JudgeError


def parse_response(text: str) -> dict[str, Any]:
    """Parse JSON from LLM response, with markdown fence fallback."""
    try:
        return json.loads(text)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        if "```json" in text:
            json_str = text.split("```json")[1].split("```")[0].strip()
            return json.loads(json_str)  # type: ignore[no-any-return]
        if "```" in text:
            json_str = text.split("```")[1].split("```")[0].strip()
            return json.loads(json_str)  # type: ignore[no-any-return]
        raise


def guess_mime(url: str) -> str:
    """Guess MIME type from URL/path extension."""
    lower = url.lower().split("?")[0]
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".webp"):
        return "image/webp"
    if lower.endswith(".gif"):
        return "image/gif"
    return "image/jpeg"


def extract_response_cost(response: Any) -> float:
    """Extract per-request cost from provider response metadata when available."""
    headers = getattr(response, "headers", None) or {}
    raw = headers.get("X-Parel-Cost") or headers.get("x-parel-cost") or "0"
    try:
        return float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0


def call_with_retries(
    call_fn: Any,
    *args: Any,
    max_retries: int = 3,
    **kwargs: Any,
) -> dict[str, Any]:
    """Call a provider function with exponential backoff retries."""
    last_error: Exception | None = None
    for attempt in range(max_retries):
        if attempt > 0:
            time.sleep(2**attempt)
        try:
            return call_fn(*args, **kwargs)
        except Exception as exc:
            last_error = exc
    raise JudgeError(
        f"Judge failed after {max_retries} retries: {last_error}"
    ) from last_error  # type: ignore[misc]
