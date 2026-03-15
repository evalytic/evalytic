"""fal.ai image generation with parallel execution."""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx

from ..exceptions import GenerationError
from .registry import ModelEntry

logger = logging.getLogger("evalytic")


@dataclass
class GenerationResult:
    """Result of generating a single image."""

    image_url: str = ""
    generation_time_ms: int = 0
    generation_cost_usd: float = 0.0
    model: str = ""
    item_id: str = ""
    status: str = "success"  # "success" | "failed"
    error: str = ""
    local_path: str = ""
    retried: bool = False  # True if this result came from a retry attempt


def generate_single(
    entry: ModelEntry,
    arguments: dict[str, Any],
    item_id: str,
    cache_dir: Path | None = None,
    timeout: int = 300,
    max_retries: int = 1,
) -> GenerationResult:
    """Generate one image via ``fal_client.subscribe()``.

    On failure, retries up to *max_retries* times with a brief pause.
    Downloads the result to *cache_dir* if provided.
    """
    try:
        import fal_client
    except ImportError:
        raise GenerationError(
            "Image generation requires fal-client. "
            "Install with: pip install evalytic[generation]"
        ) from None

    last_error = ""
    total_start = time.monotonic()

    for attempt in range(1 + max_retries):
        start = time.monotonic()
        try:
            result = fal_client.subscribe(
                entry.endpoint,
                arguments=arguments,
                client_timeout=timeout,
            )

            # fal.ai models return images in different shapes
            if "images" in result and result["images"]:
                image_url = result["images"][0]["url"]
            elif "image" in result and isinstance(result["image"], dict):
                image_url = result["image"]["url"]
            else:
                raise GenerationError(f"Unexpected response shape: {list(result.keys())}")

            elapsed_ms = int((time.monotonic() - total_start) * 1000)

            local_path = ""
            if cache_dir is not None:
                local_path = _download_image(image_url, cache_dir, entry.short_name, item_id)

            return GenerationResult(
                image_url=image_url,
                generation_time_ms=elapsed_ms,
                generation_cost_usd=entry.cost_per_image,
                model=entry.short_name,
                item_id=item_id,
                local_path=local_path,
                retried=attempt > 0,
            )
        except Exception as exc:
            last_error = str(exc)
            elapsed_ms = int((time.monotonic() - start) * 1000)
            if attempt < max_retries:
                logger.warning(
                    "[%s/%s] attempt %d failed (%ds): %s — retrying",
                    entry.short_name, item_id, attempt + 1,
                    elapsed_ms // 1000, last_error[:120],
                )
                time.sleep(2)
            else:
                logger.error(
                    "[%s/%s] failed after %d attempt(s) (%ds): %s",
                    entry.short_name, item_id, attempt + 1,
                    int((time.monotonic() - total_start)),
                    last_error[:200],
                )

    total_elapsed_ms = int((time.monotonic() - total_start) * 1000)
    return GenerationResult(
        model=entry.short_name,
        item_id=item_id,
        status="failed",
        error=last_error,
        generation_time_ms=total_elapsed_ms,
        retried=max_retries > 0,
    )


def generate_batch(
    entry: ModelEntry,
    items: list[dict[str, Any]],
    concurrency: int = 4,
    cache_dir: Path | None = None,
    timeout: int = 300,
    on_progress: Callable[[GenerationResult], None] | None = None,
) -> list[GenerationResult]:
    """Generate images for all *items* with bounded parallelism.

    Each item dict must have ``"item_id"`` and ``"arguments"`` keys.
    """
    results: list[GenerationResult] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                generate_single,
                entry,
                item["arguments"],
                item["item_id"],
                cache_dir,
                timeout,
            ): item["item_id"]
            for item in items
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if on_progress:
                on_progress(result)
    return results


def _download_image(url: str, cache_dir: Path, model: str, item_id: str) -> str:
    """Download an image to the local cache. Returns the local path."""
    model_dir = cache_dir / model
    model_dir.mkdir(parents=True, exist_ok=True)
    ext = ".jpg"
    if ".png" in url.lower():
        ext = ".png"
    elif ".webp" in url.lower():
        ext = ".webp"
    path = model_dir / f"{item_id}{ext}"
    resp = httpx.get(url, follow_redirects=True, timeout=30)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return str(path)
