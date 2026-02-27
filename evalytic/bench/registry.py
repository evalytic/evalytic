"""fal.ai model registry for bench runs."""

from __future__ import annotations

import difflib
import logging
import os
from dataclasses import dataclass

from ..exceptions import ValidationError

logger = logging.getLogger("evalytic")


@dataclass(frozen=True)
class ModelEntry:
    """A registered fal.ai model."""

    short_name: str
    endpoint: str
    pipeline: str  # "text2img" | "img2img" | "utility"
    cost_per_image: float  # estimated USD
    image_field: str = "image_url"  # some models use "image_urls" (array)

    @property
    def display_cost(self) -> str:
        return f"~${self.cost_per_image:.3f}"


class UnknownModelError(ValidationError):
    """Raised when a model name cannot be resolved."""

    def __init__(self, name: str, suggestion: str | None = None) -> None:
        self.name = name
        self.suggestion = suggestion
        parts = [f"Unknown model {name!r}."]
        if suggestion:
            parts.append(f"Did you mean {suggestion!r}?")
        super().__init__(" ".join(parts))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, ModelEntry] = {
    # text2img
    "flux-schnell": ModelEntry("flux-schnell", "fal-ai/flux/schnell", "text2img", 0.003),
    "flux-dev": ModelEntry("flux-dev", "fal-ai/flux/dev", "text2img", 0.025),
    "flux-pro": ModelEntry("flux-pro", "fal-ai/flux-pro/v1.1", "text2img", 0.05),
    "flux-2-dev": ModelEntry("flux-2-dev", "fal-ai/flux-2/dev", "text2img", 0.025),
    "flux-2-pro": ModelEntry("flux-2-pro", "fal-ai/flux-2/pro", "text2img", 0.05),
    "sdxl": ModelEntry("sdxl", "fal-ai/fast-sdxl", "text2img", 0.01),
    "sd3": ModelEntry("sd3", "fal-ai/stable-diffusion-v3-medium", "text2img", 0.03),
    "recraft-v3": ModelEntry("recraft-v3", "fal-ai/recraft-v3", "text2img", 0.04),
    "imagen-3": ModelEntry("imagen-3", "fal-ai/imagen3", "text2img", 0.04),
    "ideogram-v3": ModelEntry("ideogram-v3", "fal-ai/ideogram/v3", "text2img", 0.08),
    "nano-banana-pro": ModelEntry("nano-banana-pro", "fal-ai/nano-banana-pro", "text2img", 0.15),
    "recraft-v4-pro": ModelEntry("recraft-v4-pro", "fal-ai/recraft/v4/pro/text-to-image", "text2img", 0.25),
    "grok-imagine": ModelEntry("grok-imagine", "xai/grok-imagine-image", "text2img", 0.02),
    "flux-2-flex": ModelEntry("flux-2-flex", "fal-ai/flux-2-flex", "text2img", 0.05),
    "hidream": ModelEntry("hidream", "fal-ai/hidream-i1-full", "text2img", 0.05),
    "seedream": ModelEntry("seedream", "fal-ai/bytedance/seedream/v5/lite/text-to-image", "text2img", 0.03),
    "imagineart": ModelEntry("imagineart", "imagineart/imagineart-1.5-preview/text-to-image", "text2img", 0.04),
    # img2img
    "flux-dev-i2i": ModelEntry("flux-dev-i2i", "fal-ai/flux/dev/image-to-image", "img2img", 0.025),
    "flux-kontext": ModelEntry("flux-kontext", "fal-ai/flux-pro/kontext", "img2img", 0.05),
    "flux-2-kontext-max": ModelEntry("flux-2-kontext-max", "fal-ai/flux-2/kontext/max", "img2img", 0.08),
    "seedream-edit": ModelEntry("seedream-edit", "fal-ai/bytedance/seedream/v5/lite/edit", "img2img", 0.03),
    "bria-edit": ModelEntry("bria-edit", "fal-ai/bria/fibo-edit/edit", "img2img", 0.04),
    "firered-edit": ModelEntry("firered-edit", "fal-ai/firered-image-edit", "img2img", 0.03),
    "reve-edit": ModelEntry("reve-edit", "fal-ai/reve/edit", "img2img", 0.04),
    # utility
    "birefnet": ModelEntry("birefnet", "fal-ai/birefnet/v2", "utility", 0.01),
    "esrgan": ModelEntry("esrgan", "fal-ai/esrgan", "utility", 0.01),
}


_SCHEMA_CACHE: dict[str, str] = {}  # endpoint → image_field
_USER_OVERRIDES: set[str] = set()  # model names with explicit register_model() calls


def detect_image_field(endpoint: str, *, registry_default: str | None = None) -> str:
    """Detect the image field name from fal.ai model schema.

    Queries the fal.ai models API to inspect the OpenAPI schema and
    determine whether the model accepts ``image_url`` or ``image_urls``.

    On failure, falls back to *registry_default* (if the model is in the
    registry) or ``"image_url"``.  A warning is logged suggesting
    ``register_model()`` for manual configuration.
    """
    if endpoint in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[endpoint]

    fallback = registry_default or "image_url"

    try:
        import httpx

        resp = httpx.get(
            "https://api.fal.ai/v1/models",
            params={"endpoint_id": endpoint, "expand": "openapi-3.0"},
            headers={"Authorization": f"Key {os.environ.get('FAL_KEY', '')}"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        models = data.get("models", [])
        if not models:
            _SCHEMA_CACHE[endpoint] = fallback
            return fallback

        schema = models[0].get("openapi", {})
        for _path, methods in schema.get("paths", {}).items():
            for _method, spec in methods.items():
                if "requestBody" not in spec:
                    continue
                rb = spec["requestBody"]["content"]["application/json"]["schema"]
                if "$ref" in rb:
                    ref_name = rb["$ref"].split("/")[-1]
                    props = schema["components"]["schemas"][ref_name].get("properties", {})
                else:
                    props = rb.get("properties", {})

                if "image_urls" in props:
                    _SCHEMA_CACHE[endpoint] = "image_urls"
                    return "image_urls"

        _SCHEMA_CACHE[endpoint] = "image_url"
        return "image_url"
    except Exception as exc:
        logger.warning(
            "Could not auto-detect image field for %s (%s). "
            "Using fallback '%s'. To set it manually:\n"
            "  from evalytic.bench import register_model\n"
            '  register_model("%s", endpoint="%s", pipeline="img2img", '
            'image_field="image_urls")',
            endpoint,
            exc,
            fallback,
            endpoint.rsplit("/", 1)[-1],
            endpoint,
        )
        _SCHEMA_CACHE[endpoint] = fallback
        return fallback


def register_model(
    name: str,
    *,
    endpoint: str | None = None,
    pipeline: str | None = None,
    cost_per_image: float | None = None,
    image_field: str | None = None,
) -> ModelEntry:
    """Register or override a model in the registry.

    Any field left as ``None`` inherits from the existing entry (if the
    model is already registered) or uses a sensible default.

    Examples::

        # Register a brand-new custom endpoint
        register_model("my-model",
            endpoint="fal-ai/my-custom/v1",
            pipeline="img2img",
            image_field="image_urls",
            cost_per_image=0.04,
        )

        # Override just the image_field of an existing model
        register_model("seedream-edit", image_field="image_urls")
    """
    existing = MODEL_REGISTRY.get(name)

    entry = ModelEntry(
        short_name=name,
        endpoint=endpoint or (existing.endpoint if existing else name),
        pipeline=pipeline or (existing.pipeline if existing else "text2img"),
        cost_per_image=cost_per_image if cost_per_image is not None else (existing.cost_per_image if existing else 0.0),
        image_field=image_field or (existing.image_field if existing else "image_url"),
    )
    MODEL_REGISTRY[name] = entry
    _USER_OVERRIDES.add(name)
    return entry


def resolve_model(name: str) -> ModelEntry:
    """Resolve a short name or literal endpoint to a ModelEntry.

    If *name* contains ``/``, it is first checked against registry entries
    by endpoint, then treated as a literal fal.ai endpoint.
    Otherwise it is looked up in MODEL_REGISTRY.  On failure, an
    ``UnknownModelError`` is raised with a fuzzy suggestion.
    """
    if "/" in name:
        # Check if any registered model has this endpoint
        for entry in MODEL_REGISTRY.values():
            if entry.endpoint == name:
                return entry
        return ModelEntry(
            short_name=name,
            endpoint=name,
            pipeline="text2img",  # default; caller may override
            cost_per_image=0.0,
        )
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    matches = difflib.get_close_matches(name, MODEL_REGISTRY.keys(), n=1, cutoff=0.5)
    raise UnknownModelError(name, suggestion=matches[0] if matches else None)


def list_models(pipeline: str | None = None) -> list[ModelEntry]:
    """Return all registered models, optionally filtered by pipeline type."""
    entries = list(MODEL_REGISTRY.values())
    if pipeline:
        entries = [e for e in entries if e.pipeline == pipeline]
    return entries


def estimate_cost(model: str, count: int) -> float:
    """Estimate total generation cost for *count* images with *model*."""
    entry = resolve_model(model)
    return entry.cost_per_image * count
