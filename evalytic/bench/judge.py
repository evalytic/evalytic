"""VLM judge -- scores images using structured prompts.

Supports multiple providers via a single Judge class:
  - Gemini (default): gemini-2.5-flash, gemini-2.5-pro, gemini-3-flash, gemini-3.1-pro
  - OpenAI: gpt-5.2, o4-mini
  - Anthropic: claude-sonnet-4-6, claude-opus-4-6, claude-haiku-4-5
  - fal.ai OpenRouter: fal/gemini-2.5-flash, fal/gpt-5.2, fal/claude-sonnet-4-6 (all via FAL_KEY)
  - Self-hosted: ollama/qwen3-vl, ollama/internvl3, lmstudio/*, local/*

Usage:
    judge = Judge("gemini-3-flash")
    judge = Judge("gpt-5.2")
    judge = Judge("claude-sonnet-4-6")
    judge = Judge("fal/gemini-2.5-flash")    # Gemini via fal.ai OpenRouter
    judge = Judge("ollama/qwen3-vl")
    judge = Judge("local/my-model", base_url="http://localhost:8090/v1")
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..exceptions import ValidationError
from ..judge.common import call_with_retries, guess_mime, parse_response
from ..judge.providers import (
    RECOMMENDED_JUDGES,
    _FAL_MODEL_MAP,
    _OPENAI_COMPAT_PROVIDERS,
    _PROVIDER_DEFAULTS,
    _parse_judge_string,
    create_provider,
)
from .types import DimensionResult

# ---------------------------------------------------------------------------
# System prompt -- identical to infra/lambda/judge/prompts/shared.py
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert visual quality evaluator for AI-generated images.\n"
    "You evaluate images on specific quality dimensions using a structured rubric.\n"
    "You must return your evaluation as valid JSON matching the requested schema.\n"
    "Be precise, objective, and provide specific evidence for your scores.\n"
    "For each dimension, include a confidence score (0.0-1.0) indicating how certain you are about your rating. "
    "Rate your confidence honestly — do not default to high confidence."
)

# ---------------------------------------------------------------------------
# Dimension prompts -- verbatim from infra/lambda/judge/prompts/
# ---------------------------------------------------------------------------

VISUAL_QUALITY_PROMPT = """Evaluate the VISUAL QUALITY of this AI-generated image.

Score on a 1.0-5.0 scale using 0.1 increments (e.g., 3.7, 4.2, 4.8):
1.0 - Very Poor: Major artifacts, distortions, blurring, or incoherent elements
2.0 - Poor: Noticeable quality issues, some artifacts, inconsistent rendering
3.0 - Average: Acceptable quality, minor artifacts or inconsistencies
4.0 - Good: High quality, clean rendering, minimal issues
5.0 - Excellent: Professional quality, no visible artifacts, sharp and coherent

Use the full range — reserve 5.0 for truly flawless images. Most good images should score 3.5-4.5.

Return JSON:
{
  "dimensions": [{
    "dimension": "visual_quality",
    "score": <1.0-5.0>,
    "confidence": <0.0-1.0>,
    "explanation": "<2-3 sentence explanation>",
    "evidence": ["<specific observation 1>", "<specific observation 2>"]
  }]
}"""

PROMPT_ADHERENCE_PROMPT = """Evaluate how well this AI-generated image matches the given prompt.

Prompt: "{prompt}"

Score on a 1.0-5.0 scale using 0.1 increments (e.g., 3.7, 4.2, 4.8):
1.0 - Very Poor: Image has almost no relation to the prompt
2.0 - Poor: Some elements present but major aspects missing or wrong
3.0 - Average: Main subject captured but details differ from prompt
4.0 - Good: Strong match to prompt with minor deviations
5.0 - Excellent: Perfect representation of the prompt in every detail

Use the full range — reserve 5.0 for images that match every detail perfectly. Most good images should score 3.5-4.5.

Return JSON:
{{
  "dimensions": [{{
    "dimension": "prompt_adherence",
    "score": <1.0-5.0>,
    "confidence": <0.0-1.0>,
    "explanation": "<2-3 sentence explanation of how well the image matches the prompt>",
    "evidence": ["<specific match or mismatch 1>", "<specific match or mismatch 2>"]
  }}]
}}"""

TEXT_RENDERING_PROMPT = """Evaluate the TEXT RENDERING quality in this AI-generated image.

Expected text content (from prompt): "{prompt}"

Score on a 1.0-5.0 scale using 0.1 increments (e.g., 3.7, 4.2, 4.8):
1.0 - Very Poor: Text is unreadable, garbled, or completely wrong
2.0 - Poor: Text partially readable but with significant errors
3.0 - Average: Text mostly readable but with some misspellings or artifacts
4.0 - Good: Text is clear and correct with minor rendering issues
5.0 - Excellent: Text is perfectly rendered, crisp, and accurate

If no text is expected in the image, score 5.0 and note "No text expected."
Use the full range — reserve 5.0 for pixel-perfect text rendering.

Return JSON:
{{
  "dimensions": [{{
    "dimension": "text_rendering",
    "score": <1.0-5.0>,
    "confidence": <0.0-1.0>,
    "explanation": "<2-3 sentence explanation>",
    "evidence": ["<specific observation 1>", "<specific observation 2>"]
  }}]
}}"""

INPUT_FIDELITY_PROMPT = """Evaluate the INPUT FIDELITY of this image transformation.

You are given two images:
- Image 1: The ORIGINAL input image
- Image 2: The TRANSFORMED output image

Score how well the output preserves key features from the input on a 1.0-5.0 scale using 0.1 increments (e.g., 3.7, 4.2, 4.8):
1.0 - Very Poor: Key features completely lost (faces unrecognizable, objects missing)
2.0 - Poor: Major features lost or significantly altered
3.0 - Average: Main subject preserved but noticeable loss of identity/features
4.0 - Good: Strong preservation of key features with minor differences
5.0 - Excellent: Perfect preservation of identity, faces, objects, and composition

Use the full range — reserve 5.0 for pixel-perfect preservation. Most good transformations should score 3.5-4.5.

Return JSON:
{
  "dimensions": [{
    "dimension": "input_fidelity",
    "score": <1.0-5.0>,
    "confidence": <0.0-1.0>,
    "explanation": "<2-3 sentence explanation comparing input and output>",
    "evidence": ["<specific preserved or lost feature 1>", "<specific preserved or lost feature 2>"]
  }]
}"""

TRANSFORMATION_QUALITY_PROMPT = """Evaluate the TRANSFORMATION QUALITY of this image transformation.

You are given two images:
- Image 1: The ORIGINAL input image
- Image 2: The TRANSFORMED output image

Score how well the intended transformation was applied on a 1.0-5.0 scale using 0.1 increments (e.g., 3.7, 4.2, 4.8):
1.0 - Very Poor: Transformation not applied or completely wrong result
2.0 - Poor: Transformation partially applied with major issues
3.0 - Average: Transformation applied but with noticeable flaws
4.0 - Good: Transformation well-applied with minor imperfections
5.0 - Excellent: Transformation perfectly applied, natural-looking result

Use the full range — reserve 5.0 for flawless transformations. Most good results should score 3.5-4.5.

Return JSON:
{
  "dimensions": [{
    "dimension": "transformation_quality",
    "score": <1.0-5.0>,
    "confidence": <0.0-1.0>,
    "explanation": "<2-3 sentence explanation of transformation quality>",
    "evidence": ["<specific observation 1>", "<specific observation 2>"]
  }]
}"""

ARTIFACT_DETECTION_PROMPT = """Evaluate the ARTIFACT level in this transformed image.

You are given two images:
- Image 1: The ORIGINAL input image
- Image 2: The TRANSFORMED output image

Score the absence of artifacts introduced by the transformation on a 1.0-5.0 scale using 0.1 increments (e.g., 3.7, 4.2, 4.8):
1.0 - Very Poor: Severe artifacts (halos, seams, color banding, blurring, ghosting)
2.0 - Poor: Multiple noticeable artifacts
3.0 - Average: Some minor artifacts visible on close inspection
4.0 - Good: Very few artifacts, barely noticeable
5.0 - Excellent: No visible artifacts, clean transformation

Use the full range — reserve 5.0 for artifact-free results.

Return JSON:
{
  "dimensions": [{
    "dimension": "artifact_detection",
    "score": <1.0-5.0>,
    "confidence": <0.0-1.0>,
    "explanation": "<2-3 sentence explanation of artifacts found or absent>",
    "evidence": ["<specific artifact or clean area 1>", "<specific artifact or clean area 2>"]
  }]
}"""

IDENTITY_PRESERVATION_PROMPT = """Evaluate the IDENTITY PRESERVATION in this image transformation.

You are given two images:
- Image 1: The ORIGINAL input image
- Image 2: The TRANSFORMED output image

FIRST: Determine if there are any human faces or people in the INPUT image.
If NO people/faces in the input image: score 5.0, confidence 1.0,
explanation "No human faces in input image — identity preservation not applicable."

If people ARE present, score on a 1.0-5.0 scale using 0.1 increments (e.g., 3.7, 4.2, 4.8):
1.0 - Very Poor: Face completely changed, person unrecognizable
2.0 - Poor: Major facial feature changes (eye shape, nose, jawline altered)
3.0 - Average: Same person recognizable but noticeable differences (skin tone, expression, proportions)
4.0 - Good: Strong resemblance, minor differences only visible on close comparison
5.0 - Excellent: Perfect identity match — face, skin tone, body proportions fully preserved

Use the full range — reserve 5.0 for indistinguishable identity matches.

Focus specifically on:
- Facial feature accuracy (eyes, nose, mouth, jawline)
- Skin tone and complexion consistency
- Body proportions and posture
- Expression preservation

Return JSON:
{
  "dimensions": [{
    "dimension": "identity_preservation",
    "score": <1.0-5.0>,
    "confidence": <0.0-1.0>,
    "explanation": "<2-3 sentence explanation of identity preservation>",
    "evidence": ["<specific preserved or changed feature 1>", "<specific preserved or changed feature 2>"]
  }]
}"""

# ---------------------------------------------------------------------------
# Dimension config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _DimConfig:
    prompt_template: str
    needs_input: bool
    needs_prompt: bool


DIMENSION_CONFIG: dict[str, _DimConfig] = {
    "visual_quality": _DimConfig(VISUAL_QUALITY_PROMPT, needs_input=False, needs_prompt=False),
    "prompt_adherence": _DimConfig(PROMPT_ADHERENCE_PROMPT, needs_input=False, needs_prompt=True),
    "text_rendering": _DimConfig(TEXT_RENDERING_PROMPT, needs_input=False, needs_prompt=True),
    "input_fidelity": _DimConfig(INPUT_FIDELITY_PROMPT, needs_input=True, needs_prompt=False),
    "transformation_quality": _DimConfig(TRANSFORMATION_QUALITY_PROMPT, needs_input=True, needs_prompt=False),
    "artifact_detection": _DimConfig(ARTIFACT_DETECTION_PROMPT, needs_input=True, needs_prompt=False),
    "identity_preservation": _DimConfig(IDENTITY_PRESERVATION_PROMPT, needs_input=True, needs_prompt=False),
}

ALL_JUDGE_DIMENSIONS = list(DIMENSION_CONFIG.keys())


# ---------------------------------------------------------------------------
# Judge (multi-provider)
# ---------------------------------------------------------------------------

class Judge:
    """Score images with VLM judges. Supports Gemini, OpenAI, Anthropic, and local models."""

    def __init__(
        self,
        judge: str = "gemini-2.5-flash",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._provider, self.provider, self.model = create_provider(
            judge, api_key=api_key, base_url=base_url,
        )
        self.judge_string = judge
        self.base_url = self._provider.base_url
        self.api_key = self._provider.api_key

        # Share the provider's httpx client for image fetching too.
        # This keeps a single _client attribute (backward compat with tests).
        self._client = self._provider._client

    def score(
        self,
        image_url: str,
        dimensions: list[str],
        prompt: str | None = None,
        input_image_url: str | None = None,
    ) -> list[DimensionResult]:
        """Evaluate an image on the given dimensions."""
        unknown = set(dimensions) - set(DIMENSION_CONFIG)
        if unknown:
            raise ValidationError(
                f"Unknown dimensions: {sorted(unknown)}. "
                f"Valid: {list(DIMENSION_CONFIG)}"
            )

        results: list[DimensionResult] = []

        for dim in dimensions:
            cfg = DIMENSION_CONFIG[dim]

            user_prompt = cfg.prompt_template
            if cfg.needs_prompt and prompt and "{prompt}" in user_prompt:
                user_prompt = user_prompt.format(prompt=prompt)

            # Collect images as provider-agnostic (b64, mime) tuples
            images: list[tuple[str, str]] = []
            if cfg.needs_input:
                if not input_image_url:
                    raise ValidationError(f"Dimension {dim!r} requires input_image_url")
                images.append(self._fetch_image_base64(input_image_url))

            images.append(self._fetch_image_base64(image_url))

            # Dispatch to provider with retries
            raw = call_with_retries(
                self._provider.complete,
                user_prompt,
                SYSTEM_PROMPT,
                images=images,
            )

            for d in raw.get("dimensions", []):
                results.append(
                    DimensionResult(
                        dimension=d["dimension"],
                        score=float(d["score"]),
                        confidence=float(d.get("confidence", 1.0)),
                        explanation=d.get("explanation", ""),
                        evidence=d.get("evidence", []),
                    )
                )

        return results

    # -- Image fetching (shared) -------------------------------------------

    def _fetch_image_base64(self, url: str) -> tuple[str, str]:
        """Download an image (or read a local file) and return ``(base64_data, mime_type)``."""
        if not url.startswith(("http://", "https://")):
            p = Path(url).expanduser()
            if not p.exists():
                raise ValidationError(f"Image file not found: {url}")
            return base64.b64encode(p.read_bytes()).decode("utf-8"), guess_mime(str(p))
        resp = self._client.get(url, follow_redirects=True)
        resp.raise_for_status()
        mime = resp.headers.get("content-type", guess_mime(url))
        return base64.b64encode(resp.content).decode("utf-8"), mime

    # -- Lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._provider.close()

    def __enter__(self) -> Judge:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # -- Backward compat (used by tests) -----------------------------------

    @staticmethod
    def _parse_response(text: str) -> dict[str, Any]:
        return parse_response(text)


# Backward compatibility alias
GeminiJudge = Judge
