"""VLM judge -- scores images using structured prompts.

Supports multiple providers via a single Judge class:
  - Gemini (default): gemini-3-flash, gemini-3.1-pro, gemini-2.5-flash, gemini-2.5-pro
  - OpenAI: gpt-5.2, gpt-4o, gpt-4o-mini, o4-mini
  - Anthropic: claude-sonnet-4-6, claude-opus-4-6, claude-haiku-4-5
  - Self-hosted: ollama/qwen3-vl, ollama/internvl3, lmstudio/*, local/*

Usage:
    judge = Judge("gemini-3-flash")
    judge = Judge("gpt-5.2")
    judge = Judge("claude-sonnet-4-6")
    judge = Judge("ollama/qwen3-vl")
    judge = Judge("local/my-model", base_url="http://localhost:8090/v1")
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from ..exceptions import JudgeError, ValidationError
from .types import DimensionResult

# ---------------------------------------------------------------------------
# System prompt for VLM judge
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
# Dimension prompts
# ---------------------------------------------------------------------------

VISUAL_QUALITY_PROMPT = """Evaluate the VISUAL QUALITY of this AI-generated image.

Score on a 1-5 scale:
1 - Very Poor: Major artifacts, distortions, blurring, or incoherent elements
2 - Poor: Noticeable quality issues, some artifacts, inconsistent rendering
3 - Average: Acceptable quality, minor artifacts or inconsistencies
4 - Good: High quality, clean rendering, minimal issues
5 - Excellent: Professional quality, no visible artifacts, sharp and coherent

Return JSON:
{
  "dimensions": [{
    "dimension": "visual_quality",
    "score": <1-5>,
    "confidence": <0.0-1.0>,
    "explanation": "<2-3 sentence explanation>",
    "evidence": ["<specific observation 1>", "<specific observation 2>"]
  }]
}"""

PROMPT_ADHERENCE_PROMPT = """Evaluate how well this AI-generated image matches the given prompt.

Prompt: "{prompt}"

Score on a 1-5 scale:
1 - Very Poor: Image has almost no relation to the prompt
2 - Poor: Some elements present but major aspects missing or wrong
3 - Average: Main subject captured but details differ from prompt
4 - Good: Strong match to prompt with minor deviations
5 - Excellent: Perfect representation of the prompt in every detail

Return JSON:
{{
  "dimensions": [{{
    "dimension": "prompt_adherence",
    "score": <1-5>,
    "confidence": <0.0-1.0>,
    "explanation": "<2-3 sentence explanation of how well the image matches the prompt>",
    "evidence": ["<specific match or mismatch 1>", "<specific match or mismatch 2>"]
  }}]
}}"""

TEXT_RENDERING_PROMPT = """Evaluate the TEXT RENDERING quality in this AI-generated image.

Expected text content (from prompt): "{prompt}"

Score on a 1-5 scale:
1 - Very Poor: Text is unreadable, garbled, or completely wrong
2 - Poor: Text partially readable but with significant errors
3 - Average: Text mostly readable but with some misspellings or artifacts
4 - Good: Text is clear and correct with minor rendering issues
5 - Excellent: Text is perfectly rendered, crisp, and accurate

If no text is expected in the image, score 5 and note "No text expected."

Return JSON:
{{
  "dimensions": [{{
    "dimension": "text_rendering",
    "score": <1-5>,
    "confidence": <0.0-1.0>,
    "explanation": "<2-3 sentence explanation>",
    "evidence": ["<specific observation 1>", "<specific observation 2>"]
  }}]
}}"""

INPUT_FIDELITY_PROMPT = """Evaluate the INPUT FIDELITY of this image transformation.

You are given two images:
- Image 1: The ORIGINAL input image
- Image 2: The TRANSFORMED output image

Score how well the output preserves key features from the input on a 1-5 scale:
1 - Very Poor: Key features completely lost (faces unrecognizable, objects missing)
2 - Poor: Major features lost or significantly altered
3 - Average: Main subject preserved but noticeable loss of identity/features
4 - Good: Strong preservation of key features with minor differences
5 - Excellent: Perfect preservation of identity, faces, objects, and composition

Return JSON:
{
  "dimensions": [{
    "dimension": "input_fidelity",
    "score": <1-5>,
    "confidence": <0.0-1.0>,
    "explanation": "<2-3 sentence explanation comparing input and output>",
    "evidence": ["<specific preserved or lost feature 1>", "<specific preserved or lost feature 2>"]
  }]
}"""

TRANSFORMATION_QUALITY_PROMPT = """Evaluate the TRANSFORMATION QUALITY of this image transformation.

You are given two images:
- Image 1: The ORIGINAL input image
- Image 2: The TRANSFORMED output image

Score how well the intended transformation was applied on a 1-5 scale:
1 - Very Poor: Transformation not applied or completely wrong result
2 - Poor: Transformation partially applied with major issues
3 - Average: Transformation applied but with noticeable flaws
4 - Good: Transformation well-applied with minor imperfections
5 - Excellent: Transformation perfectly applied, natural-looking result

Return JSON:
{
  "dimensions": [{
    "dimension": "transformation_quality",
    "score": <1-5>,
    "confidence": <0.0-1.0>,
    "explanation": "<2-3 sentence explanation of transformation quality>",
    "evidence": ["<specific observation 1>", "<specific observation 2>"]
  }]
}"""

ARTIFACT_DETECTION_PROMPT = """Evaluate the ARTIFACT level in this transformed image.

You are given two images:
- Image 1: The ORIGINAL input image
- Image 2: The TRANSFORMED output image

Score the absence of artifacts introduced by the transformation on a 1-5 scale:
1 - Very Poor: Severe artifacts (halos, seams, color banding, blurring, ghosting)
2 - Poor: Multiple noticeable artifacts
3 - Average: Some minor artifacts visible on close inspection
4 - Good: Very few artifacts, barely noticeable
5 - Excellent: No visible artifacts, clean transformation

Return JSON:
{
  "dimensions": [{
    "dimension": "artifact_detection",
    "score": <1-5>,
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
If NO people/faces in the input image: score 5, confidence 1.0,
explanation "No human faces in input image — identity preservation not applicable."

If people ARE present, score on a 1-5 scale:
1 - Very Poor: Face completely changed, person unrecognizable
2 - Poor: Major facial feature changes (eye shape, nose, jawline altered)
3 - Average: Same person recognizable but noticeable differences (skin tone, expression, proportions)
4 - Good: Strong resemblance, minor differences only visible on close comparison
5 - Excellent: Perfect identity match — face, skin tone, body proportions fully preserved

Focus specifically on:
- Facial feature accuracy (eyes, nose, mouth, jawline)
- Skin tone and complexion consistency
- Body proportions and posture
- Expression preservation

Return JSON:
{
  "dimensions": [{
    "dimension": "identity_preservation",
    "score": <1-5>,
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
# Provider config
# ---------------------------------------------------------------------------

_PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "env_key": "GEMINI_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com",
        "env_key": "ANTHROPIC_API_KEY",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "env_key": None,
    },
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "env_key": None,
    },
    "local": {
        "base_url": "http://localhost:8090/v1",
        "env_key": None,
    },
}

# Providers that use OpenAI-compatible API format
_OPENAI_COMPAT_PROVIDERS = {"openai", "ollama", "lmstudio", "local"}

# Short aliases → actual Gemini API model names.
# Users type "gemini-3-flash", API expects "gemini-3-flash-preview".
_GEMINI_MODEL_ALIASES: dict[str, str] = {
    "gemini-3-flash": "gemini-3-flash-preview",
    "gemini-3.1-pro": "gemini-3.1-pro-preview",
    "gemini-3-pro": "gemini-3-pro-preview",
}

# Recommended judge models by provider (for docs/CLI help)
RECOMMENDED_JUDGES: dict[str, list[str]] = {
    "gemini": [
        "gemini-3-flash",       # Best quality/price — MMMU 87.6%, $0.50/M input
        "gemini-3.1-pro",       # Flagship — $2.00/M input
        "gemini-2.5-flash",     # Budget — $0.30/M input
    ],
    "openai": [
        "gpt-5.2",             # Best accuracy — MMMU-Pro 86.5%, $1.75/M input
        "gpt-4o",              # Proven — $2.50/M input
        "gpt-4o-mini",         # Ultra-budget — $0.15/M input
        "o4-mini",             # Reasoning — $1.10/M input
    ],
    "anthropic": [
        "claude-sonnet-4-6",   # Balanced — $3.00/M input
        "claude-opus-4-6",     # Premium — $5.00/M input
        "claude-haiku-4-5",    # Fast — $0.80/M input
    ],
    "self-hosted": [
        "ollama/qwen3-vl",            # Best open-source VLM
        "ollama/internvl3",            # Strong open-source alternative
        "ollama/glm-4.1v-9b-thinking", # Compact reasoning VLM
    ],
}

# Prefixes that map to OpenAI provider (gpt-*, o1-*, o3-*, o4-*, chatgpt-*)
_OPENAI_PREFIXES = ("gpt", "o1", "o3", "o4", "chatgpt")


def _parse_judge_string(judge: str) -> tuple[str, str]:
    """Parse a judge string into (provider, model).

    Examples:
        "gemini-3-flash"            -> ("gemini", "gemini-3-flash")
        "gpt-5.2"                   -> ("openai", "gpt-5.2")
        "o4-mini"                   -> ("openai", "o4-mini")
        "claude-sonnet-4-6"         -> ("anthropic", "claude-sonnet-4-6")
        "openai/gpt-5.2"            -> ("openai", "gpt-5.2")
        "ollama/qwen3-vl"           -> ("ollama", "qwen3-vl")
        "local/my-model"            -> ("local", "my-model")
    """
    if "/" in judge:
        provider, model = judge.split("/", 1)
        if provider not in _PROVIDER_DEFAULTS:
            raise ValidationError(
                f"Unknown judge provider: {provider!r}. "
                f"Valid: {sorted(_PROVIDER_DEFAULTS)}"
            )
        return provider, model

    # Bare model name -- infer provider from prefix
    if judge.startswith("gemini"):
        return "gemini", judge
    for prefix in _OPENAI_PREFIXES:
        if judge.startswith(prefix):
            return "openai", judge
    if judge.startswith("claude"):
        return "anthropic", judge

    # Default to gemini for backward compat
    return "gemini", judge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _guess_mime(url: str) -> str:
    lower = url.lower().split("?")[0]
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".webp"):
        return "image/webp"
    if lower.endswith(".gif"):
        return "image/gif"
    return "image/jpeg"


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
        self.provider, self.model = _parse_judge_string(judge)
        self.judge_string = judge

        # Resolve short aliases to actual API model names
        if self.provider == "gemini":
            self.model = _GEMINI_MODEL_ALIASES.get(self.model, self.model)

        defaults = _PROVIDER_DEFAULTS[self.provider]
        self.base_url = base_url or defaults["base_url"]

        env_key = defaults["env_key"]
        self.api_key = api_key or (os.environ.get(env_key, "") if env_key else "")

        self._client = httpx.Client(timeout=60)

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

            # Dispatch to provider
            raw = self._call_api(images, user_prompt)

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
            return base64.b64encode(p.read_bytes()).decode("utf-8"), _guess_mime(str(p))
        resp = self._client.get(url, follow_redirects=True)
        resp.raise_for_status()
        mime = resp.headers.get("content-type", _guess_mime(url))
        return base64.b64encode(resp.content).decode("utf-8"), mime

    # -- Provider dispatch -------------------------------------------------

    def _call_api(
        self,
        images: list[tuple[str, str]],
        user_prompt: str,
    ) -> dict[str, Any]:
        """Route to the correct provider with retries."""
        if self.provider == "gemini":
            call_fn = self._call_gemini
        elif self.provider in _OPENAI_COMPAT_PROVIDERS:
            call_fn = self._call_openai_compat
        elif self.provider == "anthropic":
            call_fn = self._call_anthropic
        else:
            raise ValidationError(f"Unknown provider: {self.provider!r}")

        last_error: Exception | None = None
        for attempt in range(3):
            if attempt > 0:
                time.sleep(2**attempt)
            try:
                return call_fn(images, user_prompt)
            except Exception as exc:
                last_error = exc
        raise JudgeError(
            f"Judge failed after 3 retries: {last_error}"
        ) from last_error  # type: ignore[misc]

    # -- Gemini ------------------------------------------------------------

    def _call_gemini(
        self,
        images: list[tuple[str, str]],
        user_prompt: str,
    ) -> dict[str, Any]:
        parts: list[dict[str, Any]] = []
        for b64, mime in images:
            parts.append({"inlineData": {"mimeType": mime, "data": b64}})
        parts.append({"text": user_prompt})

        payload = {
            "contents": [{"parts": parts}],
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"

        resp = self._client.post(url, json=payload)
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        return self._parse_response(text)

    # -- OpenAI-compatible (OpenAI, Ollama, LM Studio, local) --------------

    def _call_openai_compat(
        self,
        images: list[tuple[str, str]],
        user_prompt: str,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for b64, mime in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
        content.append({"type": "text", "text": user_prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        url = f"{self.base_url}/chat/completions"

        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        resp = self._client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        return self._parse_response(text)

    # -- Anthropic ---------------------------------------------------------

    def _call_anthropic(
        self,
        images: list[tuple[str, str]],
        user_prompt: str,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for b64, mime in images:
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": b64},
            })
        content.append({"type": "text", "text": user_prompt})

        payload = {
            "model": self.model,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 1024,
            "temperature": 0.1,
        }
        url = f"{self.base_url}/v1/messages"

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        resp = self._client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        text = resp.json()["content"][0]["text"]
        return self._parse_response(text)

    # -- Response parsing (shared) -----------------------------------------

    @staticmethod
    def _parse_response(text: str) -> dict[str, Any]:
        """Parse JSON from VLM response, with markdown fence fallback."""
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

    # -- Lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Judge:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# Backward compatibility alias
GeminiJudge = Judge
