"""LocalJudge -- direct Gemini scoring using the same prompts as the Evalytic Lambda judge.

When the Evalytic API is deployed, swap to the real SDK with a single env-var toggle.
Until then this module gives *real VLM scores* with zero backend dependency.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Prompts for VLM judge dimensions
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert visual quality evaluator for AI-generated images.\n"
    "You evaluate images on specific quality dimensions using a structured rubric.\n"
    "You must return your evaluation as valid JSON matching the requested schema.\n"
    "Be precise, objective, and provide specific evidence for your scores."
)

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
    "explanation": "<2-3 sentence explanation of how well the image matches the prompt>",
    "evidence": ["<specific match or mismatch 1>", "<specific match or mismatch 2>"]
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
    "explanation": "<2-3 sentence explanation of artifacts found or absent>",
    "evidence": ["<specific artifact or clean area 1>", "<specific artifact or clean area 2>"]
  }]
}"""


# ---------------------------------------------------------------------------
# Dimension config -- which prompt to use, and which images are required
# ---------------------------------------------------------------------------

DIMENSION_CONFIG: dict[str, dict[str, Any]] = {
    "visual_quality": {
        "prompt_template": VISUAL_QUALITY_PROMPT,
        "needs_input": False,
    },
    "prompt_adherence": {
        "prompt_template": PROMPT_ADHERENCE_PROMPT,
        "needs_input": False,
    },
    "input_fidelity": {
        "prompt_template": INPUT_FIDELITY_PROMPT,
        "needs_input": True,
    },
    "transformation_quality": {
        "prompt_template": TRANSFORMATION_QUALITY_PROMPT,
        "needs_input": True,
    },
    "artifact_detection": {
        "prompt_template": ARTIFACT_DETECTION_PROMPT,
        "needs_input": True,
    },
}


# ---------------------------------------------------------------------------
# Image helpers for Gemini API
# ---------------------------------------------------------------------------

def _guess_mime(url: str) -> str:
    lower = url.lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".webp"):
        return "image/webp"
    if lower.endswith(".gif"):
        return "image/gif"
    return "image/jpeg"


def _fetch_image_base64(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Evalytic/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return base64.b64encode(resp.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Result dataclasses (mirrors sdk/evalytic/types.py)
# ---------------------------------------------------------------------------

@dataclass
class LocalDimensionScore:
    dimension: str
    score: float
    explanation: str = ""
    evidence: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"LocalDimensionScore(dimension={self.dimension!r}, score={self.score})"


@dataclass
class LocalEvalResult:
    scores: list[LocalDimensionScore] = field(default_factory=list)
    image_url: str = ""
    input_image_url: str = ""

    @property
    def overall_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.score for s in self.scores) / len(self.scores)

    @property
    def display_score(self) -> str:
        return f"{self.overall_score:.1f}/5"

    def dimension(self, name: str) -> LocalDimensionScore | None:
        for s in self.scores:
            if s.dimension == name:
                return s
        return None


# ---------------------------------------------------------------------------
# LocalJudge -- Gemini REST API, same pattern as the Lambda judge
# ---------------------------------------------------------------------------

class LocalJudge:
    """Score images with Gemini using the exact same prompts as the Evalytic judge Lambda."""

    def __init__(self, api_key: str | None = None, model: str = "gemini-2.0-flash"):
        self.api_key = api_key or os.environ["GEMINI_API_KEY"]
        self.model = model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

    def score(
        self,
        image_url: str,
        dimensions: list[str],
        prompt: str | None = None,
        input_image_url: str | None = None,
    ) -> LocalEvalResult:
        """Evaluate an image on the given dimensions. Returns a LocalEvalResult."""
        all_scores: list[LocalDimensionScore] = []

        for dim in dimensions:
            cfg = DIMENSION_CONFIG[dim]

            # Build the user prompt, substituting {prompt} if needed
            user_prompt = cfg["prompt_template"]
            if prompt and "{prompt}" in user_prompt:
                user_prompt = user_prompt.format(prompt=prompt)

            # Build image parts
            parts: list[dict] = []
            if cfg["needs_input"]:
                if not input_image_url:
                    raise ValueError(f"Dimension {dim!r} requires input_image_url")
                parts.append({
                    "inlineData": {
                        "mimeType": _guess_mime(input_image_url),
                        "data": _fetch_image_base64(input_image_url),
                    }
                })

            parts.append({
                "inlineData": {
                    "mimeType": _guess_mime(image_url),
                    "data": _fetch_image_base64(image_url),
                }
            })
            parts.append({"text": user_prompt})

            # Call Gemini
            raw = self._call_gemini(parts)

            # Parse dimension scores from response
            for d in raw.get("dimensions", []):
                all_scores.append(LocalDimensionScore(
                    dimension=d["dimension"],
                    score=float(d["score"]),
                    explanation=d.get("explanation", ""),
                    evidence=d.get("evidence", []),
                ))

        return LocalEvalResult(
            scores=all_scores,
            image_url=image_url,
            input_image_url=input_image_url or "",
        )

    def _call_gemini(self, parts: list[dict]) -> dict[str, Any]:
        payload = {
            "contents": [{"parts": parts}],
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }

        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        text = result["candidates"][0]["content"]["parts"][0]["text"]

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            elif "```" in text:
                json_str = text.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
            raise
