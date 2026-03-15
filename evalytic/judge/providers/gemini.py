"""Gemini REST API provider."""

from __future__ import annotations

from typing import Any

import httpx

from ..common import parse_response
from .base import BaseProvider


class GeminiProvider(BaseProvider):
    """Gemini via direct REST API (generativelanguage.googleapis.com)."""

    def __init__(self, model: str, api_key: str, base_url: str) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self._client = httpx.Client(timeout=60)

    def complete(
        self,
        user_prompt: str,
        system_prompt: str,
        images: list[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        parts: list[dict[str, Any]] = []
        if images:
            for b64, mime in images:
                parts.append({"inlineData": {"mimeType": mime, "data": b64}})
        parts.append({"text": user_prompt})

        payload = {
            "contents": [{"parts": parts}],
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"

        resp = self._client.post(url, json=payload)
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        return parse_response(text)

    def close(self) -> None:
        self._client.close()
