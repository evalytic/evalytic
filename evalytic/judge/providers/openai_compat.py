"""OpenAI-compatible provider (OpenAI, fal.ai, Ollama, LM Studio, local)."""

from __future__ import annotations

from typing import Any

import httpx

from ..common import extract_response_cost, parse_response
from .base import BaseProvider


class OpenAICompatProvider(BaseProvider):
    """OpenAI-compatible chat completions API."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        provider: str,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.provider = provider
        self._client = httpx.Client(timeout=60)

    def complete(
        self,
        user_prompt: str,
        system_prompt: str,
        images: list[tuple[str, str]] | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.last_cost = 0.0
        content: list[dict[str, Any]] = []
        if images:
            for b64, mime in images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                })
        content.append({"type": "text", "text": user_prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        if response_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": response_schema,
            }
        url = f"{self.base_url}/chat/completions"

        headers: dict[str, str] = {}
        if self.api_key:
            if self.provider == "fal":
                headers["Authorization"] = f"Key {self.api_key}"
            else:
                headers["Authorization"] = f"Bearer {self.api_key}"

        resp = self._client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        self.last_cost = extract_response_cost(resp)
        text = resp.json()["choices"][0]["message"]["content"]
        return parse_response(text)

    def close(self) -> None:
        self._client.close()
