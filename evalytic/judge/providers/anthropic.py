"""Anthropic Messages API provider."""

from __future__ import annotations

from typing import Any

import httpx

from ..common import parse_response
from .base import BaseProvider


class AnthropicProvider(BaseProvider):
    """Anthropic via direct REST API (api.anthropic.com)."""

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
        content: list[dict[str, Any]] = []
        if images:
            for b64, mime in images:
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": mime, "data": b64},
                })
        content.append({"type": "text", "text": user_prompt})

        payload = {
            "model": self.model,
            "system": system_prompt,
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
        return parse_response(text)

    def close(self) -> None:
        self._client.close()
