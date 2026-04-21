"""AWS Bedrock provider for Claude models."""

from __future__ import annotations

import base64
import io
import json
from typing import Any

from ..common import parse_response
from .base import BaseProvider

_BEDROCK_MAX_IMAGE_BYTES = 4_500_000  # 5MB limit, leave margin


class BedrockProvider(BaseProvider):
    """Claude via AWS Bedrock (uses boto3 + AWS credentials from profile/env)."""

    def __init__(self, model: str, region: str = "eu-central-1") -> None:
        import boto3

        self.model = model
        self.base_url = f"bedrock:{region}"
        self.api_key = ""
        self._client = boto3.client("bedrock-runtime", region_name=region)

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
                raw = base64.b64decode(b64)
                # Detect actual mime from content (file extension can lie)
                mime = self._detect_mime(raw)
                if len(raw) > _BEDROCK_MAX_IMAGE_BYTES:
                    b64, mime = self._compress_to_jpeg(raw)
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": mime, "data": b64},
                })
        content.append({"type": "text", "text": user_prompt})

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "system": system_prompt,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 1024,
            "temperature": 0.1,
        })

        resp = self._client.invoke_model(
            modelId=self.model,
            contentType="application/json",
            accept="application/json",
            body=body,
        )

        result = json.loads(resp["body"].read())
        text = result["content"][0]["text"]
        return parse_response(text)

    @staticmethod
    def _detect_mime(raw: bytes) -> str:
        """Detect actual image mime type from magic bytes."""
        if raw[:8] == b'\x89PNG\r\n\x1a\n':
            return "image/png"
        if raw[:2] == b'\xff\xd8':
            return "image/jpeg"
        if raw[:4] == b'RIFF' and raw[8:12] == b'WEBP':
            return "image/webp"
        if raw[:3] == b'GIF':
            return "image/gif"
        return "image/png"  # fallback

    @staticmethod
    def _compress_to_jpeg(raw: bytes) -> tuple[str, str]:
        """Compress image to JPEG to fit under Bedrock's 5MB limit.

        Same dimensions, no resize — only format conversion (PNG→JPEG).
        """
        from PIL import Image

        img = Image.open(io.BytesIO(raw))
        if img.mode == "RGBA":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"

    def close(self) -> None:
        pass
