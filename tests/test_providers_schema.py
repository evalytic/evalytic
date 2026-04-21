"""Verify provider implementations accept the new ``response_schema`` kwarg
without crashing and that OpenAI-compatible payloads switch format correctly.
"""

from __future__ import annotations

import inspect
import json
from typing import Any

import httpx
import pytest

from evalytic.judge.providers.anthropic import AnthropicProvider
from evalytic.judge.providers.base import BaseProvider
from evalytic.judge.providers.gemini import GeminiProvider
from evalytic.judge.providers.openai_compat import OpenAICompatProvider


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class _RecordingClient:
    def __init__(self, response_payload: dict) -> None:
        self.last_url: str | None = None
        self.last_json: dict | None = None
        self.last_headers: dict | None = None
        self._response_payload = response_payload

    def post(self, url: str, *, json: dict, headers: dict) -> _FakeResponse:
        self.last_url = url
        self.last_json = json
        self.last_headers = headers
        return _FakeResponse(self._response_payload)

    def close(self) -> None:
        return None


class TestOpenAICompatSchema:
    def test_schema_payload_when_provided(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = OpenAICompatProvider(
            model="gpt-4",
            api_key="sk-fake",
            base_url="https://api.openai.com/v1",
            provider="openai",
        )
        client = _RecordingClient({
            "choices": [{"message": {"content": json.dumps({"score": 0.9})}}]
        })
        provider._client = client  # type: ignore[assignment]

        schema = {"name": "metric", "schema": {"type": "object"}}
        result = provider.complete(
            user_prompt="hello",
            system_prompt="system",
            response_schema=schema,
        )
        assert result == {"score": 0.9}
        assert client.last_json is not None
        assert client.last_json["response_format"]["type"] == "json_schema"
        assert client.last_json["response_format"]["json_schema"] == schema

    def test_json_object_default_when_no_schema(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = OpenAICompatProvider(
            model="gpt-4",
            api_key="sk-fake",
            base_url="https://api.openai.com/v1",
            provider="openai",
        )
        client = _RecordingClient({
            "choices": [{"message": {"content": json.dumps({"ok": True})}}]
        })
        provider._client = client  # type: ignore[assignment]

        provider.complete(user_prompt="hi", system_prompt="sys")
        assert client.last_json is not None
        assert client.last_json["response_format"] == {"type": "json_object"}


class TestBaseSignature:
    def test_base_provider_complete_signature_stable(self) -> None:
        sig = inspect.signature(BaseProvider.complete)
        params = list(sig.parameters.keys())
        assert params[:1] == ["self"]
        assert "user_prompt" in params
        assert "system_prompt" in params
        assert "images" in params
        assert "response_schema" in params

    @pytest.mark.parametrize(
        "provider_cls",
        [GeminiProvider, AnthropicProvider, OpenAICompatProvider],
    )
    def test_provider_complete_has_response_schema_param(self, provider_cls) -> None:
        sig = inspect.signature(provider_cls.complete)
        assert "response_schema" in sig.parameters
