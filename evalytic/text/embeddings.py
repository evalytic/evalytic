"""Embedding resolution and cosine helpers for text eval metrics."""

from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from typing import Any

import httpx

from ..exceptions import ValidationError
from ..judge.common import extract_response_cost

DEFAULT_LOCAL_MODEL = "all-MiniLM-L6-v2"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_FAL_MODEL = "openai/text-embedding-3-small"
DEFAULT_FAL_BASE_URL = "https://fal.run/openrouter/router/openai/v1"


class BaseEmbedder(ABC):
    last_cost: float = 0.0

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def close(self) -> None:
        return None


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model: str = DEFAULT_LOCAL_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model
        self._model = SentenceTransformer(model)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [list(map(float, vector)) for vector in vectors]


class APIEmbedder(BaseEmbedder):
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        headers: dict[str, str],
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.headers = headers
        self._client = httpx.Client(timeout=60)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.last_cost = 0.0
        resp = self._client.post(
            f"{self.base_url}/embeddings",
            headers=self.headers,
            json={"model": self.model, "input": texts},
        )
        resp.raise_for_status()
        self.last_cost = extract_response_cost(resp)
        data = resp.json()["data"]
        return [list(map(float, row["embedding"])) for row in data]

    def close(self) -> None:
        self._client.close()


def resolve_embedder(config: dict[str, Any] | None = None) -> BaseEmbedder:
    cfg = (config or {}).get("embeddings", {})
    preferred_provider = cfg.get("provider")
    local_model = cfg.get("model", DEFAULT_LOCAL_MODEL)

    if preferred_provider in (None, "", "sentence-transformers", "local"):
        try:
            return SentenceTransformerEmbedder(local_model)
        except Exception:
            if preferred_provider in ("sentence-transformers", "local"):
                raise

    custom_api = _resolve_custom_api_embedder(cfg)
    if custom_api is not None:
        return custom_api

    if os.environ.get("OPENAI_API_KEY"):
        return APIEmbedder(
            base_url=DEFAULT_OPENAI_BASE_URL,
            model=cfg.get("model", DEFAULT_OPENAI_MODEL),
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        )

    if os.environ.get("FAL_KEY"):
        return APIEmbedder(
            base_url=DEFAULT_FAL_BASE_URL,
            model=cfg.get("model", DEFAULT_FAL_MODEL),
            headers={"Authorization": f"Key {os.environ['FAL_KEY']}"},
        )

    raise ValidationError(
        "Embeddings are required for this metric. Install evalytic[embeddings] "
        "for local embeddings, set OPENAI_API_KEY / FAL_KEY for built-in API embeddings, "
        "or configure [embeddings].base_url for a custom OpenAI-compatible endpoint."
    )


def _resolve_custom_api_embedder(cfg: dict[str, Any]) -> APIEmbedder | None:
    base_url = str(cfg.get("base_url", "")).strip()
    if not base_url:
        return None

    api_key = _resolve_custom_api_key(cfg)
    if not api_key:
        api_key_env = cfg.get("api_key_env")
        suffix = (
            f" via env var {api_key_env!r}" if api_key_env else " via OPENAI_API_KEY or embeddings.api_key"
        )
        raise ValidationError(
            "Custom embedding endpoint configured but no API key was found"
            f"{suffix}."
        )

    auth_scheme = str(cfg.get("auth_scheme", "bearer")).strip().lower()
    if auth_scheme not in {"bearer", "key"}:
        raise ValidationError(
            "embeddings.auth_scheme must be 'bearer' or 'key' when embeddings.base_url is set."
        )
    auth_value = f"Key {api_key}" if auth_scheme == "key" else f"Bearer {api_key}"
    return APIEmbedder(
        base_url=base_url,
        model=cfg.get("model", DEFAULT_OPENAI_MODEL),
        headers={"Authorization": auth_value},
    )


def _resolve_custom_api_key(cfg: dict[str, Any]) -> str:
    explicit = str(cfg.get("api_key", "")).strip()
    if explicit:
        return explicit

    env_name = str(cfg.get("api_key_env", "")).strip()
    if env_name:
        return os.environ.get(env_name, "")

    return os.environ.get("OPENAI_API_KEY", "")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))
