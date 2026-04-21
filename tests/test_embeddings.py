"""Embedding resolver and cosine similarity tests (no network / no model loads)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from evalytic.exceptions import ValidationError
from evalytic.text import embeddings as emb_module
from evalytic.text.embeddings import (
    APIEmbedder,
    SentenceTransformerEmbedder,
    cosine_similarity,
    resolve_embedder,
)


class _FakeLocalEmbedder:
    """Stand-in for SentenceTransformerEmbedder."""

    instances: list["_FakeLocalEmbedder"] = []

    def __init__(self, model: str = "dummy") -> None:
        self.model_name = model
        _FakeLocalEmbedder.instances.append(self)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def close(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("FAL_KEY", raising=False)
    _FakeLocalEmbedder.instances.clear()


class TestResolveEmbedder:
    def test_prefers_sentence_transformers_when_import_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(emb_module, "SentenceTransformerEmbedder", _FakeLocalEmbedder)
        embedder = resolve_embedder()
        assert isinstance(embedder, _FakeLocalEmbedder)

    def test_falls_back_to_openai_when_local_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_import(*_a, **_kw):
            raise RuntimeError("no sentence-transformers")

        monkeypatch.setattr(emb_module, "SentenceTransformerEmbedder", raise_import)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        embedder = resolve_embedder()
        assert isinstance(embedder, APIEmbedder)
        assert "api.openai.com" in embedder.base_url
        embedder.close()

    def test_uses_custom_base_url_with_openai_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_import(*_a, **_kw):
            raise RuntimeError("no sentence-transformers")

        monkeypatch.setattr(emb_module, "SentenceTransformerEmbedder", raise_import)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        embedder = resolve_embedder(
            config={"embeddings": {"base_url": "https://api.parel.cloud/v1", "model": "gemini-embedding-2"}}
        )
        assert isinstance(embedder, APIEmbedder)
        assert embedder.base_url == "https://api.parel.cloud/v1"
        assert embedder.model == "gemini-embedding-2"
        assert embedder.headers["Authorization"] == "Bearer sk-test"
        embedder.close()

    def test_uses_custom_api_key_env_and_auth_scheme(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_import(*_a, **_kw):
            raise RuntimeError("no sentence-transformers")

        monkeypatch.setattr(emb_module, "SentenceTransformerEmbedder", raise_import)
        monkeypatch.setenv("PAREL_TEST_API_KEY", "pk-test")
        embedder = resolve_embedder(
            config={
                "embeddings": {
                    "base_url": "https://api.parel.cloud/v1",
                    "model": "gemini-embedding-2",
                    "api_key_env": "PAREL_TEST_API_KEY",
                    "auth_scheme": "bearer",
                }
            }
        )
        assert isinstance(embedder, APIEmbedder)
        assert embedder.headers["Authorization"] == "Bearer pk-test"
        embedder.close()

    def test_custom_base_url_without_api_key_raises_validation_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_import(*_a, **_kw):
            raise RuntimeError("no sentence-transformers")

        monkeypatch.setattr(emb_module, "SentenceTransformerEmbedder", raise_import)
        with pytest.raises(ValidationError):
            resolve_embedder(config={"embeddings": {"base_url": "https://api.parel.cloud/v1"}})

    def test_falls_back_to_fal_when_no_openai(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_import(*_a, **_kw):
            raise RuntimeError("no sentence-transformers")

        monkeypatch.setattr(emb_module, "SentenceTransformerEmbedder", raise_import)
        monkeypatch.setenv("FAL_KEY", "fal-test")
        embedder = resolve_embedder()
        assert isinstance(embedder, APIEmbedder)
        assert "fal.run" in embedder.base_url
        embedder.close()

    def test_forced_local_raises_when_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_import(*_a, **_kw):
            raise RuntimeError("no sentence-transformers")

        monkeypatch.setattr(emb_module, "SentenceTransformerEmbedder", raise_import)
        with pytest.raises(RuntimeError):
            resolve_embedder(config={"embeddings": {"provider": "local"}})

    def test_raises_validation_error_when_all_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_import(*_a, **_kw):
            raise RuntimeError("no sentence-transformers")

        monkeypatch.setattr(emb_module, "SentenceTransformerEmbedder", raise_import)
        with pytest.raises(ValidationError):
            resolve_embedder()


class TestAPIEmbedder:
    def test_captures_parel_cost_header(self) -> None:
        embedder = APIEmbedder(
            base_url="https://api.parel.cloud/v1",
            model="gemini-embedding-2",
            headers={"Authorization": "Bearer test"},
        )
        response = MagicMock()
        response.headers = {"X-Parel-Cost": "0.042"}
        response.raise_for_status = MagicMock()
        response.json.return_value = {"data": [{"embedding": [1.0, 0.0]}]}
        embedder._client.post = MagicMock(return_value=response)

        vectors = embedder.embed_texts(["hello"])

        assert vectors == [[1.0, 0.0]]
        assert embedder.last_cost == pytest.approx(0.042, rel=1e-6)
        embedder.close()


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0, rel=1e-6)

    def test_orthogonal_vectors(self) -> None:
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_zero_vector_returns_0(self) -> None:
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_opposite_vectors_clamped_to_0(self) -> None:
        # Cosine of opposite vectors is -1, but embeddings helper clamps to [0,1]
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == 0.0
