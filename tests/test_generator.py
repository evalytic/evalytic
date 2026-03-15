"""Tests for evalytic.bench.generator -- all offline with mocks."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evalytic.bench.generator import GenerationResult, _download_image
from evalytic.bench.registry import ModelEntry


@pytest.fixture()
def entry() -> ModelEntry:
    return ModelEntry("flux-schnell", "fal-ai/flux/schnell", "text2img", 0.003)


@pytest.fixture()
def mock_fal_client():
    """Install a fake fal_client module so generate_single can import it."""
    mod = types.ModuleType("fal_client")
    mod.subscribe = MagicMock()
    sys.modules["fal_client"] = mod
    yield mod
    sys.modules.pop("fal_client", None)


class TestFalClientOptional:
    def test_missing_fal_client_raises_generation_error(self, entry: ModelEntry) -> None:
        from evalytic.exceptions import GenerationError

        import builtins
        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "fal_client":
                raise ImportError("No module named 'fal_client'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=blocked_import):
            from evalytic.bench.generator import generate_single

            with pytest.raises(GenerationError, match="fal-client"):
                generate_single(entry, {"prompt": "A cat"}, "item-001")


class TestGenerateSingle:
    def test_success(self, entry: ModelEntry, tmp_path: Path, mock_fal_client: types.ModuleType) -> None:
        from evalytic.bench.generator import generate_single

        mock_fal_client.subscribe = MagicMock(
            return_value={"images": [{"url": "https://example.com/img.jpg"}]}
        )
        with patch("evalytic.bench.generator._download_image", return_value=str(tmp_path / "img.jpg")):
            result = generate_single(entry, {"prompt": "A cat"}, "item-001", tmp_path)

        assert result.status == "success"
        assert result.image_url == "https://example.com/img.jpg"
        assert result.model == "flux-schnell"
        assert result.item_id == "item-001"
        assert result.generation_time_ms >= 0

    def test_image_dict_response(self, entry: ModelEntry, mock_fal_client: types.ModuleType) -> None:
        from evalytic.bench.generator import generate_single

        mock_fal_client.subscribe = MagicMock(
            return_value={"image": {"url": "https://example.com/out.png"}}
        )
        result = generate_single(entry, {"prompt": "A dog"}, "item-002")

        assert result.status == "success"
        assert result.image_url == "https://example.com/out.png"

    def test_unexpected_response_fails(self, entry: ModelEntry, mock_fal_client: types.ModuleType) -> None:
        from evalytic.bench.generator import generate_single

        mock_fal_client.subscribe = MagicMock(return_value={"data": "unexpected"})
        result = generate_single(entry, {"prompt": "A cat"}, "item-003")

        assert result.status == "failed"
        assert "Unexpected response shape" in result.error

    def test_exception_returns_failed(self, entry: ModelEntry, mock_fal_client: types.ModuleType) -> None:
        from evalytic.bench.generator import generate_single

        mock_fal_client.subscribe = MagicMock(side_effect=RuntimeError("API timeout"))
        result = generate_single(entry, {"prompt": "A cat"}, "item-004")

        assert result.status == "failed"
        assert "API timeout" in result.error
        assert result.generation_time_ms >= 0

    def test_no_cache_dir(self, entry: ModelEntry, mock_fal_client: types.ModuleType) -> None:
        from evalytic.bench.generator import generate_single

        mock_fal_client.subscribe = MagicMock(
            return_value={"images": [{"url": "https://example.com/img.jpg"}]}
        )
        result = generate_single(entry, {"prompt": "A cat"}, "item-005", cache_dir=None)

        assert result.local_path == ""


class TestGenerateBatch:
    def test_parallel_execution(self, entry: ModelEntry, mock_fal_client: types.ModuleType) -> None:
        from evalytic.bench.generator import generate_batch

        mock_fal_client.subscribe = MagicMock(
            return_value={"images": [{"url": "https://example.com/img.jpg"}]}
        )
        items = [
            {"item_id": "item-001", "arguments": {"prompt": "A cat"}},
            {"item_id": "item-002", "arguments": {"prompt": "A dog"}},
        ]
        progress_calls: list[GenerationResult] = []

        results = generate_batch(
            entry, items, concurrency=2, on_progress=progress_calls.append,
        )

        assert len(results) == 2
        assert len(progress_calls) == 2
        assert all(r.status == "success" for r in results)

    def test_partial_failure(self, entry: ModelEntry, mock_fal_client: types.ModuleType) -> None:
        """First item fails on both attempts (original + retry), second succeeds."""
        from evalytic.bench.generator import generate_batch

        call_count = 0

        def mock_subscribe(endpoint, arguments, client_timeout=300):
            nonlocal call_count
            call_count += 1
            # Calls 1-2 are item-001 (attempt + retry) — both fail
            if call_count <= 2:
                raise RuntimeError("Fail")
            return {"images": [{"url": "https://example.com/ok.jpg"}]}

        mock_fal_client.subscribe = mock_subscribe
        items = [
            {"item_id": "item-001", "arguments": {"prompt": "A cat"}},
            {"item_id": "item-002", "arguments": {"prompt": "A dog"}},
        ]

        results = generate_batch(entry, items, concurrency=1)

        statuses = sorted(r.status for r in results)
        assert statuses == ["failed", "success"]

    def test_retry_recovers(self, entry: ModelEntry, mock_fal_client: types.ModuleType) -> None:
        """First attempt fails, retry succeeds."""
        from evalytic.bench.generator import generate_single

        call_count = 0

        def mock_subscribe(endpoint, arguments, client_timeout=300):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Transient error")
            return {"images": [{"url": "https://example.com/ok.jpg"}]}

        mock_fal_client.subscribe = mock_subscribe
        result = generate_single(entry, {"prompt": "A cat"}, "item-001")

        assert result.status == "success"
        assert result.retried is True
        assert call_count == 2


class TestDownloadImage:
    def test_download_jpg(self, tmp_path: Path) -> None:
        import httpx

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.content = b"\xff\xd8\xff\xe0fake-jpg"

        with patch("evalytic.bench.generator.httpx") as mock_httpx:
            mock_httpx.get.return_value = mock_resp
            path = _download_image(
                "https://example.com/photo.jpg", tmp_path, "flux-schnell", "item-001",
            )

        assert path.endswith(".jpg")
        assert Path(path).exists()

    def test_download_png(self, tmp_path: Path) -> None:
        import httpx

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.content = b"\x89PNGfake-png"

        with patch("evalytic.bench.generator.httpx") as mock_httpx:
            mock_httpx.get.return_value = mock_resp
            path = _download_image(
                "https://example.com/photo.png", tmp_path, "flux-pro", "item-002",
            )

        assert path.endswith(".png")

    def test_creates_model_subdirectory(self, tmp_path: Path) -> None:
        import httpx

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.content = b"data"

        with patch("evalytic.bench.generator.httpx") as mock_httpx:
            mock_httpx.get.return_value = mock_resp
            _download_image(
                "https://example.com/img.jpg", tmp_path, "new-model", "item-001",
            )

        assert (tmp_path / "new-model").is_dir()
