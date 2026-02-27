"""Tests for evalytic.bench.judge -- all offline with mocks."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from evalytic.bench.judge import (
    DIMENSION_CONFIG,
    Judge,
    _parse_judge_string,
)

# Backward-compat alias still importable
from evalytic.bench.judge import GeminiJudge  # noqa: F401


@pytest.fixture()
def judge() -> Judge:
    return Judge(judge="gemini-2.5-flash", api_key="test-key")


class TestDimensionValidation:
    def test_unknown_dimension_raises(self, judge: Judge) -> None:
        with pytest.raises(ValueError, match="Unknown dimensions"):
            judge.score("https://example.com/img.jpg", ["invalid_dim"])

    def test_multiple_unknown_dimensions(self, judge: Judge) -> None:
        with pytest.raises(ValueError, match="Unknown dimensions") as exc_info:
            judge.score("https://example.com/img.jpg", ["bad_one", "bad_two"])
        assert "bad_one" in str(exc_info.value)
        assert "bad_two" in str(exc_info.value)

    def test_valid_dimensions_accepted(self, judge: Judge) -> None:
        """Valid dimensions should not raise (we mock the actual API call)."""
        gemini_response = {
            "candidates": [{"content": {"parts": [{"text": json.dumps({
                "dimensions": [{"dimension": "visual_quality", "score": 4.0,
                                "explanation": "Good", "evidence": ["clean"]}]
            })}]}}]
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = gemini_response
        mock_resp.raise_for_status = MagicMock()

        with patch.object(judge._client, "get") as mock_get, \
             patch.object(judge._client, "post", return_value=mock_resp):
            mock_get.return_value = MagicMock(
                content=b"fake-image",
                headers={"content-type": "image/jpeg"},
            )
            mock_get.return_value.raise_for_status = MagicMock()
            results = judge.score("https://example.com/img.jpg", ["visual_quality"])

        assert len(results) == 1
        assert results[0].dimension == "visual_quality"
        assert results[0].score == 4.0


class TestParseResponse:
    def test_clean_json(self) -> None:
        text = json.dumps({"dimensions": [{"dimension": "visual_quality", "score": 4}]})
        result = GeminiJudge._parse_response(text)
        assert result["dimensions"][0]["score"] == 4

    def test_markdown_fence_json(self) -> None:
        text = '```json\n{"dimensions": [{"dimension": "visual_quality", "score": 3}]}\n```'
        result = GeminiJudge._parse_response(text)
        assert result["dimensions"][0]["score"] == 3

    def test_markdown_fence_no_lang(self) -> None:
        text = '```\n{"dimensions": [{"dimension": "visual_quality", "score": 2}]}\n```'
        result = GeminiJudge._parse_response(text)
        assert result["dimensions"][0]["score"] == 2

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            GeminiJudge._parse_response("not json at all")


class TestScore:
    def test_text2img_scoring(self, judge: Judge) -> None:
        gemini_response = {
            "candidates": [{"content": {"parts": [{"text": json.dumps({
                "dimensions": [{"dimension": "visual_quality", "score": 4.5,
                                "explanation": "Excellent", "evidence": ["sharp"]}]
            })}]}}]
        }

        mock_post = MagicMock()
        mock_post.json.return_value = gemini_response
        mock_post.raise_for_status = MagicMock()

        mock_get = MagicMock()
        mock_get.content = b"fake-image"
        mock_get.headers = {"content-type": "image/jpeg"}
        mock_get.raise_for_status = MagicMock()

        with patch.object(judge._client, "get", return_value=mock_get), \
             patch.object(judge._client, "post", return_value=mock_post):
            results = judge.score(
                "https://example.com/img.jpg",
                ["visual_quality"],
                prompt="A cat",
            )

        assert len(results) == 1
        assert results[0].score == 4.5
        assert results[0].explanation == "Excellent"

    def test_img2img_requires_input_image(self, judge: Judge) -> None:
        """img2img dimensions need input_image_url."""
        with pytest.raises(ValueError, match="requires input_image_url"):
            # Mock the image fetch so we get to the validation
            with patch.object(judge._client, "get") as mock_get:
                mock_get.return_value = MagicMock(
                    content=b"fake", headers={"content-type": "image/jpeg"},
                )
                mock_get.return_value.raise_for_status = MagicMock()
                judge.score(
                    "https://example.com/out.jpg",
                    ["input_fidelity"],
                    input_image_url=None,
                )

    def test_img2img_with_input_image(self, judge: Judge) -> None:
        gemini_response = {
            "candidates": [{"content": {"parts": [{"text": json.dumps({
                "dimensions": [{"dimension": "input_fidelity", "score": 3.5,
                                "explanation": "Decent", "evidence": ["face ok"]}]
            })}]}}]
        }

        mock_post = MagicMock()
        mock_post.json.return_value = gemini_response
        mock_post.raise_for_status = MagicMock()

        mock_get = MagicMock()
        mock_get.content = b"fake-image"
        mock_get.headers = {"content-type": "image/jpeg"}
        mock_get.raise_for_status = MagicMock()

        with patch.object(judge._client, "get", return_value=mock_get), \
             patch.object(judge._client, "post", return_value=mock_post):
            results = judge.score(
                "https://example.com/out.jpg",
                ["input_fidelity"],
                input_image_url="https://example.com/in.jpg",
            )

        assert len(results) == 1
        assert results[0].dimension == "input_fidelity"

    def test_multiple_dimensions(self, judge: Judge) -> None:
        def mock_post_side_effect(*args, **kwargs):
            """Return different scores per call."""
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {
                "candidates": [{"content": {"parts": [{"text": json.dumps({
                    "dimensions": [{"dimension": "visual_quality", "score": 4.0,
                                    "explanation": "ok", "evidence": []}]
                })}]}}]
            }
            return resp

        mock_get = MagicMock()
        mock_get.content = b"fake-image"
        mock_get.headers = {"content-type": "image/jpeg"}
        mock_get.raise_for_status = MagicMock()

        with patch.object(judge._client, "get", return_value=mock_get), \
             patch.object(judge._client, "post", side_effect=mock_post_side_effect):
            results = judge.score(
                "https://example.com/img.jpg",
                ["visual_quality", "prompt_adherence"],
                prompt="A cat",
            )

        assert len(results) == 2


class TestRetryLogic:
    def test_retries_on_failure(self, judge: Judge) -> None:
        """Should retry 3 times before raising JudgeError."""
        from evalytic.exceptions import JudgeError

        mock_get = MagicMock()
        mock_get.content = b"fake-image"
        mock_get.headers = {"content-type": "image/jpeg"}
        mock_get.raise_for_status = MagicMock()

        with patch.object(judge._client, "get", return_value=mock_get), \
             patch.object(judge._client, "post", side_effect=RuntimeError("API error")), \
             patch("evalytic.bench.judge.time.sleep"):
            with pytest.raises(JudgeError, match="API error"):
                judge.score("https://example.com/img.jpg", ["visual_quality"])


class TestFetchImageBase64:
    def test_returns_base64_and_mime(self, judge: Judge) -> None:
        mock_resp = MagicMock()
        mock_resp.content = b"fake-image-data"
        mock_resp.headers = {"content-type": "image/png"}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(judge._client, "get", return_value=mock_resp):
            b64, mime = judge._fetch_image_base64("https://example.com/img.png")

        import base64
        assert base64.b64decode(b64) == b"fake-image-data"
        assert mime == "image/png"

    def test_mime_fallback_from_url(self, judge: Judge) -> None:
        mock_resp = MagicMock()
        mock_resp.content = b"data"
        mock_resp.headers = {}
        mock_resp.raise_for_status = MagicMock()

        with patch.object(judge._client, "get", return_value=mock_resp):
            _, mime = judge._fetch_image_base64("https://example.com/img.webp")

        assert mime == "image/webp"


class TestDimensionConfig:
    def test_all_seven_dimensions_configured(self) -> None:
        expected = {
            "visual_quality", "prompt_adherence", "text_rendering",
            "input_fidelity", "transformation_quality", "artifact_detection",
            "identity_preservation",
        }
        assert set(DIMENSION_CONFIG.keys()) == expected

    def test_text2img_dimensions_no_input(self) -> None:
        for dim in ["visual_quality", "prompt_adherence", "text_rendering"]:
            assert not DIMENSION_CONFIG[dim].needs_input

    def test_img2img_dimensions_need_input(self) -> None:
        for dim in ["input_fidelity", "transformation_quality", "artifact_detection",
                     "identity_preservation"]:
            assert DIMENSION_CONFIG[dim].needs_input

    def test_identity_preservation_no_prompt_needed(self) -> None:
        assert not DIMENSION_CONFIG["identity_preservation"].needs_prompt


class TestIdentityPreservation:
    def test_identity_preservation_scoring(self, judge: Judge) -> None:
        """identity_preservation needs input + output images."""
        gemini_response = {
            "candidates": [{"content": {"parts": [{"text": json.dumps({
                "dimensions": [{"dimension": "identity_preservation", "score": 5.0,
                                "confidence": 1.0,
                                "explanation": "No human faces in input image — identity preservation not applicable.",
                                "evidence": ["No faces detected"]}]
            })}]}}]
        }

        mock_post = MagicMock()
        mock_post.json.return_value = gemini_response
        mock_post.raise_for_status = MagicMock()

        mock_get = MagicMock()
        mock_get.content = b"fake-image"
        mock_get.headers = {"content-type": "image/jpeg"}
        mock_get.raise_for_status = MagicMock()

        with patch.object(judge._client, "get", return_value=mock_get), \
             patch.object(judge._client, "post", return_value=mock_post):
            results = judge.score(
                "https://example.com/out.jpg",
                ["identity_preservation"],
                input_image_url="https://example.com/in.jpg",
            )

        assert len(results) == 1
        assert results[0].dimension == "identity_preservation"
        assert results[0].score == 5.0

    def test_identity_preservation_requires_input(self, judge: Judge) -> None:
        with pytest.raises(ValueError, match="requires input_image_url"):
            with patch.object(judge._client, "get") as mock_get:
                mock_get.return_value = MagicMock(
                    content=b"fake", headers={"content-type": "image/jpeg"},
                )
                mock_get.return_value.raise_for_status = MagicMock()
                judge.score(
                    "https://example.com/out.jpg",
                    ["identity_preservation"],
                    input_image_url=None,
                )


class TestConfidenceParsing:
    def test_confidence_extracted(self, judge: Judge) -> None:
        """VLM response with confidence should be parsed."""
        gemini_response = {
            "candidates": [{"content": {"parts": [{"text": json.dumps({
                "dimensions": [{"dimension": "visual_quality", "score": 4.0,
                                "confidence": 0.85,
                                "explanation": "Good", "evidence": ["clean"]}]
            })}]}}]
        }

        mock_post = MagicMock()
        mock_post.json.return_value = gemini_response
        mock_post.raise_for_status = MagicMock()

        mock_get = MagicMock()
        mock_get.content = b"fake-image"
        mock_get.headers = {"content-type": "image/jpeg"}
        mock_get.raise_for_status = MagicMock()

        with patch.object(judge._client, "get", return_value=mock_get), \
             patch.object(judge._client, "post", return_value=mock_post):
            results = judge.score("https://example.com/img.jpg", ["visual_quality"])

        assert results[0].confidence == 0.85

    def test_confidence_defaults_when_missing(self, judge: Judge) -> None:
        """VLM response without confidence should default to 1.0."""
        gemini_response = {
            "candidates": [{"content": {"parts": [{"text": json.dumps({
                "dimensions": [{"dimension": "visual_quality", "score": 4.0,
                                "explanation": "Good", "evidence": ["clean"]}]
            })}]}}]
        }

        mock_post = MagicMock()
        mock_post.json.return_value = gemini_response
        mock_post.raise_for_status = MagicMock()

        mock_get = MagicMock()
        mock_get.content = b"fake-image"
        mock_get.headers = {"content-type": "image/jpeg"}
        mock_get.raise_for_status = MagicMock()

        with patch.object(judge._client, "get", return_value=mock_get), \
             patch.object(judge._client, "post", return_value=mock_post):
            results = judge.score("https://example.com/img.jpg", ["visual_quality"])

        assert results[0].confidence == 1.0


class TestParseJudgeString:
    def test_bare_gemini(self) -> None:
        assert _parse_judge_string("gemini-2.5-flash") == ("gemini", "gemini-2.5-flash")

    def test_bare_gpt(self) -> None:
        assert _parse_judge_string("gpt-4o") == ("openai", "gpt-4o")

    def test_bare_claude(self) -> None:
        assert _parse_judge_string("claude-sonnet-4") == ("anthropic", "claude-sonnet-4")

    def test_explicit_openai(self) -> None:
        assert _parse_judge_string("openai/gpt-4o") == ("openai", "gpt-4o")

    def test_explicit_ollama(self) -> None:
        assert _parse_judge_string("ollama/qwen2.5-vl:7b") == ("ollama", "qwen2.5-vl:7b")

    def test_explicit_anthropic(self) -> None:
        assert _parse_judge_string("anthropic/claude-sonnet-4") == ("anthropic", "claude-sonnet-4")

    def test_explicit_local(self) -> None:
        assert _parse_judge_string("local/my-model") == ("local", "my-model")

    def test_explicit_lmstudio(self) -> None:
        assert _parse_judge_string("lmstudio/qwen2.5-vl") == ("lmstudio", "qwen2.5-vl")

    def test_unknown_provider_raises(self) -> None:
        from evalytic.exceptions import ValidationError
        with pytest.raises(ValidationError, match="Unknown judge provider"):
            _parse_judge_string("badprovider/model")

    def test_bare_o4_mini(self) -> None:
        assert _parse_judge_string("o4-mini") == ("openai", "o4-mini")

    def test_bare_o3(self) -> None:
        assert _parse_judge_string("o3-pro") == ("openai", "o3-pro")

    def test_bare_chatgpt(self) -> None:
        assert _parse_judge_string("chatgpt-5.2") == ("openai", "chatgpt-5.2")

    def test_bare_gpt5(self) -> None:
        assert _parse_judge_string("gpt-5.2") == ("openai", "gpt-5.2")

    def test_bare_gemini3(self) -> None:
        assert _parse_judge_string("gemini-3-flash") == ("gemini", "gemini-3-flash")

    def test_bare_claude_sonnet_46(self) -> None:
        assert _parse_judge_string("claude-sonnet-4-6") == ("anthropic", "claude-sonnet-4-6")

    def test_unknown_bare_defaults_gemini(self) -> None:
        assert _parse_judge_string("some-random-model") == ("gemini", "some-random-model")


class TestJudgeProviderInit:
    def test_gemini_default(self) -> None:
        j = Judge("gemini-2.5-flash", api_key="k")
        assert j.provider == "gemini"
        assert j.model == "gemini-2.5-flash"
        assert "googleapis" in j.base_url
        j.close()

    def test_openai_provider(self) -> None:
        j = Judge("openai/gpt-4o", api_key="k")
        assert j.provider == "openai"
        assert j.model == "gpt-4o"
        assert "openai.com" in j.base_url
        j.close()

    def test_ollama_no_key_required(self) -> None:
        j = Judge("ollama/qwen2.5-vl:7b")
        assert j.provider == "ollama"
        assert j.api_key == ""
        assert "11434" in j.base_url
        j.close()

    def test_custom_base_url(self) -> None:
        j = Judge("local/my-model", base_url="http://localhost:9999/v1")
        assert j.base_url == "http://localhost:9999/v1"
        j.close()

    def test_backward_compat_alias(self) -> None:
        assert GeminiJudge is Judge


class TestOpenAICompatCall:
    def test_openai_request_format(self) -> None:
        """OpenAI-compat call should use chat/completions format."""
        j = Judge("openai/gpt-4o", api_key="test-key")

        openai_response = {
            "choices": [{"message": {"content": json.dumps({
                "dimensions": [{"dimension": "visual_quality", "score": 4.0,
                                "explanation": "Good", "evidence": ["clean"]}]
            })}}]
        }

        mock_post = MagicMock()
        mock_post.json.return_value = openai_response
        mock_post.raise_for_status = MagicMock()

        mock_get = MagicMock()
        mock_get.content = b"fake-image"
        mock_get.headers = {"content-type": "image/jpeg"}
        mock_get.raise_for_status = MagicMock()

        with patch.object(j._client, "get", return_value=mock_get), \
             patch.object(j._client, "post", return_value=mock_post) as post_call:
            results = j.score("https://example.com/img.jpg", ["visual_quality"])

        assert len(results) == 1
        assert results[0].score == 4.0

        # Verify OpenAI format
        call_args = post_call.call_args
        assert "chat/completions" in call_args.args[0]
        payload = call_args.kwargs["json"]
        assert payload["model"] == "gpt-4o"
        assert "messages" in payload
        j.close()


class TestAnthropicCall:
    def test_anthropic_request_format(self) -> None:
        """Anthropic call should use messages API format."""
        j = Judge("anthropic/claude-sonnet-4", api_key="test-key")

        anthropic_response = {
            "content": [{"text": json.dumps({
                "dimensions": [{"dimension": "visual_quality", "score": 3.5,
                                "explanation": "Ok", "evidence": ["decent"]}]
            })}]
        }

        mock_post = MagicMock()
        mock_post.json.return_value = anthropic_response
        mock_post.raise_for_status = MagicMock()

        mock_get = MagicMock()
        mock_get.content = b"fake-image"
        mock_get.headers = {"content-type": "image/jpeg"}
        mock_get.raise_for_status = MagicMock()

        with patch.object(j._client, "get", return_value=mock_get), \
             patch.object(j._client, "post", return_value=mock_post) as post_call:
            results = j.score("https://example.com/img.jpg", ["visual_quality"])

        assert len(results) == 1
        assert results[0].score == 3.5

        # Verify Anthropic format
        call_args = post_call.call_args
        assert "messages" in call_args.args[0]
        headers = call_args.kwargs["headers"]
        assert headers["x-api-key"] == "test-key"
        assert "anthropic-version" in headers
        j.close()
