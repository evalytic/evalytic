"""Tests for evalytic eval command -- all offline with mocks."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from evalytic.cli.main import cli


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


class TestEvalHelp:
    def test_help_output(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["eval", "--help"])
        assert result.exit_code == 0
        assert "--image" in result.output
        assert "--dimensions" in result.output
        assert "--prompt" in result.output
        assert "--input-image" in result.output


class TestEvalValidation:
    def test_missing_image_fails(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["eval", "-d", "visual_quality"])
        assert result.exit_code != 0

    def test_unknown_dimension_exits_2(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["eval", "--image", "photo.jpg", "-d", "bogus_dim"])
        assert result.exit_code == 2
        assert "Unknown dimensions" in result.output

    def test_prompt_adherence_without_prompt_exits_2(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["eval", "--image", "photo.jpg", "-d", "prompt_adherence"])
        assert result.exit_code == 2
        assert "requires --prompt" in result.output

    def test_input_fidelity_without_input_image_exits_2(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["eval", "--image", "photo.jpg", "-d", "input_fidelity"])
        assert result.exit_code == 2
        assert "requires --input-image" in result.output

    def test_missing_gemini_key_exits_2(self, runner: CliRunner) -> None:
        env = {k: v for k, v in __import__("os").environ.items() if k != "GEMINI_API_KEY"}
        with patch.dict("os.environ", env, clear=True), \
             patch("evalytic.cli.main.load_dotenv"), \
             patch("evalytic.config.apply_keys"):
            result = runner.invoke(cli, ["eval", "--image", "photo.jpg", "-d", "visual_quality"])
        assert result.exit_code == 2
        assert "GEMINI_API_KEY" in result.output


class TestEvalScoring:
    def test_successful_eval(self, runner: CliRunner) -> None:
        from evalytic.bench.types import DimensionResult

        mock_results = [
            DimensionResult(dimension="visual_quality", score=4.2, explanation="Good quality"),
        ]

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), \
             patch("evalytic.cli.eval_cmd.Judge") as MockJudge:
            instance = MockJudge.return_value.__enter__.return_value
            instance.score.return_value = mock_results

            result = runner.invoke(cli, ["eval", "--image", "photo.jpg", "-d", "visual_quality"])

        assert result.exit_code == 0
        assert "visual_quality" in result.output
        assert "4.2" in result.output
        assert "Conf" in result.output  # Confidence column header
        assert "100%" in result.output  # Default confidence

    def test_confidence_column_with_custom_value(self, runner: CliRunner) -> None:
        from evalytic.bench.types import DimensionResult

        mock_results = [
            DimensionResult(dimension="visual_quality", score=4.0, confidence=0.72, explanation="Ok"),
        ]

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), \
             patch("evalytic.cli.eval_cmd.Judge") as MockJudge:
            instance = MockJudge.return_value.__enter__.return_value
            instance.score.return_value = mock_results

            result = runner.invoke(cli, ["eval", "--image", "photo.jpg", "-d", "visual_quality"])

        assert result.exit_code == 0
        assert "72%" in result.output

    def test_json_output(self, runner: CliRunner, tmp_path: Path) -> None:
        from evalytic.bench.types import DimensionResult

        mock_results = [
            DimensionResult(
                dimension="visual_quality", score=4.0,
                explanation="Good", evidence=["clean rendering"],
            ),
        ]
        out_file = tmp_path / "result.json"

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), \
             patch("evalytic.cli.eval_cmd.Judge") as MockJudge:
            instance = MockJudge.return_value.__enter__.return_value
            instance.score.return_value = mock_results

            result = runner.invoke(cli, [
                "eval", "--image", "photo.jpg",
                "-d", "visual_quality",
                "-o", str(out_file),
            ])

        assert result.exit_code == 0
        data = json.loads(out_file.read_text())
        assert data["image"] == "photo.jpg"
        assert len(data["dimensions"]) == 1
        assert data["dimensions"][0]["score"] == 4.0
        assert data["dimensions"][0]["confidence"] == 1.0

    def test_json_output_custom_confidence(self, runner: CliRunner, tmp_path: Path) -> None:
        from evalytic.bench.types import DimensionResult

        mock_results = [
            DimensionResult(
                dimension="visual_quality", score=3.5,
                confidence=0.65, explanation="Meh",
            ),
        ]
        out_file = tmp_path / "result.json"

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), \
             patch("evalytic.cli.eval_cmd.Judge") as MockJudge:
            instance = MockJudge.return_value.__enter__.return_value
            instance.score.return_value = mock_results

            result = runner.invoke(cli, [
                "eval", "--image", "photo.jpg",
                "-d", "visual_quality",
                "-o", str(out_file),
            ])

        assert result.exit_code == 0
        data = json.loads(out_file.read_text())
        assert data["dimensions"][0]["confidence"] == 0.65

    def test_eval_with_prompt(self, runner: CliRunner) -> None:
        from evalytic.bench.types import DimensionResult

        mock_results = [
            DimensionResult(dimension="prompt_adherence", score=3.5, explanation="Decent"),
        ]

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), \
             patch("evalytic.cli.eval_cmd.Judge") as MockJudge:
            instance = MockJudge.return_value.__enter__.return_value
            instance.score.return_value = mock_results

            result = runner.invoke(cli, [
                "eval", "--image", "photo.jpg",
                "-d", "prompt_adherence",
                "--prompt", "A cat wearing a hat",
            ])

        assert result.exit_code == 0
        instance.score.assert_called_once_with(
            image_url="photo.jpg",
            dimensions=["prompt_adherence"],
            prompt="A cat wearing a hat",
            input_image_url=None,
        )


class TestEvalAutoDetect:
    def test_no_dimensions_defaults_to_visual_quality(self, runner: CliRunner) -> None:
        """No -d flag, no --prompt, no --input-image -> visual_quality only."""
        from evalytic.bench.types import DimensionResult

        mock_results = [
            DimensionResult(dimension="visual_quality", score=4.0, explanation="Good"),
        ]

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), \
             patch("evalytic.cli.eval_cmd.Judge") as MockJudge:
            instance = MockJudge.return_value.__enter__.return_value
            instance.score.return_value = mock_results

            result = runner.invoke(cli, ["eval", "--image", "photo.jpg"])

        assert result.exit_code == 0
        instance.score.assert_called_once_with(
            image_url="photo.jpg",
            dimensions=["visual_quality"],
            prompt=None,
            input_image_url=None,
        )

    def test_prompt_adds_prompt_adherence(self, runner: CliRunner) -> None:
        """--prompt without -d -> visual_quality + prompt_adherence."""
        from evalytic.bench.types import DimensionResult

        mock_results = [
            DimensionResult(dimension="visual_quality", score=4.0, explanation="Good"),
            DimensionResult(dimension="prompt_adherence", score=3.5, explanation="Decent"),
        ]

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), \
             patch("evalytic.cli.eval_cmd.Judge") as MockJudge:
            instance = MockJudge.return_value.__enter__.return_value
            instance.score.return_value = mock_results

            result = runner.invoke(cli, [
                "eval", "--image", "photo.jpg", "--prompt", "A cat",
            ])

        assert result.exit_code == 0
        called_dims = instance.score.call_args.kwargs["dimensions"]
        assert "visual_quality" in called_dims
        assert "prompt_adherence" in called_dims

    def test_text_keyword_adds_text_rendering(self, runner: CliRunner) -> None:
        """Prompt with text keyword -> also text_rendering."""
        from evalytic.bench.types import DimensionResult

        mock_results = [
            DimensionResult(dimension="visual_quality", score=4.0, explanation="Good"),
            DimensionResult(dimension="prompt_adherence", score=3.5, explanation="Ok"),
            DimensionResult(dimension="text_rendering", score=3.0, explanation="Ok"),
        ]

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), \
             patch("evalytic.cli.eval_cmd.Judge") as MockJudge:
            instance = MockJudge.return_value.__enter__.return_value
            instance.score.return_value = mock_results

            result = runner.invoke(cli, [
                "eval", "--image", "photo.jpg",
                "--prompt", "Write the word HELLO in bold letters",
            ])

        assert result.exit_code == 0
        called_dims = instance.score.call_args.kwargs["dimensions"]
        assert "text_rendering" in called_dims

    def test_input_image_adds_img2img_dims(self, runner: CliRunner) -> None:
        """--input-image -> visual_quality + img2img dimensions."""
        from evalytic.bench.types import DimensionResult

        mock_results = [
            DimensionResult(dimension="visual_quality", score=4.0, explanation="Good"),
            DimensionResult(dimension="input_fidelity", score=3.5, explanation="Ok"),
            DimensionResult(dimension="transformation_quality", score=3.8, explanation="Ok"),
            DimensionResult(dimension="artifact_detection", score=4.2, explanation="Clean"),
        ]

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), \
             patch("evalytic.cli.eval_cmd.Judge") as MockJudge:
            instance = MockJudge.return_value.__enter__.return_value
            instance.score.return_value = mock_results

            result = runner.invoke(cli, [
                "eval", "--image", "out.jpg", "--input-image", "in.jpg",
            ])

        assert result.exit_code == 0
        called_dims = instance.score.call_args.kwargs["dimensions"]
        assert "input_fidelity" in called_dims
        assert "transformation_quality" in called_dims
        assert "artifact_detection" in called_dims

    def test_explicit_d_overrides_auto(self, runner: CliRunner) -> None:
        """Explicit -d should override auto-detection."""
        from evalytic.bench.types import DimensionResult

        mock_results = [
            DimensionResult(dimension="visual_quality", score=4.0, explanation="Good"),
        ]

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}), \
             patch("evalytic.cli.eval_cmd.Judge") as MockJudge:
            instance = MockJudge.return_value.__enter__.return_value
            instance.score.return_value = mock_results

            result = runner.invoke(cli, [
                "eval", "--image", "photo.jpg",
                "--prompt", "A cat",
                "-d", "visual_quality",  # explicit, should NOT add prompt_adherence
            ])

        assert result.exit_code == 0
        called_dims = instance.score.call_args.kwargs["dimensions"]
        assert called_dims == ["visual_quality"]


class TestLocalFileSupport:
    def test_local_file_scoring(self, tmp_path: Path) -> None:
        """Judge should read local files instead of making HTTP requests."""
        from evalytic.bench.judge import GeminiJudge

        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\nfake-png-data")

        judge = GeminiJudge(api_key="test-key")
        b64, mime = judge._fetch_image_base64(str(img))

        import base64
        assert base64.b64decode(b64) == b"\x89PNG\r\n\x1a\nfake-png-data"
        assert mime == "image/png"

    def test_local_file_not_found(self) -> None:
        from evalytic.bench.judge import GeminiJudge
        from evalytic.exceptions import ValidationError

        judge = GeminiJudge(api_key="test-key")
        with pytest.raises(ValidationError, match="Image file not found"):
            judge._fetch_image_base64("/nonexistent/image.jpg")
