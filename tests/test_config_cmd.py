"""Tests for evalytic config command -- all offline with mocks."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from evalytic.cli.main import cli


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


class TestConfigHelp:
    def test_config_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["config", "--help"])
        assert result.exit_code == 0
        assert "show" in result.output

    def test_config_show_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["config", "show", "--help"])
        assert result.exit_code == 0
        assert "configuration" in result.output.lower()


class TestConfigShow:
    def test_shows_api_keys_masked(self, runner: CliRunner) -> None:
        """API keys in env should be shown masked."""
        with patch.dict("os.environ", {
            "GEMINI_API_KEY": "AIzaSyBtest1234567890",
            "FAL_KEY": "fal_test_abcdefghijklmnop",
        }), patch("evalytic.cli.main.load_dotenv"):
            result = runner.invoke(cli, ["config", "show"])

        assert result.exit_code == 0
        assert "AIza" in result.output  # First 4 chars visible
        assert "***" in result.output  # Masked middle
        assert "7890" in result.output  # Last 4 chars visible

    def test_shows_not_set(self, runner: CliRunner) -> None:
        """Missing keys show (not set)."""
        env = {k: v for k, v in __import__("os").environ.items()
               if k not in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")}
        with patch.dict("os.environ", env, clear=True), \
             patch("evalytic.cli.main.load_dotenv"):
            result = runner.invoke(cli, ["config", "show"])

        assert result.exit_code == 0
        assert "(not set)" in result.output

    def test_shows_judge_setting(self, runner: CliRunner) -> None:
        """Judge setting should be shown."""
        with patch("evalytic.cli.main.load_dotenv"):
            result = runner.invoke(cli, ["config", "show"])

        assert result.exit_code == 0
        assert "judge" in result.output
        assert "gemini" in result.output

    def test_shows_config_file_path(self, runner: CliRunner, tmp_path: Path) -> None:
        """Should show path to evalytic.toml when it exists."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path("evalytic.toml").write_text('[bench]\njudge = "gemini-2.0-flash"\n')
            with patch("evalytic.cli.main.load_dotenv"):
                result = runner.invoke(cli, ["config", "show"])

        assert result.exit_code == 0
        assert "Config file:" in result.output

    def test_no_config_file(self, runner: CliRunner, tmp_path: Path) -> None:
        """When no config file, suggest evalytic init."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            with patch("evalytic.cli.main.load_dotenv"):
                result = runner.invoke(cli, ["config", "show"])

        assert result.exit_code == 0
        assert "evalytic init" in result.output


class TestConfigKeyMasking:
    def test_mask_short_key(self) -> None:
        from evalytic.cli.config_cmd import _mask_key

        assert _mask_key("abc") == "***"

    def test_mask_normal_key(self) -> None:
        from evalytic.cli.config_cmd import _mask_key

        result = _mask_key("AIzaSyBtest1234567890abcdef")
        assert result == "AIza***cdef"

    def test_mask_boundary(self) -> None:
        from evalytic.cli.config_cmd import _mask_key

        assert _mask_key("12345678") == "***"  # exactly 8 chars
        result = _mask_key("123456789")  # 9 chars
        assert result == "1234***6789"
