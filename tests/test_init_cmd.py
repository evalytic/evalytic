"""Tests for evalytic init command -- all offline with mocks."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from evalytic.cli.main import cli


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


class TestInitHelp:
    def test_help_output(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["init", "--help"])
        assert result.exit_code == 0
        assert "--force" in result.output
        assert "--skip-demo" in result.output
        assert "wizard" in result.output.lower()


class TestInitTomlCreation:
    def test_creates_toml_with_keys(self, runner: CliRunner, tmp_path: Path) -> None:
        """Full wizard flow: picks use case 1, provides keys, writes toml."""
        # Simulate: use case 1, gemini key, fal key, skip demo
        user_input = "1\nAItest1234567890\nfal_test_key_abcdef\nn\n"

        with runner.isolated_filesystem(temp_dir=tmp_path):
            with patch("evalytic.cli.init_cmd._validate_gemini_key", return_value=True), \
                 patch("evalytic.cli.init_cmd._validate_fal_key", return_value=True), \
                 patch.dict("os.environ", {}, clear=False), \
                 patch("evalytic.cli.main.load_dotenv"):
                # Remove keys from env to force prompting
                env = {k: v for k, v in __import__("os").environ.items()
                       if k not in ("GEMINI_API_KEY", "FAL_KEY")}
                with patch.dict("os.environ", env, clear=True):
                    result = runner.invoke(cli, ["init", "--skip-demo"], input=user_input)

            assert result.exit_code == 0
            toml_path = Path("evalytic.toml")
            assert toml_path.exists()
            content = toml_path.read_text()
            assert "[keys]" in content
            assert "[bench]" in content
            assert 'judge = "gemini-2.5-flash"' in content

    def test_skip_overwrite_existing(self, runner: CliRunner, tmp_path: Path) -> None:
        """Don't overwrite existing toml when user says no."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path("evalytic.toml").write_text("[bench]\njudge = \"old\"\n")
            # Answer "n" to overwrite
            result = runner.invoke(cli, ["init"], input="n\n")
            assert result.exit_code == 0
            assert "Keeping existing" in result.output
            assert Path("evalytic.toml").read_text().startswith("[bench]")

    def test_force_overwrites(self, runner: CliRunner, tmp_path: Path) -> None:
        """--force overwrites without asking."""
        # Use case 3 (no fal needed), provide gemini key
        user_input = "3\nAItest1234567890\n"

        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path("evalytic.toml").write_text("[bench]\njudge = \"old\"\n")

            env = {k: v for k, v in __import__("os").environ.items()
                   if k not in ("GEMINI_API_KEY", "FAL_KEY")}
            with patch.dict("os.environ", env, clear=True), \
                 patch("evalytic.cli.init_cmd._validate_gemini_key", return_value=True), \
                 patch("evalytic.cli.main.load_dotenv"):
                result = runner.invoke(cli, ["init", "--force", "--skip-demo"], input=user_input)

            assert result.exit_code == 0
            content = Path("evalytic.toml").read_text()
            assert 'judge = "gemini-2.5-flash"' in content


class TestInitEnvKeyDetection:
    def test_detects_existing_env_keys(self, runner: CliRunner, tmp_path: Path) -> None:
        """When keys are in env, wizard skips prompting for them."""
        user_input = "1\n"  # Just use case selection

        with runner.isolated_filesystem(temp_dir=tmp_path):
            with patch.dict("os.environ", {
                "GEMINI_API_KEY": "AIexisting123",
                "FAL_KEY": "fal_existing_456",
            }), patch("evalytic.cli.main.load_dotenv"):
                result = runner.invoke(cli, ["init", "--force", "--skip-demo"], input=user_input)

            assert result.exit_code == 0
            assert "from environment" in result.output


class TestInitUseCases:
    def test_use_case_3_no_fal(self, runner: CliRunner, tmp_path: Path) -> None:
        """Use case 3 (score existing) should not ask for FAL key."""
        user_input = "3\nAItest1234567890\n"

        with runner.isolated_filesystem(temp_dir=tmp_path):
            env = {k: v for k, v in __import__("os").environ.items()
                   if k not in ("GEMINI_API_KEY", "FAL_KEY")}
            with patch.dict("os.environ", env, clear=True), \
                 patch("evalytic.cli.init_cmd._validate_gemini_key", return_value=True), \
                 patch("evalytic.cli.main.load_dotenv"):
                result = runner.invoke(cli, ["init", "--force", "--skip-demo"], input=user_input)

            assert result.exit_code == 0
            content = Path("evalytic.toml").read_text()
            assert "gemini" in content
            # FAL key should NOT be in toml since use case 3 doesn't need it
            assert "fal" not in content.lower() or "fal" not in content.split("[keys]")[-1].split("[")[0]


class TestInitHelpers:
    def test_mask_key_short(self) -> None:
        from evalytic.cli.init_cmd import _mask_key

        assert _mask_key("short") == "***"

    def test_mask_key_long(self) -> None:
        from evalytic.cli.init_cmd import _mask_key

        result = _mask_key("AIzaSyBtest1234567890abcdef")
        assert result.startswith("AIza")
        assert result.endswith("cdef")
        assert "***" in result

    def test_write_toml_structure(self, tmp_path: Path) -> None:
        from evalytic.cli.init_cmd import _write_toml

        path = tmp_path / "evalytic.toml"
        _write_toml(path, {"gemini": "AItest123", "fal": "fal_test"})
        content = path.read_text()
        assert "[keys]" in content
        assert 'gemini = "AItest123"' in content
        assert 'fal = "fal_test"' in content
        assert "[bench]" in content

    def test_write_toml_empty_keys(self, tmp_path: Path) -> None:
        from evalytic.cli.init_cmd import _write_toml

        path = tmp_path / "evalytic.toml"
        _write_toml(path, {})
        content = path.read_text()
        assert "[keys]" not in content
        assert "[bench]" in content
