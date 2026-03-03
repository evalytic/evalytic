"""Tests for ``evalytic demo`` command."""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from evalytic.cli.demo_cmd import SHOWCASE_URL, _DEMOS, demo_cmd


@patch("evalytic.cli.demo_cmd.webbrowser.open")
def test_demo_no_args(mock_open):
    """evaly demo → opens showcase page."""
    result = CliRunner().invoke(demo_cmd, [])
    assert result.exit_code == 0
    mock_open.assert_called_once_with(SHOWCASE_URL)


@patch("evalytic.cli.demo_cmd.webbrowser.open")
def test_demo_with_case(mock_open):
    """evaly demo face → opens face report URL."""
    result = CliRunner().invoke(demo_cmd, ["face"])
    assert result.exit_code == 0
    mock_open.assert_called_once_with(_DEMOS["face"]["url"])


def test_demo_invalid_case():
    """evaly demo foobar → Click error."""
    result = CliRunner().invoke(demo_cmd, ["foobar"])
    assert result.exit_code != 0


@patch("evalytic.cli.demo_cmd.webbrowser.open")
def test_demo_output(mock_open):
    """Terminal output contains showcase title and case numbering."""
    result = CliRunner().invoke(demo_cmd, [])
    assert "Benchmark Showcase" in result.output
    assert "01" in result.output
    assert "04" in result.output
    assert SHOWCASE_URL in result.output
