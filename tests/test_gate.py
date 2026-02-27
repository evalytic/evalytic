"""Tests for evalytic gate command -- all offline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from evalytic.cli.main import cli


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def sample_report(tmp_path: Path) -> Path:
    """Create a minimal bench report JSON."""
    data = {
        "name": "test-bench",
        "version": "1.0",
        "summary": {
            "flux-schnell": {
                "overall_score": 3.8,
                "dimension_averages": {
                    "visual_quality": 4.0,
                    "prompt_adherence": 3.6,
                },
            },
            "flux-pro": {
                "overall_score": 4.5,
                "dimension_averages": {
                    "visual_quality": 4.8,
                    "prompt_adherence": 4.2,
                },
            },
        },
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture()
def baseline_report(tmp_path: Path) -> Path:
    """Create a baseline bench report for regression testing."""
    data = {
        "name": "baseline-bench",
        "version": "1.0",
        "summary": {
            "flux-schnell": {
                "overall_score": 4.0,
                "dimension_averages": {
                    "visual_quality": 4.2,
                    "prompt_adherence": 3.8,
                },
            },
            "flux-pro": {
                "overall_score": 4.6,
                "dimension_averages": {
                    "visual_quality": 4.9,
                    "prompt_adherence": 4.3,
                },
            },
        },
    }
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(data))
    return path


class TestGateHelp:
    def test_help_output(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["gate", "--help"])
        assert result.exit_code == 0
        assert "--report" in result.output
        assert "--threshold" in result.output
        assert "--dimension-threshold" in result.output
        assert "--baseline" in result.output


class TestGateErrors:
    def test_file_not_found_exits_2(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["gate", "--report", "/nonexistent/report.json", "--threshold", "3.0"])
        assert result.exit_code == 2
        assert "not found" in result.output

    def test_invalid_json_exits_2(self, runner: CliRunner, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json{{{")
        result = runner.invoke(cli, ["gate", "--report", str(bad_file), "--threshold", "3.0"])
        assert result.exit_code == 2
        assert "Invalid JSON" in result.output

    def test_invalid_report_format_exits_2(self, runner: CliRunner, tmp_path: Path) -> None:
        no_summary = tmp_path / "nosummary.json"
        no_summary.write_text(json.dumps({"name": "test"}))
        result = runner.invoke(cli, ["gate", "--report", str(no_summary), "--threshold", "3.0"])
        assert result.exit_code == 2
        assert "missing 'summary'" in result.output


class TestGateNoChecks:
    def test_no_checks_warns_and_passes(self, runner: CliRunner, sample_report: Path) -> None:
        result = runner.invoke(cli, ["gate", "--report", str(sample_report)])
        assert result.exit_code == 0
        assert "No checks configured" in result.output


class TestGateThreshold:
    def test_threshold_pass(self, runner: CliRunner, sample_report: Path) -> None:
        result = runner.invoke(cli, ["gate", "--report", str(sample_report), "--threshold", "3.5"])
        assert result.exit_code == 0
        assert "PASS" in result.output

    def test_threshold_fail(self, runner: CliRunner, sample_report: Path) -> None:
        result = runner.invoke(cli, ["gate", "--report", str(sample_report), "--threshold", "4.2"])
        assert result.exit_code == 1
        assert "FAIL" in result.output


class TestGateDimensionThreshold:
    def test_dimension_threshold_pass(self, runner: CliRunner, sample_report: Path) -> None:
        result = runner.invoke(cli, [
            "gate", "--report", str(sample_report),
            "--dimension-threshold", "visual_quality:3.5",
        ])
        assert result.exit_code == 0
        assert "PASS" in result.output

    def test_dimension_threshold_fail(self, runner: CliRunner, sample_report: Path) -> None:
        result = runner.invoke(cli, [
            "gate", "--report", str(sample_report),
            "--dimension-threshold", "prompt_adherence:4.5",
        ])
        assert result.exit_code == 1
        assert "FAIL" in result.output

    def test_invalid_dimension_threshold_format(self, runner: CliRunner, sample_report: Path) -> None:
        result = runner.invoke(cli, [
            "gate", "--report", str(sample_report),
            "--dimension-threshold", "invalid_format",
        ])
        assert result.exit_code == 2
        assert "Invalid dimension threshold" in result.output


class TestGateConfidence:
    def test_min_confidence_pass(self, runner: CliRunner, tmp_path: Path) -> None:
        """avg_confidence above threshold should pass."""
        data = {
            "name": "test-bench",
            "version": "1.0",
            "summary": {
                "flux-schnell": {
                    "overall_score": 4.0,
                    "avg_confidence": 0.85,
                    "dimension_averages": {"visual_quality": 4.0},
                },
            },
        }
        path = tmp_path / "conf_report.json"
        path.write_text(json.dumps(data))
        result = runner.invoke(cli, [
            "gate", "--report", str(path), "--min-confidence", "0.7",
        ])
        assert result.exit_code == 0
        assert "PASS" in result.output

    def test_min_confidence_fail(self, runner: CliRunner, tmp_path: Path) -> None:
        """avg_confidence below threshold should fail."""
        data = {
            "name": "test-bench",
            "version": "1.0",
            "summary": {
                "flux-schnell": {
                    "overall_score": 4.0,
                    "avg_confidence": 0.45,
                    "dimension_averages": {"visual_quality": 4.0},
                },
            },
        }
        path = tmp_path / "conf_report.json"
        path.write_text(json.dumps(data))
        result = runner.invoke(cli, [
            "gate", "--report", str(path), "--min-confidence", "0.7",
        ])
        assert result.exit_code == 1
        assert "FAIL" in result.output

    def test_min_confidence_default_when_missing(self, runner: CliRunner, sample_report: Path) -> None:
        """Report without avg_confidence should default to 1.0 and pass."""
        result = runner.invoke(cli, [
            "gate", "--report", str(sample_report), "--min-confidence", "0.5",
        ])
        assert result.exit_code == 0
        assert "PASS" in result.output

    def test_help_shows_min_confidence(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["gate", "--help"])
        assert "--min-confidence" in result.output


class TestGateRegression:
    def test_regression_pass(self, runner: CliRunner, sample_report: Path, baseline_report: Path) -> None:
        """Small drops within threshold should pass."""
        result = runner.invoke(cli, [
            "gate", "--report", str(sample_report),
            "--baseline", str(baseline_report),
            "--regression-threshold", "0.5",
        ])
        assert result.exit_code == 0
        assert "PASS" in result.output

    def test_regression_fail(self, runner: CliRunner, sample_report: Path, baseline_report: Path) -> None:
        """Drops exceeding threshold should fail."""
        result = runner.invoke(cli, [
            "gate", "--report", str(sample_report),
            "--baseline", str(baseline_report),
            "--regression-threshold", "0.1",
        ])
        assert result.exit_code == 1
        assert "FAIL" in result.output


class TestGateJsonOutput:
    def test_json_output_to_file(self, runner: CliRunner, sample_report: Path, tmp_path: Path) -> None:
        """--json-output writes structured JSON to a file."""
        out_file = tmp_path / "gate-result.json"
        result = runner.invoke(cli, [
            "gate", "--report", str(sample_report),
            "--threshold", "3.5",
            "--json-output", str(out_file),
        ])
        assert result.exit_code == 0
        data = json.loads(out_file.read_text())
        assert data["status"] == "pass"
        assert "checks" in data
        assert data["summary"]["total_checks"] > 0
        assert data["summary"]["passed"] > 0

    def test_json_output_to_stdout(self, runner: CliRunner, sample_report: Path) -> None:
        """--json-output - writes JSON to stdout."""
        result = runner.invoke(cli, [
            "gate", "--report", str(sample_report),
            "--threshold", "3.5",
            "--json-output", "-",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "pass"

    def test_json_output_fail(self, runner: CliRunner, sample_report: Path, tmp_path: Path) -> None:
        """Failed gate writes status=fail in JSON."""
        out_file = tmp_path / "gate-result.json"
        result = runner.invoke(cli, [
            "gate", "--report", str(sample_report),
            "--threshold", "4.8",
            "--json-output", str(out_file),
        ])
        assert result.exit_code == 1
        data = json.loads(out_file.read_text())
        assert data["status"] == "fail"
        assert data["summary"]["failed"] > 0

    def test_json_output_error(self, runner: CliRunner, tmp_path: Path) -> None:
        """Errors write status=error in JSON."""
        out_file = tmp_path / "gate-result.json"
        result = runner.invoke(cli, [
            "gate", "--report", "/nonexistent/report.json",
            "--threshold", "3.0",
            "--json-output", str(out_file),
        ])
        assert result.exit_code == 2
        data = json.loads(out_file.read_text())
        assert data["status"] == "error"
        assert "error" in data

    def test_json_output_no_checks(self, runner: CliRunner, sample_report: Path, tmp_path: Path) -> None:
        """No checks configured writes pass with warning in JSON."""
        out_file = tmp_path / "gate-result.json"
        result = runner.invoke(cli, [
            "gate", "--report", str(sample_report),
            "--json-output", str(out_file),
        ])
        assert result.exit_code == 0
        data = json.loads(out_file.read_text())
        assert data["status"] == "pass"
        assert "warning" in data

    def test_json_output_suppresses_terminal(self, runner: CliRunner, sample_report: Path) -> None:
        """--json-output - should not include Rich table output."""
        result = runner.invoke(cli, [
            "gate", "--report", str(sample_report),
            "--threshold", "3.5",
            "--json-output", "-",
        ])
        assert result.exit_code == 0
        # Should be valid JSON (no Rich markup mixed in)
        data = json.loads(result.output)
        assert isinstance(data, dict)
        # Should NOT contain Rich table markers
        assert "Gate Results" not in result.output


@pytest.fixture()
def report_with_items(tmp_path: Path) -> Path:
    """Create a report with per-item results for regression testing."""
    data = {
        "name": "current-bench",
        "version": "1.0",
        "summary": {
            "flux-schnell": {
                "overall_score": 3.8,
                "dimension_averages": {"visual_quality": 4.0, "prompt_adherence": 3.6},
            },
        },
        "items": [
            {
                "item_id": "cat-window",
                "results": {
                    "flux-schnell": {
                        "status": "success",
                        "scores": [
                            {"dimension": "visual_quality", "score": 4.0},
                            {"dimension": "prompt_adherence", "score": 3.5},
                        ],
                    },
                },
            },
            {
                "item_id": "dog-park",
                "results": {
                    "flux-schnell": {
                        "status": "success",
                        "scores": [
                            {"dimension": "visual_quality", "score": 4.0},
                            {"dimension": "prompt_adherence", "score": 3.7},
                        ],
                    },
                },
            },
        ],
    }
    path = tmp_path / "current_items.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture()
def baseline_with_items(tmp_path: Path) -> Path:
    """Create a baseline report with per-item results."""
    data = {
        "name": "baseline-bench",
        "version": "1.0",
        "summary": {
            "flux-schnell": {
                "overall_score": 4.2,
                "dimension_averages": {"visual_quality": 4.5, "prompt_adherence": 3.9},
            },
        },
        "items": [
            {
                "item_id": "cat-window",
                "results": {
                    "flux-schnell": {
                        "status": "success",
                        "scores": [
                            {"dimension": "visual_quality", "score": 4.5},
                            {"dimension": "prompt_adherence", "score": 4.0},
                        ],
                    },
                },
            },
            {
                "item_id": "dog-park",
                "results": {
                    "flux-schnell": {
                        "status": "success",
                        "scores": [
                            {"dimension": "visual_quality", "score": 4.5},
                            {"dimension": "prompt_adherence", "score": 3.8},
                        ],
                    },
                },
            },
        ],
    }
    path = tmp_path / "baseline_items.json"
    path.write_text(json.dumps(data))
    return path


class TestGateItemRegression:
    def test_item_regression_pass(
        self, runner: CliRunner, report_with_items: Path, baseline_with_items: Path,
    ) -> None:
        """Per-item drops within threshold should pass."""
        result = runner.invoke(cli, [
            "gate", "--report", str(report_with_items),
            "--baseline", str(baseline_with_items),
            "--regression-threshold", "0.6",
        ])
        assert result.exit_code == 0
        assert "PASS" in result.output

    def test_item_regression_fail(
        self, runner: CliRunner, report_with_items: Path, baseline_with_items: Path,
    ) -> None:
        """Per-item drops exceeding threshold should fail."""
        result = runner.invoke(cli, [
            "gate", "--report", str(report_with_items),
            "--baseline", str(baseline_with_items),
            "--regression-threshold", "0.1",
        ])
        assert result.exit_code == 1
        assert "FAIL" in result.output

    def test_item_regression_json_output(
        self, runner: CliRunner, report_with_items: Path, baseline_with_items: Path, tmp_path: Path,
    ) -> None:
        """Per-item regressions appear in JSON output."""
        out_file = tmp_path / "gate-items.json"
        result = runner.invoke(cli, [
            "gate", "--report", str(report_with_items),
            "--baseline", str(baseline_with_items),
            "--regression-threshold", "0.1",
            "--json-output", str(out_file),
        ])
        assert result.exit_code == 1
        data = json.loads(out_file.read_text())
        assert data["status"] == "fail"
        # Should have item_regression type checks
        item_checks = [c for c in data["checks"] if c["type"] == "item_regression"]
        assert len(item_checks) > 0
        assert all(c["passed"] is False for c in item_checks)
        assert "item_regressions" in data

    def test_item_regression_skips_failed_items(
        self, runner: CliRunner, baseline_with_items: Path, tmp_path: Path,
    ) -> None:
        """Items with status != success should be skipped in regression."""
        current_data = {
            "name": "current",
            "version": "1.0",
            "summary": {
                "flux-schnell": {
                    "overall_score": 3.0,
                    "dimension_averages": {"visual_quality": 3.0},
                },
            },
            "items": [
                {
                    "item_id": "cat-window",
                    "results": {
                        "flux-schnell": {
                            "status": "failed",
                            "scores": [],
                        },
                    },
                },
            ],
        }
        current_path = tmp_path / "current_failed.json"
        current_path.write_text(json.dumps(current_data))
        out_file = tmp_path / "gate-skip.json"
        result = runner.invoke(cli, [
            "gate", "--report", str(current_path),
            "--baseline", str(baseline_with_items),
            "--regression-threshold", "0.1",
            "--json-output", str(out_file),
        ])
        data = json.loads(out_file.read_text())
        # Failed items should not generate item_regression checks
        item_checks = [c for c in data["checks"] if c["type"] == "item_regression"]
        assert len(item_checks) == 0

    def test_item_regression_new_items_skipped(
        self, runner: CliRunner, baseline_with_items: Path, tmp_path: Path,
    ) -> None:
        """Items not in baseline should be skipped (no regression to detect)."""
        current_data = {
            "name": "current",
            "version": "1.0",
            "summary": {
                "flux-schnell": {
                    "overall_score": 4.0,
                    "dimension_averages": {"visual_quality": 4.0},
                },
            },
            "items": [
                {
                    "item_id": "new-item-not-in-baseline",
                    "results": {
                        "flux-schnell": {
                            "status": "success",
                            "scores": [
                                {"dimension": "visual_quality", "score": 2.0},
                            ],
                        },
                    },
                },
            ],
        }
        current_path = tmp_path / "current_new.json"
        current_path.write_text(json.dumps(current_data))
        out_file = tmp_path / "gate-new.json"
        result = runner.invoke(cli, [
            "gate", "--report", str(current_path),
            "--baseline", str(baseline_with_items),
            "--regression-threshold", "0.1",
            "--json-output", str(out_file),
        ])
        data = json.loads(out_file.read_text())
        item_checks = [c for c in data["checks"] if c["type"] == "item_regression"]
        assert len(item_checks) == 0
