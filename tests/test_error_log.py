"""Tests for the error log writer."""

from __future__ import annotations

import tempfile
from pathlib import Path

from evalytic.bench.types import BenchReport, RunError
from evalytic.report.error_log import write_error_log


def _make_error(
    phase: str = "generation",
    model: str = "flux-schnell",
    item_id: str = "item-001",
    dimension: str = "",
    message: str = "something failed",
    level: str = "error",
) -> RunError:
    return RunError(
        timestamp="2026-02-27T14:30:01+00:00",
        level=level,
        phase=phase,
        model=model,
        item_id=item_id,
        dimension=dimension,
        message=message,
    )


def _make_report(errors: list[RunError] | None = None) -> BenchReport:
    return BenchReport(
        name="bench-test123",
        models=["flux-schnell", "flux-pro"],
        judge="gemini-2.5-flash",
        pipeline="text2img",
        created_at="2026-02-27T14:30:00+00:00",
        errors=errors or [],
    )


class TestWriteErrorLog:
    def test_write_error_log_with_errors(self) -> None:
        errors = [
            _make_error(phase="generation", message="content_policy_violation"),
            _make_error(phase="scoring", model="flux-pro", item_id="item-002",
                        dimension="visual_quality", message="Judge timeout"),
        ]
        report = _make_report(errors)

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "errors.log")
            write_error_log(report, path)

            content = Path(path).read_text()
            assert "content_policy_violation" in content
            assert "Judge timeout" in content
            assert "flux-schnell" in content
            assert "flux-pro" in content

    def test_write_error_log_no_errors(self) -> None:
        report = _make_report([])

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "errors.log")
            write_error_log(report, path)

            assert not Path(path).exists()

    def test_error_log_header_format(self) -> None:
        errors = [_make_error()]
        report = _make_report(errors)

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "errors.log")
            write_error_log(report, path)

            content = Path(path).read_text()
            assert content.startswith("Evalytic Bench Error Log\n")
            assert "Run: bench-test123" in content
            assert "Models: flux-schnell, flux-pro" in content
            assert "Errors: 1, Warnings: 0" in content

    def test_error_log_entry_format(self) -> None:
        err = _make_error(
            phase="generation",
            model="flux-schnell",
            item_id="item-001",
            message="content_policy_violation",
        )
        report = _make_report([err])

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "errors.log")
            write_error_log(report, path)

            content = Path(path).read_text()
            assert "[14:30:01] ERROR generation flux-schnell/item-001" in content
            assert "  content_policy_violation" in content

    def test_error_log_warning_count(self) -> None:
        errors = [
            _make_error(level="error", message="fail1"),
            _make_error(level="warning", message="warn1"),
            _make_error(level="warning", message="warn2"),
        ]
        report = _make_report(errors)

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "errors.log")
            write_error_log(report, path)

            content = Path(path).read_text()
            assert "Errors: 1, Warnings: 2" in content
