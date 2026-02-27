"""Human-readable error log writer for bench reports."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..bench.types import BenchReport


def write_error_log(report: BenchReport, path: str) -> None:
    """Write a human-readable error log. Does nothing if no errors."""
    if not report.errors:
        return

    error_count = sum(1 for e in report.errors if e.level == "error")
    warning_count = sum(1 for e in report.errors if e.level == "warning")

    lines: list[str] = [
        "Evalytic Bench Error Log",
        "========================",
        f"Run: {report.name}",
        f"Date: {report.created_at}",
        f"Models: {', '.join(report.models)}",
        f"Errors: {error_count}, Warnings: {warning_count}",
        "",
    ]

    for err in report.errors:
        # Extract time portion from ISO timestamp for compact display
        ts_short = err.timestamp.split("T")[1][:8] if "T" in err.timestamp else err.timestamp
        level = err.level.upper()

        # Build location string: phase model/item_id/dimension
        location = f"{err.phase} {err.model}"
        if err.item_id:
            location += f"/{err.item_id}"
        if err.dimension:
            location += f"/{err.dimension}"

        lines.append(f"[{ts_short}] {level} {location}")
        lines.append(f"  {err.message}")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
