"""Coverage parsing and threshold handling for mala validation.

This module provides:
- CoverageResult: result of parsing a coverage report
- parse_coverage_xml: parse coverage.xml and return CoverageResult
- check_coverage_threshold: compare coverage against minimum threshold
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum
from pathlib import Path  # noqa: TC003 - used at runtime for .exists() and ET.parse()


class CoverageStatus(Enum):
    """Status of coverage parsing/validation."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


@dataclass(frozen=True)
class CoverageResult:
    """Result of parsing and validating a coverage report.

    Attributes:
        percent: Coverage percentage (0.0-100.0), or None if parsing failed.
        passed: Whether coverage meets the threshold.
        status: Status of the coverage check.
        report_path: Path to the coverage report file.
        failure_reason: Explanation for failure/error (None if passed).
        line_rate: Raw line rate from XML (0.0-1.0), or None if unavailable.
        branch_rate: Raw branch rate from XML (0.0-1.0), or None if unavailable.
    """

    percent: float | None
    passed: bool
    status: CoverageStatus
    report_path: Path | None
    failure_reason: str | None = None
    line_rate: float | None = None
    branch_rate: float | None = None

    def short_summary(self) -> str:
        """One-line summary for logs/prompts."""
        if self.passed:
            return f"coverage {self.percent:.1f}% passed"
        if self.failure_reason:
            return self.failure_reason
        return f"coverage {self.percent:.1f}% failed"


def parse_coverage_xml(report_path: Path) -> CoverageResult:
    """Parse a coverage.xml file and extract coverage metrics.

    Args:
        report_path: Path to the coverage.xml file.

    Returns:
        CoverageResult with parsed metrics or error information.
    """
    if not report_path.exists():
        return CoverageResult(
            percent=None,
            passed=False,
            status=CoverageStatus.ERROR,
            report_path=report_path,
            failure_reason=f"Coverage report not found: {report_path}",
        )

    try:
        tree = ET.parse(report_path)
    except ET.ParseError as e:
        return CoverageResult(
            percent=None,
            passed=False,
            status=CoverageStatus.ERROR,
            report_path=report_path,
            failure_reason=f"Invalid coverage XML: {e}",
        )

    root = tree.getroot()

    # Check for expected root element
    if root.tag != "coverage":
        return CoverageResult(
            percent=None,
            passed=False,
            status=CoverageStatus.ERROR,
            report_path=report_path,
            failure_reason=f"Invalid coverage XML: expected <coverage> root, got <{root.tag}>",
        )

    # Extract line-rate and branch-rate from coverage element
    line_rate_str = root.get("line-rate")
    branch_rate_str = root.get("branch-rate")

    if line_rate_str is None:
        return CoverageResult(
            percent=None,
            passed=False,
            status=CoverageStatus.ERROR,
            report_path=report_path,
            failure_reason="Invalid coverage XML: missing line-rate attribute",
        )

    try:
        line_rate = float(line_rate_str)
    except ValueError:
        return CoverageResult(
            percent=None,
            passed=False,
            status=CoverageStatus.ERROR,
            report_path=report_path,
            failure_reason=f"Invalid coverage XML: line-rate '{line_rate_str}' is not a number",
        )

    branch_rate: float | None = None
    if branch_rate_str is not None:
        try:
            branch_rate = float(branch_rate_str)
        except ValueError:
            pass  # Branch rate is optional, ignore parse errors

    # Convert line rate (0.0-1.0) to percentage (0.0-100.0)
    percent = line_rate * 100.0

    return CoverageResult(
        percent=percent,
        passed=True,  # Will be updated by check_coverage_threshold
        status=CoverageStatus.PASSED,
        report_path=report_path,
        line_rate=line_rate,
        branch_rate=branch_rate,
    )


def check_coverage_threshold(
    result: CoverageResult,
    min_percent: float,
) -> CoverageResult:
    """Check if coverage meets the minimum threshold.

    Args:
        result: A CoverageResult from parse_coverage_xml.
        min_percent: Minimum required coverage percentage (0.0-100.0).

    Returns:
        A new CoverageResult with passed/status updated based on threshold.
    """
    # If parsing already failed, return as-is
    if result.status == CoverageStatus.ERROR or result.percent is None:
        return result

    passed = result.percent >= min_percent

    if passed:
        return CoverageResult(
            percent=result.percent,
            passed=True,
            status=CoverageStatus.PASSED,
            report_path=result.report_path,
            failure_reason=None,
            line_rate=result.line_rate,
            branch_rate=result.branch_rate,
        )

    return CoverageResult(
        percent=result.percent,
        passed=False,
        status=CoverageStatus.FAILED,
        report_path=result.report_path,
        failure_reason=f"Coverage {result.percent:.1f}% is below threshold {min_percent:.1f}%",
        line_rate=result.line_rate,
        branch_rate=result.branch_rate,
    )


def parse_and_check_coverage(
    report_path: Path,
    min_percent: float,
) -> CoverageResult:
    """Parse coverage XML and check against threshold in one call.

    This is a convenience function that combines parse_coverage_xml
    and check_coverage_threshold.

    Args:
        report_path: Path to the coverage.xml file.
        min_percent: Minimum required coverage percentage (0.0-100.0).

    Returns:
        CoverageResult with parsing and threshold check results.
    """
    result = parse_coverage_xml(report_path)
    return check_coverage_threshold(result, min_percent)
