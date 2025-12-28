"""Coverage parsing and threshold handling for mala validation.

This module provides:
- CoverageResult: result of parsing a coverage report
- parse_coverage_xml: parse coverage.xml and return CoverageResult
- check_coverage_threshold: compare coverage against minimum threshold
- get_baseline_coverage: extract coverage percentage from existing baseline file
- is_baseline_stale: check if baseline file is older than last commit or repo is dirty
"""

from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum
from pathlib import Path  # noqa: TC003 - used at runtime for .exists() and ET.parse()


class CoverageStatus(Enum):
    """Status of coverage parsing/validation."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    PARSED = "parsed"  # Successfully parsed, but threshold not yet checked


@dataclass(frozen=True)
class CoverageResult:
    """Result of parsing and validating a coverage report.

    Attributes:
        percent: Coverage percentage (0.0-100.0), or None if parsing failed.
        passed: Whether coverage meets the threshold (False until threshold checked).
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
        if self.status == CoverageStatus.PARSED:
            return f"coverage {self.percent:.1f}% (threshold not checked)"
        return f"coverage {self.percent:.1f}% failed"


def parse_coverage_xml(report_path: Path) -> CoverageResult:
    """Parse a coverage.xml file and extract coverage metrics.

    Note: This function returns status=PARSED with passed=False. Callers must
    use check_coverage_threshold() to determine if coverage meets requirements.

    Args:
        report_path: Path to the coverage.xml file.

    Returns:
        CoverageResult with parsed metrics (status=PARSED) or error information.
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
    except OSError as e:
        return CoverageResult(
            percent=None,
            passed=False,
            status=CoverageStatus.ERROR,
            report_path=report_path,
            failure_reason=f"Cannot read coverage report: {e}",
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
        passed=False,  # Must call check_coverage_threshold to set passed=True
        status=CoverageStatus.PARSED,
        report_path=report_path,
        line_rate=line_rate,
        branch_rate=branch_rate,
    )


def check_coverage_threshold(
    result: CoverageResult,
    min_percent: float | None,
) -> CoverageResult:
    """Check if coverage meets the minimum threshold.

    Args:
        result: A CoverageResult from parse_coverage_xml.
        min_percent: Minimum required coverage percentage (0.0-100.0), or None
            to skip threshold checking (always passes).

    Returns:
        A new CoverageResult with passed/status updated based on threshold.
    """
    # If parsing failed, return as-is
    if result.status == CoverageStatus.ERROR or result.percent is None:
        return result

    # If no threshold specified, consider it passed
    if min_percent is None:
        return CoverageResult(
            percent=result.percent,
            passed=True,
            status=CoverageStatus.PASSED,
            report_path=result.report_path,
            failure_reason=None,
            line_rate=result.line_rate,
            branch_rate=result.branch_rate,
        )

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
    min_percent: float | None,
) -> CoverageResult:
    """Parse coverage XML and check against threshold in one call.

    This is a convenience function that combines parse_coverage_xml
    and check_coverage_threshold.

    Args:
        report_path: Path to the coverage.xml file.
        min_percent: Minimum required coverage percentage (0.0-100.0), or None
            to skip threshold checking (always passes).

    Returns:
        CoverageResult with parsing and threshold check results.
    """
    result = parse_coverage_xml(report_path)
    return check_coverage_threshold(result, min_percent)


def get_baseline_coverage(report_path: Path) -> float | None:
    """Extract coverage percentage from a baseline coverage report.

    This function is used to read a previously saved coverage baseline file
    to get the minimum coverage threshold for "no decrease" checking.

    Args:
        report_path: Path to the coverage.xml baseline file.

    Returns:
        Coverage percentage (0.0-100.0) if file exists and is valid, None if
        the file is missing.

    Raises:
        ValueError: If the file exists but cannot be parsed (malformed XML,
            missing required attributes, etc.).
    """
    if not report_path.exists():
        return None

    result = parse_coverage_xml(report_path)

    if result.status == CoverageStatus.ERROR:
        raise ValueError(result.failure_reason)

    return result.percent


def is_baseline_stale(report_path: Path, repo_path: Path) -> bool:
    """Check if the coverage baseline file is stale and needs refresh.

    A baseline is considered stale if:
    - The baseline file doesn't exist
    - The repo has uncommitted changes (dirty working tree)
    - The baseline file's mtime is older than the last commit time
    - Git commands fail (non-git repo or git errors)

    Args:
        report_path: Path to the coverage.xml baseline file.
        repo_path: Path to the git repository root.

    Returns:
        True if baseline is stale or doesn't exist, False if baseline is fresh.
    """
    # Missing baseline is considered stale
    if not report_path.exists():
        return True

    try:
        # Check for dirty working tree
        dirty_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        if dirty_result.stdout.strip():
            # Has uncommitted changes
            return True

        # Get last commit timestamp (Unix epoch seconds)
        commit_time_result = subprocess.run(
            ["git", "log", "-1", "--format=%ct"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        commit_time_str = commit_time_result.stdout.strip()
        if not commit_time_str:
            # No commits in repo
            return True

        commit_time = int(commit_time_str)

        # Get baseline file mtime
        baseline_mtime = report_path.stat().st_mtime

        # Stale if baseline is older than last commit
        return baseline_mtime < commit_time

    except (subprocess.CalledProcessError, ValueError, OSError):
        # Git command failed, path error, or parse error - treat as stale
        return True
