"""Unit tests for src/validation/coverage.py - coverage parsing and thresholds.

Tests coverage XML parsing, threshold checking, and error handling
with fixture XML files for pass/fail/invalid cases.
"""

import os
import subprocess
from pathlib import Path
from unittest.mock import patch

from src.validation.coverage import (
    CoverageResult,
    CoverageStatus,
    check_coverage_threshold,
    get_baseline_coverage,
    is_baseline_stale,
    parse_and_check_coverage,
    parse_coverage_xml,
)


# Fixture XML content for different test cases
VALID_COVERAGE_XML_90_PERCENT = """\
<?xml version="1.0" ?>
<coverage version="7.4.0" timestamp="1234567890" lines-valid="100" lines-covered="90" line-rate="0.9" branches-covered="45" branches-valid="50" branch-rate="0.9" complexity="0">
    <packages>
        <package name="src" line-rate="0.9" branch-rate="0.9" complexity="0">
            <classes>
                <class name="app.py" filename="src/app.py" line-rate="0.9" branch-rate="0.9" complexity="0">
                    <lines>
                        <line number="1" hits="1"/>
                        <line number="2" hits="1"/>
                    </lines>
                </class>
            </classes>
        </package>
    </packages>
</coverage>
"""

VALID_COVERAGE_XML_50_PERCENT = """\
<?xml version="1.0" ?>
<coverage version="7.4.0" timestamp="1234567890" lines-valid="100" lines-covered="50" line-rate="0.5" branches-covered="25" branches-valid="50" branch-rate="0.5" complexity="0">
    <packages>
        <package name="src" line-rate="0.5" branch-rate="0.5" complexity="0">
            <classes/>
        </package>
    </packages>
</coverage>
"""

VALID_COVERAGE_XML_100_PERCENT = """\
<?xml version="1.0" ?>
<coverage version="7.4.0" timestamp="1234567890" lines-valid="100" lines-covered="100" line-rate="1.0" branch-rate="1.0" complexity="0">
    <packages/>
</coverage>
"""

VALID_COVERAGE_XML_NO_BRANCH = """\
<?xml version="1.0" ?>
<coverage version="7.4.0" timestamp="1234567890" lines-valid="100" lines-covered="75" line-rate="0.75" complexity="0">
    <packages/>
</coverage>
"""

INVALID_XML_SYNTAX = """\
<?xml version="1.0" ?>
<coverage version="7.4.0" line-rate="0.9"
    <packages>  <!-- Missing closing > -->
"""

INVALID_XML_WRONG_ROOT = """\
<?xml version="1.0" ?>
<report version="7.4.0" line-rate="0.9">
    <packages/>
</report>
"""

INVALID_XML_MISSING_LINE_RATE = """\
<?xml version="1.0" ?>
<coverage version="7.4.0" timestamp="1234567890">
    <packages/>
</coverage>
"""

INVALID_XML_BAD_LINE_RATE = """\
<?xml version="1.0" ?>
<coverage version="7.4.0" line-rate="not-a-number">
    <packages/>
</coverage>
"""


class TestCoverageResult:
    """Test CoverageResult dataclass."""

    def test_passed_result(self) -> None:
        result = CoverageResult(
            percent=90.0,
            passed=True,
            status=CoverageStatus.PASSED,
            report_path=Path("coverage.xml"),
            line_rate=0.9,
            branch_rate=0.9,
        )
        assert result.percent == 90.0
        assert result.passed is True
        assert result.status == CoverageStatus.PASSED
        assert result.failure_reason is None
        assert result.line_rate == 0.9
        assert result.branch_rate == 0.9

    def test_failed_result(self) -> None:
        result = CoverageResult(
            percent=50.0,
            passed=False,
            status=CoverageStatus.FAILED,
            report_path=Path("coverage.xml"),
            failure_reason="Coverage 50.0% is below threshold 85.0%",
        )
        assert result.percent == 50.0
        assert result.passed is False
        assert result.status == CoverageStatus.FAILED
        assert "below threshold" in result.failure_reason  # type: ignore[operator]

    def test_error_result(self) -> None:
        result = CoverageResult(
            percent=None,
            passed=False,
            status=CoverageStatus.ERROR,
            report_path=Path("missing.xml"),
            failure_reason="Coverage report not found",
        )
        assert result.percent is None
        assert result.passed is False
        assert result.status == CoverageStatus.ERROR

    def test_parsed_result(self) -> None:
        result = CoverageResult(
            percent=90.0,
            passed=False,
            status=CoverageStatus.PARSED,
            report_path=Path("coverage.xml"),
            line_rate=0.9,
        )
        assert result.percent == 90.0
        assert result.passed is False
        assert result.status == CoverageStatus.PARSED

    def test_short_summary_passed(self) -> None:
        result = CoverageResult(
            percent=90.0,
            passed=True,
            status=CoverageStatus.PASSED,
            report_path=Path("coverage.xml"),
        )
        assert result.short_summary() == "coverage 90.0% passed"

    def test_short_summary_failed_with_reason(self) -> None:
        result = CoverageResult(
            percent=50.0,
            passed=False,
            status=CoverageStatus.FAILED,
            report_path=Path("coverage.xml"),
            failure_reason="Coverage 50.0% is below threshold 85.0%",
        )
        assert result.short_summary() == "Coverage 50.0% is below threshold 85.0%"

    def test_short_summary_failed_without_reason(self) -> None:
        result = CoverageResult(
            percent=50.0,
            passed=False,
            status=CoverageStatus.FAILED,
            report_path=Path("coverage.xml"),
        )
        assert result.short_summary() == "coverage 50.0% failed"

    def test_short_summary_parsed(self) -> None:
        result = CoverageResult(
            percent=75.0,
            passed=False,
            status=CoverageStatus.PARSED,
            report_path=Path("coverage.xml"),
        )
        assert result.short_summary() == "coverage 75.0% (threshold not checked)"


class TestParseCoverageXml:
    """Test parse_coverage_xml function."""

    def test_parse_valid_90_percent(self, tmp_path: Path) -> None:
        report = tmp_path / "coverage.xml"
        report.write_text(VALID_COVERAGE_XML_90_PERCENT)

        result = parse_coverage_xml(report)

        assert result.percent == 90.0
        assert result.passed is False  # Not checked against threshold yet
        assert result.status == CoverageStatus.PARSED
        assert result.report_path == report
        assert result.failure_reason is None
        assert result.line_rate == 0.9
        assert result.branch_rate == 0.9

    def test_parse_valid_50_percent(self, tmp_path: Path) -> None:
        report = tmp_path / "coverage.xml"
        report.write_text(VALID_COVERAGE_XML_50_PERCENT)

        result = parse_coverage_xml(report)

        assert result.percent == 50.0
        assert result.passed is False  # Not checked against threshold yet
        assert result.status == CoverageStatus.PARSED
        assert result.line_rate == 0.5
        assert result.branch_rate == 0.5

    def test_parse_valid_100_percent(self, tmp_path: Path) -> None:
        report = tmp_path / "coverage.xml"
        report.write_text(VALID_COVERAGE_XML_100_PERCENT)

        result = parse_coverage_xml(report)

        assert result.percent == 100.0
        assert result.passed is False  # Not checked against threshold yet
        assert result.status == CoverageStatus.PARSED
        assert result.line_rate == 1.0
        assert result.branch_rate == 1.0

    def test_parse_valid_no_branch_rate(self, tmp_path: Path) -> None:
        report = tmp_path / "coverage.xml"
        report.write_text(VALID_COVERAGE_XML_NO_BRANCH)

        result = parse_coverage_xml(report)

        assert result.percent == 75.0
        assert result.passed is False  # Not checked against threshold yet
        assert result.status == CoverageStatus.PARSED
        assert result.line_rate == 0.75
        assert result.branch_rate is None

    def test_parse_missing_file(self, tmp_path: Path) -> None:
        report = tmp_path / "nonexistent.xml"

        result = parse_coverage_xml(report)

        assert result.percent is None
        assert result.passed is False
        assert result.status == CoverageStatus.ERROR
        assert result.report_path == report
        assert "not found" in result.failure_reason  # type: ignore[operator]

    def test_parse_directory_path(self, tmp_path: Path) -> None:
        """Test that parsing a directory returns an error (OSError case)."""
        result = parse_coverage_xml(tmp_path)

        assert result.percent is None
        assert result.passed is False
        assert result.status == CoverageStatus.ERROR
        assert result.report_path == tmp_path
        assert "Cannot read coverage report" in result.failure_reason  # type: ignore[operator]

    def test_parse_invalid_xml_syntax(self, tmp_path: Path) -> None:
        report = tmp_path / "coverage.xml"
        report.write_text(INVALID_XML_SYNTAX)

        result = parse_coverage_xml(report)

        assert result.percent is None
        assert result.passed is False
        assert result.status == CoverageStatus.ERROR
        assert "Invalid coverage XML" in result.failure_reason  # type: ignore[operator]

    def test_parse_wrong_root_element(self, tmp_path: Path) -> None:
        report = tmp_path / "coverage.xml"
        report.write_text(INVALID_XML_WRONG_ROOT)

        result = parse_coverage_xml(report)

        assert result.percent is None
        assert result.passed is False
        assert result.status == CoverageStatus.ERROR
        assert "expected <coverage> root" in result.failure_reason  # type: ignore[operator]

    def test_parse_missing_line_rate(self, tmp_path: Path) -> None:
        report = tmp_path / "coverage.xml"
        report.write_text(INVALID_XML_MISSING_LINE_RATE)

        result = parse_coverage_xml(report)

        assert result.percent is None
        assert result.passed is False
        assert result.status == CoverageStatus.ERROR
        assert "missing line-rate" in result.failure_reason  # type: ignore[operator]

    def test_parse_bad_line_rate_value(self, tmp_path: Path) -> None:
        report = tmp_path / "coverage.xml"
        report.write_text(INVALID_XML_BAD_LINE_RATE)

        result = parse_coverage_xml(report)

        assert result.percent is None
        assert result.passed is False
        assert result.status == CoverageStatus.ERROR
        assert "not a number" in result.failure_reason  # type: ignore[operator]


class TestCheckCoverageThreshold:
    """Test check_coverage_threshold function."""

    def test_check_above_threshold(self) -> None:
        result = CoverageResult(
            percent=90.0,
            passed=False,
            status=CoverageStatus.PARSED,
            report_path=Path("coverage.xml"),
            line_rate=0.9,
        )

        checked = check_coverage_threshold(result, min_percent=85.0)

        assert checked.passed is True
        assert checked.status == CoverageStatus.PASSED
        assert checked.failure_reason is None

    def test_check_exactly_at_threshold(self) -> None:
        result = CoverageResult(
            percent=85.0,
            passed=False,
            status=CoverageStatus.PARSED,
            report_path=Path("coverage.xml"),
            line_rate=0.85,
        )

        checked = check_coverage_threshold(result, min_percent=85.0)

        assert checked.passed is True
        assert checked.status == CoverageStatus.PASSED

    def test_check_below_threshold(self) -> None:
        result = CoverageResult(
            percent=50.0,
            passed=False,
            status=CoverageStatus.PARSED,
            report_path=Path("coverage.xml"),
            line_rate=0.5,
        )

        checked = check_coverage_threshold(result, min_percent=85.0)

        assert checked.passed is False
        assert checked.status == CoverageStatus.FAILED
        assert checked.failure_reason is not None
        assert "50.0%" in checked.failure_reason
        assert "85.0%" in checked.failure_reason

    def test_check_preserves_error_status(self) -> None:
        result = CoverageResult(
            percent=None,
            passed=False,
            status=CoverageStatus.ERROR,
            report_path=Path("missing.xml"),
            failure_reason="Coverage report not found",
        )

        checked = check_coverage_threshold(result, min_percent=85.0)

        # Should return original result unchanged
        assert checked.status == CoverageStatus.ERROR
        assert checked.failure_reason == "Coverage report not found"

    def test_check_preserves_metrics(self) -> None:
        result = CoverageResult(
            percent=90.0,
            passed=False,
            status=CoverageStatus.PARSED,
            report_path=Path("coverage.xml"),
            line_rate=0.9,
            branch_rate=0.85,
        )

        checked = check_coverage_threshold(result, min_percent=85.0)

        assert checked.line_rate == 0.9
        assert checked.branch_rate == 0.85
        assert checked.report_path == Path("coverage.xml")

    def test_check_none_threshold_passes(self) -> None:
        """When min_percent is None, coverage check should always pass."""
        result = CoverageResult(
            percent=50.0,
            passed=False,
            status=CoverageStatus.PARSED,
            report_path=Path("coverage.xml"),
            line_rate=0.5,
            branch_rate=0.4,
        )

        checked = check_coverage_threshold(result, min_percent=None)

        assert checked.passed is True
        assert checked.status == CoverageStatus.PASSED
        assert checked.failure_reason is None
        assert checked.percent == 50.0
        assert checked.line_rate == 0.5
        assert checked.branch_rate == 0.4

    def test_check_none_threshold_preserves_error(self) -> None:
        """When min_percent is None but result has error, error is preserved."""
        result = CoverageResult(
            percent=None,
            passed=False,
            status=CoverageStatus.ERROR,
            report_path=Path("missing.xml"),
            failure_reason="Coverage report not found",
        )

        checked = check_coverage_threshold(result, min_percent=None)

        assert checked.status == CoverageStatus.ERROR
        assert checked.failure_reason == "Coverage report not found"


class TestParseAndCheckCoverage:
    """Test parse_and_check_coverage convenience function."""

    def test_parse_and_check_passes(self, tmp_path: Path) -> None:
        report = tmp_path / "coverage.xml"
        report.write_text(VALID_COVERAGE_XML_90_PERCENT)

        result = parse_and_check_coverage(report, min_percent=85.0)

        assert result.passed is True
        assert result.status == CoverageStatus.PASSED
        assert result.percent == 90.0

    def test_parse_and_check_fails_threshold(self, tmp_path: Path) -> None:
        report = tmp_path / "coverage.xml"
        report.write_text(VALID_COVERAGE_XML_50_PERCENT)

        result = parse_and_check_coverage(report, min_percent=85.0)

        assert result.passed is False
        assert result.status == CoverageStatus.FAILED
        assert result.percent == 50.0
        assert "below threshold" in result.failure_reason  # type: ignore[operator]

    def test_parse_and_check_missing_file(self, tmp_path: Path) -> None:
        report = tmp_path / "nonexistent.xml"

        result = parse_and_check_coverage(report, min_percent=85.0)

        assert result.passed is False
        assert result.status == CoverageStatus.ERROR
        assert "not found" in result.failure_reason  # type: ignore[operator]

    def test_parse_and_check_invalid_xml(self, tmp_path: Path) -> None:
        report = tmp_path / "coverage.xml"
        report.write_text(INVALID_XML_SYNTAX)

        result = parse_and_check_coverage(report, min_percent=85.0)

        assert result.passed is False
        assert result.status == CoverageStatus.ERROR

    def test_parse_and_check_none_threshold(self, tmp_path: Path) -> None:
        """When min_percent is None, parse_and_check_coverage always passes."""
        report = tmp_path / "coverage.xml"
        report.write_text(VALID_COVERAGE_XML_50_PERCENT)

        result = parse_and_check_coverage(report, min_percent=None)

        assert result.passed is True
        assert result.status == CoverageStatus.PASSED
        assert result.percent == 50.0


class TestCoverageStatus:
    """Test CoverageStatus enum."""

    def test_status_values(self) -> None:
        assert CoverageStatus.PASSED.value == "passed"
        assert CoverageStatus.FAILED.value == "failed"
        assert CoverageStatus.ERROR.value == "error"
        assert CoverageStatus.PARSED.value == "parsed"


class TestGetBaselineCoverage:
    """Test get_baseline_coverage function."""

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        """Missing file should return None, not raise."""
        report = tmp_path / "nonexistent.xml"

        result = get_baseline_coverage(report)

        assert result is None

    def test_valid_file_returns_percentage(self, tmp_path: Path) -> None:
        """Valid coverage XML should return the percentage."""
        report = tmp_path / "coverage.xml"
        report.write_text(VALID_COVERAGE_XML_90_PERCENT)

        result = get_baseline_coverage(report)

        assert result == 90.0

    def test_valid_file_100_percent(self, tmp_path: Path) -> None:
        """100% coverage should be returned correctly."""
        report = tmp_path / "coverage.xml"
        report.write_text(VALID_COVERAGE_XML_100_PERCENT)

        result = get_baseline_coverage(report)

        assert result == 100.0

    def test_malformed_xml_raises_valueerror(self, tmp_path: Path) -> None:
        """Malformed XML should raise ValueError, not return None."""
        report = tmp_path / "coverage.xml"
        report.write_text(INVALID_XML_SYNTAX)

        import pytest

        with pytest.raises(ValueError) as exc_info:
            get_baseline_coverage(report)

        assert "Invalid coverage XML" in str(exc_info.value)

    def test_wrong_root_element_raises_valueerror(self, tmp_path: Path) -> None:
        """Wrong root element should raise ValueError."""
        report = tmp_path / "coverage.xml"
        report.write_text(INVALID_XML_WRONG_ROOT)

        import pytest

        with pytest.raises(ValueError) as exc_info:
            get_baseline_coverage(report)

        assert "expected <coverage> root" in str(exc_info.value)

    def test_missing_line_rate_raises_valueerror(self, tmp_path: Path) -> None:
        """Missing line-rate attribute should raise ValueError."""
        report = tmp_path / "coverage.xml"
        report.write_text(INVALID_XML_MISSING_LINE_RATE)

        import pytest

        with pytest.raises(ValueError) as exc_info:
            get_baseline_coverage(report)

        assert "missing line-rate" in str(exc_info.value)

    def test_bad_line_rate_raises_valueerror(self, tmp_path: Path) -> None:
        """Non-numeric line-rate should raise ValueError."""
        report = tmp_path / "coverage.xml"
        report.write_text(INVALID_XML_BAD_LINE_RATE)

        import pytest

        with pytest.raises(ValueError) as exc_info:
            get_baseline_coverage(report)

        assert "not a number" in str(exc_info.value)


class TestIsBaselineStale:
    """Test is_baseline_stale function."""

    def test_missing_baseline_is_stale(self, tmp_path: Path) -> None:
        """Missing baseline file should be considered stale."""
        report = tmp_path / "nonexistent.xml"

        result = is_baseline_stale(report, tmp_path)

        assert result is True

    def test_dirty_repo_is_stale(self, tmp_path: Path) -> None:
        """Repo with uncommitted changes should be considered stale."""
        # Create baseline file
        report = tmp_path / "coverage.xml"
        report.write_text(VALID_COVERAGE_XML_90_PERCENT)

        # Mock git status to return dirty
        mock_result = subprocess.CompletedProcess(
            args=["git", "status", "--porcelain"],
            returncode=0,
            stdout="M src/app.py\n",
            stderr="",
        )

        with patch("src.validation.coverage.subprocess.run", return_value=mock_result):
            result = is_baseline_stale(report, tmp_path)

        assert result is True

    def test_stale_mtime_is_stale(self, tmp_path: Path) -> None:
        """Baseline older than last commit should be considered stale."""
        # Create baseline file
        report = tmp_path / "coverage.xml"
        report.write_text(VALID_COVERAGE_XML_90_PERCENT)

        # Set baseline mtime to old time (year 2000)
        old_time = 946684800
        os.utime(report, (old_time, old_time))

        # Mock git commands
        def mock_run(
            args: list[str], **_kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            if "status" in args:
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
            elif "log" in args:
                # Commit time is more recent than baseline mtime
                return subprocess.CompletedProcess(
                    args, 0, stdout="1700000000\n", stderr=""
                )
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        with patch("src.validation.coverage.subprocess.run", side_effect=mock_run):
            result = is_baseline_stale(report, tmp_path)

        assert result is True

    def test_fresh_baseline_not_stale(self, tmp_path: Path) -> None:
        """Fresh baseline after clean commit should not be stale."""
        # Create baseline file
        report = tmp_path / "coverage.xml"
        report.write_text(VALID_COVERAGE_XML_90_PERCENT)

        # Set baseline mtime to very recent (year 2099)
        future_time = 4102444800
        os.utime(report, (future_time, future_time))

        # Mock git commands
        def mock_run(
            args: list[str], **_kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            if "status" in args:
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
            elif "log" in args:
                # Commit time is older than baseline mtime
                return subprocess.CompletedProcess(
                    args, 0, stdout="1700000000\n", stderr=""
                )
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        with patch("src.validation.coverage.subprocess.run", side_effect=mock_run):
            result = is_baseline_stale(report, tmp_path)

        assert result is False

    def test_non_git_repo_is_stale(self, tmp_path: Path) -> None:
        """Non-git repo (git commands fail) should be considered stale."""
        # Create baseline file
        report = tmp_path / "coverage.xml"
        report.write_text(VALID_COVERAGE_XML_90_PERCENT)

        # Mock git to fail
        with patch(
            "src.validation.coverage.subprocess.run",
            side_effect=subprocess.CalledProcessError(128, "git"),
        ):
            result = is_baseline_stale(report, tmp_path)

        assert result is True

    def test_empty_repo_no_commits_is_stale(self, tmp_path: Path) -> None:
        """Repo with no commits should be considered stale."""
        # Create baseline file
        report = tmp_path / "coverage.xml"
        report.write_text(VALID_COVERAGE_XML_90_PERCENT)

        # Mock git commands - status clean but log returns empty
        def mock_run(
            args: list[str], **_kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            if "status" in args:
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
            elif "log" in args:
                # No commits - empty output
                return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        with patch("src.validation.coverage.subprocess.run", side_effect=mock_run):
            result = is_baseline_stale(report, tmp_path)

        assert result is True
