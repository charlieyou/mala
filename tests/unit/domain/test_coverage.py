"""Unit tests for src/validation/coverage.py - coverage parsing and thresholds.

Tests coverage XML parsing, threshold checking, and error handling
with fixture XML files for pass/fail/invalid cases.
"""

import os
from pathlib import Path
from collections.abc import Callable
from unittest.mock import Mock

from src.infra.tools.command_runner import CommandResult
from src.domain.validation.config import YamlCoverageConfig
from tests.fakes.lock_manager import FakeLockManager
from src.domain.validation.coverage import (
    BaselineCoverageService,
    CoverageResult,
    CoverageStatus,
    check_coverage_from_config,
    check_coverage_threshold,
    get_baseline_coverage,
    is_baseline_stale,
    parse_and_check_coverage,
    parse_coverage_xml,
)


def make_mock_runner(
    run_fn: Callable[..., CommandResult],
) -> Mock:
    """Create a mock CommandRunnerPort with a custom run function."""
    mock = Mock()
    mock.run = run_fn
    return mock


def make_mock_env_config() -> Mock:
    """Create a mock EnvConfigPort."""
    mock = Mock()
    mock.scripts_dir = Path("/mock/scripts")
    mock.cache_dir = Path("/mock/cache")
    mock.lock_dir = Path("/tmp/mock-locks")
    mock.find_cerberus_bin_path.return_value = None
    return mock


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
    """Test CoverageResult dataclass methods."""

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

    def test_check_floating_point_precision_edge_case(self) -> None:
        """Test that floating-point precision doesn't cause false failures.

        This tests the edge case where coverage like 88.79999999999999 should
        pass against threshold 88.8 since they display as the same value.
        """
        # Simulate a floating-point precision issue where the actual value
        # is infinitesimally below the threshold due to float representation
        result = CoverageResult(
            percent=88.8 - 1e-10,  # Just below 88.8 due to float precision
            passed=False,
            status=CoverageStatus.PARSED,
            report_path=Path("coverage.xml"),
            line_rate=0.888 - 1e-12,
        )

        checked = check_coverage_threshold(result, min_percent=88.8)

        # Should pass because the difference is within epsilon tolerance
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

        # Missing file doesn't need command runner, but function requires it
        mock_runner = make_mock_runner(lambda *args, **kwargs: None)
        result = is_baseline_stale(report, tmp_path, command_runner=mock_runner)

        assert result is True

    def test_dirty_repo_is_stale(self, tmp_path: Path) -> None:
        """Repo with uncommitted changes should be considered stale."""
        # Create baseline file
        report = tmp_path / "coverage.xml"
        report.write_text(VALID_COVERAGE_XML_90_PERCENT)

        # Mock git status to return dirty
        def mock_run(args: list[str], **kwargs: object) -> CommandResult:
            return CommandResult(
                command=args,
                returncode=0,
                stdout="M src/app.py\n",
                stderr="",
            )

        mock_runner = make_mock_runner(mock_run)
        result = is_baseline_stale(report, tmp_path, command_runner=mock_runner)

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
        def mock_run(args: list[str], **kwargs: object) -> CommandResult:
            if "status" in args:
                return CommandResult(command=args, returncode=0, stdout="", stderr="")
            elif "log" in args:
                # Commit time is more recent than baseline mtime
                return CommandResult(
                    command=args, returncode=0, stdout="1700000000\n", stderr=""
                )
            return CommandResult(command=args, returncode=0, stdout="", stderr="")

        mock_runner = make_mock_runner(mock_run)
        result = is_baseline_stale(report, tmp_path, command_runner=mock_runner)

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
        def mock_run(args: list[str], **kwargs: object) -> CommandResult:
            if "status" in args:
                return CommandResult(command=args, returncode=0, stdout="", stderr="")
            elif "log" in args:
                # Commit time is older than baseline mtime
                return CommandResult(
                    command=args, returncode=0, stdout="1700000000\n", stderr=""
                )
            return CommandResult(command=args, returncode=0, stdout="", stderr="")

        mock_runner = make_mock_runner(mock_run)
        result = is_baseline_stale(report, tmp_path, command_runner=mock_runner)

        assert result is False

    def test_non_git_repo_is_stale(self, tmp_path: Path) -> None:
        """Non-git repo (git commands fail) should be considered stale."""
        # Create baseline file
        report = tmp_path / "coverage.xml"
        report.write_text(VALID_COVERAGE_XML_90_PERCENT)

        # Mock git to fail (return non-zero exit code)
        def mock_run(args: list[str], **kwargs: object) -> CommandResult:
            return CommandResult(
                command=args,
                returncode=128,
                stdout="",
                stderr="fatal: not a git repository",
            )

        mock_runner = make_mock_runner(mock_run)
        result = is_baseline_stale(report, tmp_path, command_runner=mock_runner)

        assert result is True

    def test_empty_repo_no_commits_is_stale(self, tmp_path: Path) -> None:
        """Repo with no commits should be considered stale."""
        # Create baseline file
        report = tmp_path / "coverage.xml"
        report.write_text(VALID_COVERAGE_XML_90_PERCENT)

        # Mock git commands - status clean but log returns empty
        def mock_run(args: list[str], **kwargs: object) -> CommandResult:
            if "status" in args:
                return CommandResult(command=args, returncode=0, stdout="", stderr="")
            elif "log" in args:
                # No commits - empty output
                return CommandResult(command=args, returncode=0, stdout="", stderr="")
            return CommandResult(command=args, returncode=0, stdout="", stderr="")

        mock_runner = make_mock_runner(mock_run)
        result = is_baseline_stale(report, tmp_path, command_runner=mock_runner)

        assert result is True


class TestNoDecreaseMode:
    """Tests for the 'no decrease' coverage mode.

    In no-decrease mode (min_percent=None), coverage is compared against
    a baseline file rather than a fixed threshold. The baseline should be
    captured before validation and refreshed when stale.
    """

    def test_baseline_used_when_threshold_none(self, tmp_path: Path) -> None:
        """When min_percent is None, baseline coverage is used as threshold."""
        report = tmp_path / "coverage.xml"
        report.write_text(VALID_COVERAGE_XML_90_PERCENT)

        # Get baseline - this would be the main repo's coverage
        baseline = get_baseline_coverage(report)
        assert baseline == 90.0

        # Create a new coverage report with same coverage - should pass
        result = parse_and_check_coverage(report, min_percent=baseline)
        assert result.passed is True
        assert result.percent == 90.0

    def test_baseline_comparison_fails_when_coverage_decreases(
        self, tmp_path: Path
    ) -> None:
        """Coverage below baseline should fail in no-decrease mode."""
        baseline_report = tmp_path / "baseline.xml"
        baseline_report.write_text(VALID_COVERAGE_XML_90_PERCENT)

        # Get baseline coverage
        baseline = get_baseline_coverage(baseline_report)
        assert baseline == 90.0

        # Create a report with lower coverage
        current_report = tmp_path / "coverage.xml"
        current_report.write_text(VALID_COVERAGE_XML_50_PERCENT)

        # Check current against baseline - should fail
        result = parse_and_check_coverage(current_report, min_percent=baseline)
        assert result.passed is False
        assert result.status == CoverageStatus.FAILED
        assert result.failure_reason is not None
        assert "50.0%" in result.failure_reason
        assert "90.0%" in result.failure_reason

    def test_baseline_comparison_passes_when_coverage_increases(
        self, tmp_path: Path
    ) -> None:
        """Coverage above baseline should pass in no-decrease mode."""
        baseline_report = tmp_path / "baseline.xml"
        baseline_report.write_text(VALID_COVERAGE_XML_50_PERCENT)

        # Get baseline coverage
        baseline = get_baseline_coverage(baseline_report)
        assert baseline == 50.0

        # Create a report with higher coverage
        current_report = tmp_path / "coverage.xml"
        current_report.write_text(VALID_COVERAGE_XML_90_PERCENT)

        # Check current against baseline - should pass
        result = parse_and_check_coverage(current_report, min_percent=baseline)
        assert result.passed is True
        assert result.status == CoverageStatus.PASSED
        assert result.percent == 90.0

    def test_baseline_comparison_passes_when_coverage_equal(
        self, tmp_path: Path
    ) -> None:
        """Coverage equal to baseline should pass in no-decrease mode."""
        report = tmp_path / "coverage.xml"
        report.write_text(VALID_COVERAGE_XML_90_PERCENT)

        # Get baseline coverage
        baseline = get_baseline_coverage(report)
        assert baseline == 90.0

        # Check same file against baseline - should pass
        result = parse_and_check_coverage(report, min_percent=baseline)
        assert result.passed is True
        assert result.percent == 90.0

    def test_stale_baseline_detected_after_new_commit(self, tmp_path: Path) -> None:
        """Baseline should be stale when there's a newer commit."""
        report = tmp_path / "coverage.xml"
        report.write_text(VALID_COVERAGE_XML_90_PERCENT)

        # Set mtime to old time
        old_time = 946684800  # Year 2000
        os.utime(report, (old_time, old_time))

        # Mock a recent commit
        def mock_run(args: list[str], **kwargs: object) -> CommandResult:
            if "status" in args:
                return CommandResult(command=args, returncode=0, stdout="", stderr="")
            elif "log" in args:
                # Commit time is 2023 (much newer than baseline)
                return CommandResult(
                    command=args, returncode=0, stdout="1700000000\n", stderr=""
                )
            return CommandResult(command=args, returncode=0, stdout="", stderr="")

        mock_runner = make_mock_runner(mock_run)
        assert is_baseline_stale(report, tmp_path, command_runner=mock_runner) is True

    def test_fresh_baseline_not_stale_after_commit(self, tmp_path: Path) -> None:
        """Baseline should not be stale when it's newer than the last commit."""
        report = tmp_path / "coverage.xml"
        report.write_text(VALID_COVERAGE_XML_90_PERCENT)

        # Set mtime to future time (simulating baseline created after commit)
        future_time = 4102444800  # Year 2099
        os.utime(report, (future_time, future_time))

        # Mock an old commit
        def mock_run(args: list[str], **kwargs: object) -> CommandResult:
            if "status" in args:
                return CommandResult(command=args, returncode=0, stdout="", stderr="")
            elif "log" in args:
                # Old commit time
                return CommandResult(
                    command=args, returncode=0, stdout="1700000000\n", stderr=""
                )
            return CommandResult(command=args, returncode=0, stdout="", stderr="")

        mock_runner = make_mock_runner(mock_run)
        assert is_baseline_stale(report, tmp_path, command_runner=mock_runner) is False

    def test_explicit_threshold_overrides_baseline(self, tmp_path: Path) -> None:
        """When an explicit threshold is provided, baseline is not used."""
        report = tmp_path / "coverage.xml"
        report.write_text(VALID_COVERAGE_XML_50_PERCENT)  # 50% coverage

        # Even though baseline might be 90%, explicit threshold of 40% should pass
        result = parse_and_check_coverage(report, min_percent=40.0)
        assert result.passed is True
        assert result.percent == 50.0

        # Explicit threshold of 60% should fail despite coverage being "good enough"
        result = parse_and_check_coverage(report, min_percent=60.0)
        assert result.passed is False
        assert result.status == CoverageStatus.FAILED

    def test_none_baseline_returns_none(self, tmp_path: Path) -> None:
        """Missing baseline file should return None, not raise."""
        missing_report = tmp_path / "nonexistent.xml"
        result = get_baseline_coverage(missing_report)
        assert result is None

    def test_malformed_baseline_raises(self, tmp_path: Path) -> None:
        """Malformed baseline file should raise ValueError."""
        import pytest

        report = tmp_path / "coverage.xml"
        report.write_text(INVALID_XML_SYNTAX)

        with pytest.raises(ValueError) as exc_info:
            get_baseline_coverage(report)
        assert "Invalid coverage XML" in str(exc_info.value)


class TestCheckCoverageFromConfig:
    """Test check_coverage_from_config function."""

    def test_none_config_returns_none(self, tmp_path: Path) -> None:
        """When config is None, should return None (skip coverage)."""
        result = check_coverage_from_config(None, tmp_path)
        assert result is None

    def test_valid_config_passes_at_threshold(self, tmp_path: Path) -> None:
        """Coverage exactly at threshold should pass."""
        report = tmp_path / "coverage.xml"
        report.write_text(VALID_COVERAGE_XML_90_PERCENT)

        config = YamlCoverageConfig(format="xml", file="coverage.xml", threshold=90.0)
        result = check_coverage_from_config(config, tmp_path)

        assert result is not None
        assert result.passed is True
        assert result.status == CoverageStatus.PASSED
        assert result.percent == 90.0

    def test_valid_config_passes_above_threshold(self, tmp_path: Path) -> None:
        """Coverage above threshold should pass."""
        report = tmp_path / "coverage.xml"
        report.write_text(VALID_COVERAGE_XML_90_PERCENT)

        config = YamlCoverageConfig(format="xml", file="coverage.xml", threshold=85.0)
        result = check_coverage_from_config(config, tmp_path)

        assert result is not None
        assert result.passed is True
        assert result.status == CoverageStatus.PASSED
        assert result.percent == 90.0

    def test_valid_config_fails_below_threshold(self, tmp_path: Path) -> None:
        """Coverage below threshold should fail."""
        report = tmp_path / "coverage.xml"
        report.write_text(VALID_COVERAGE_XML_50_PERCENT)

        config = YamlCoverageConfig(format="xml", file="coverage.xml", threshold=85.0)
        result = check_coverage_from_config(config, tmp_path)

        assert result is not None
        assert result.passed is False
        assert result.status == CoverageStatus.FAILED
        assert result.percent == 50.0
        assert result.failure_reason is not None
        assert "50.0%" in result.failure_reason
        assert "85.0%" in result.failure_reason

    def test_custom_file_path(self, tmp_path: Path) -> None:
        """Should use custom file path from config."""
        subdir = tmp_path / "reports"
        subdir.mkdir()
        report = subdir / "custom-coverage.xml"
        report.write_text(VALID_COVERAGE_XML_90_PERCENT)

        config = YamlCoverageConfig(
            format="xml", file="reports/custom-coverage.xml", threshold=85.0
        )
        result = check_coverage_from_config(config, tmp_path)

        assert result is not None
        assert result.passed is True
        assert result.percent == 90.0

    def test_absolute_file_path(self, tmp_path: Path) -> None:
        """Should handle absolute file paths."""
        report = tmp_path / "coverage.xml"
        report.write_text(VALID_COVERAGE_XML_90_PERCENT)

        config = YamlCoverageConfig(format="xml", file=str(report), threshold=85.0)
        result = check_coverage_from_config(config, tmp_path)

        assert result is not None
        assert result.passed is True
        assert result.percent == 90.0

    def test_missing_coverage_file_error(self, tmp_path: Path) -> None:
        """Missing coverage file should return error result."""
        config = YamlCoverageConfig(
            format="xml", file="nonexistent.xml", threshold=85.0
        )
        result = check_coverage_from_config(config, tmp_path)

        assert result is not None
        assert result.passed is False
        assert result.status == CoverageStatus.ERROR
        assert result.failure_reason is not None
        assert "not found" in result.failure_reason

    def test_invalid_xml_error(self, tmp_path: Path) -> None:
        """Invalid XML should return error result."""
        report = tmp_path / "coverage.xml"
        report.write_text(INVALID_XML_SYNTAX)

        config = YamlCoverageConfig(format="xml", file="coverage.xml", threshold=85.0)
        result = check_coverage_from_config(config, tmp_path)

        assert result is not None
        assert result.passed is False
        assert result.status == CoverageStatus.ERROR
        assert result.failure_reason is not None
        assert "Invalid coverage XML" in result.failure_reason

    def test_threshold_zero_passes_any_coverage(self, tmp_path: Path) -> None:
        """Threshold of 0 should pass any coverage."""
        report = tmp_path / "coverage.xml"
        report.write_text(VALID_COVERAGE_XML_50_PERCENT)

        config = YamlCoverageConfig(format="xml", file="coverage.xml", threshold=0.0)
        result = check_coverage_from_config(config, tmp_path)

        assert result is not None
        assert result.passed is True
        assert result.percent == 50.0

    def test_threshold_100_requires_full_coverage(self, tmp_path: Path) -> None:
        """Threshold of 100 should only pass 100% coverage."""
        # Test failing case
        report = tmp_path / "coverage.xml"
        report.write_text(VALID_COVERAGE_XML_90_PERCENT)

        config = YamlCoverageConfig(format="xml", file="coverage.xml", threshold=100.0)
        result = check_coverage_from_config(config, tmp_path)

        assert result is not None
        assert result.passed is False
        assert result.status == CoverageStatus.FAILED

        # Test passing case
        report.write_text(VALID_COVERAGE_XML_100_PERCENT)
        result = check_coverage_from_config(config, tmp_path)

        assert result is not None
        assert result.passed is True
        assert result.percent == 100.0


class TestBaselineCoverageService:
    """Test BaselineCoverageService with YamlCoverageConfig."""

    def test_no_coverage_config_returns_unavailable(self, tmp_path: Path) -> None:
        """Without coverage config, baseline refresh is unavailable."""
        from unittest.mock import MagicMock

        # Create service without coverage config
        service = BaselineCoverageService(
            repo_path=tmp_path,
            env_config=make_mock_env_config(),
            command_runner=make_mock_runner(lambda *a, **kw: None),
            lock_manager=FakeLockManager(),
            coverage_config=None,
        )

        # Mock spec - shouldn't matter since we fail early
        mock_spec = MagicMock()

        result = service.refresh_if_stale(mock_spec)

        assert result.success is False
        assert result.error is not None
        assert "no coverage configuration" in result.error

    def test_no_coverage_command_returns_unavailable(self, tmp_path: Path) -> None:
        """Without coverage command, baseline refresh is unavailable."""
        from unittest.mock import MagicMock

        # Create config without command
        config = YamlCoverageConfig(
            format="xml",
            file="coverage.xml",
            threshold=85.0,
            command=None,
        )
        service = BaselineCoverageService(
            repo_path=tmp_path,
            env_config=make_mock_env_config(),
            command_runner=make_mock_runner(lambda *a, **kw: None),
            lock_manager=FakeLockManager(),
            coverage_config=config,
        )

        # Mock spec - shouldn't matter since we fail early
        mock_spec = MagicMock()

        result = service.refresh_if_stale(mock_spec)

        assert result.success is False
        assert result.error is not None
        assert "no coverage command configured" in result.error

    def test_coverage_config_timeout_is_used(self, tmp_path: Path) -> None:
        """Coverage config timeout should be used for command execution."""
        # Create config with custom timeout
        config = YamlCoverageConfig(
            format="xml",
            file="coverage.xml",
            threshold=85.0,
            command="uv run pytest --cov",
            timeout=600,
        )
        service = BaselineCoverageService(
            repo_path=tmp_path,
            env_config=make_mock_env_config(),
            command_runner=make_mock_runner(lambda *a, **kw: None),
            lock_manager=FakeLockManager(),
            coverage_config=config,
            step_timeout_seconds=120.0,  # This should be ignored
        )

        # Verify config is stored correctly
        assert service.coverage_config is not None
        assert service.coverage_config.timeout == 600
        assert service.coverage_config.command == "uv run pytest --cov"

    def test_fallback_to_step_timeout(self, tmp_path: Path) -> None:
        """When coverage config has no timeout, step_timeout_seconds is used."""
        # Create config without timeout
        config = YamlCoverageConfig(
            format="xml",
            file="coverage.xml",
            threshold=85.0,
            command="uv run pytest --cov",
            timeout=None,
        )
        service = BaselineCoverageService(
            repo_path=tmp_path,
            env_config=make_mock_env_config(),
            command_runner=make_mock_runner(lambda *a, **kw: None),
            lock_manager=FakeLockManager(),
            coverage_config=config,
            step_timeout_seconds=180.0,
        )

        # Verify config is stored correctly
        assert service.coverage_config is not None
        assert service.coverage_config.timeout is None
        # step_timeout_seconds should be the fallback
        assert service.step_timeout_seconds == 180.0

    def test_fresh_baseline_returns_without_refresh(self, tmp_path: Path) -> None:
        """Fresh baseline should return cached value without running command."""
        from unittest.mock import MagicMock

        # Create coverage.xml baseline
        report = tmp_path / "coverage.xml"
        report.write_text(VALID_COVERAGE_XML_90_PERCENT)

        # Set future mtime (simulating fresh baseline)
        future_time = 4102444800  # Year 2099
        os.utime(report, (future_time, future_time))

        # Create config with command
        config = YamlCoverageConfig(
            format="xml",
            file="coverage.xml",
            threshold=85.0,
            command="uv run pytest --cov",
        )

        # Mock git commands to indicate clean repo
        def mock_run(args: list[str], **kwargs: object) -> CommandResult:
            if "status" in args:
                return CommandResult(command=args, returncode=0, stdout="", stderr="")
            elif "log" in args:
                # Old commit time
                return CommandResult(
                    command=args, returncode=0, stdout="1700000000\n", stderr=""
                )
            return CommandResult(command=args, returncode=0, stdout="", stderr="")

        mock_runner = make_mock_runner(mock_run)
        lock_manager = FakeLockManager()
        service = BaselineCoverageService(
            repo_path=tmp_path,
            env_config=make_mock_env_config(),
            command_runner=mock_runner,
            lock_manager=lock_manager,
            coverage_config=config,
        )

        # Mock spec
        mock_spec = MagicMock()

        result = service.refresh_if_stale(mock_spec)

        # Should return existing baseline without refresh
        assert result.success is True
        assert result.percent == 90.0

    def test_injected_lock_manager_try_lock_called(self, tmp_path: Path) -> None:
        """When lock_manager is injected, it should be used for locking."""
        from unittest.mock import MagicMock

        # Create stale baseline (old mtime)
        report = tmp_path / "coverage.xml"
        report.write_text(VALID_COVERAGE_XML_90_PERCENT)
        old_time = 1000000000  # Year 2001
        os.utime(report, (old_time, old_time))

        # Create config with command
        config = YamlCoverageConfig(
            format="xml",
            file="coverage.xml",
            threshold=85.0,
            command="uv run pytest --cov",
        )

        # Mock git commands to indicate stale repo
        def mock_run(args: list[str], **kwargs: object) -> CommandResult:
            if "status" in args:
                return CommandResult(command=args, returncode=0, stdout="", stderr="")
            elif "log" in args:
                # Recent commit time (after baseline)
                return CommandResult(
                    command=args, returncode=0, stdout="2000000000\n", stderr=""
                )
            return CommandResult(command=args, returncode=0, stdout="", stderr="")

        mock_runner = make_mock_runner(mock_run)

        # Pre-populate lock with another agent to simulate contention (causes timeout)
        lock_manager = FakeLockManager()
        lock_manager.locks["coverage-baseline.lock"] = "other-agent"

        service = BaselineCoverageService(
            repo_path=tmp_path,
            env_config=make_mock_env_config(),
            command_runner=mock_runner,
            lock_manager=lock_manager,
            coverage_config=config,
        )

        mock_spec = MagicMock()
        result = service.refresh_if_stale(mock_spec)

        # Should fail with timeout since lock is held by another agent
        assert result.success is False
        assert "Timeout" in (result.error or "")

        # Verify lock manager recorded the acquisition attempts
        assert len(lock_manager.acquire_calls) > 0
        assert len(lock_manager.wait_for_lock_calls) > 0

    def test_injected_lock_manager_release_called_on_success(
        self, tmp_path: Path
    ) -> None:
        """When lock is acquired, release_lock should be called with agent_id."""
        from unittest.mock import MagicMock

        # Create stale baseline (old mtime, recent commit means stale)
        report = tmp_path / "coverage.xml"
        report.write_text(VALID_COVERAGE_XML_90_PERCENT)
        old_time = 1000000000
        os.utime(report, (old_time, old_time))

        config = YamlCoverageConfig(
            format="xml",
            file="coverage.xml",
            threshold=85.0,
            command="uv run pytest --cov",
        )

        # Mock git: stale first check, then fresh after lock (double-check pattern)
        call_count = {"is_stale_check": 0}

        def mock_run(args: list[str], **kwargs: object) -> CommandResult:
            if "status" in args:
                return CommandResult(command=args, returncode=0, stdout="", stderr="")
            elif "log" in args:
                call_count["is_stale_check"] += 1
                if call_count["is_stale_check"] == 1:
                    # First check: stale (commit after baseline mtime)
                    return CommandResult(
                        command=args, returncode=0, stdout="2000000000\n", stderr=""
                    )
                else:
                    # After lock: fresh (commit before new baseline mtime)
                    return CommandResult(
                        command=args, returncode=0, stdout="500000000\n", stderr=""
                    )
            return CommandResult(command=args, returncode=0, stdout="", stderr="")

        mock_runner = make_mock_runner(mock_run)

        # Subclass FakeLockManager to update mtime when lock is acquired
        class LockManagerWithCallback(FakeLockManager):
            def try_lock(
                self, filepath: str, agent_id: str, repo_namespace: str | None = None
            ) -> bool:
                result = super().try_lock(filepath, agent_id, repo_namespace)
                if result:
                    # Simulate another agent refreshing baseline while we waited
                    future_time = 4102444800
                    os.utime(report, (future_time, future_time))
                return result

        lock_manager = LockManagerWithCallback()

        service = BaselineCoverageService(
            repo_path=tmp_path,
            env_config=make_mock_env_config(),
            command_runner=mock_runner,
            lock_manager=lock_manager,
            coverage_config=config,
        )

        mock_spec = MagicMock()
        result = service.refresh_if_stale(mock_spec)

        # Should succeed with cached baseline
        assert result.success is True
        assert result.percent == 90.0

        # Verify lock was acquired and released (lock should be empty after release)
        assert len(lock_manager.acquire_calls) > 0
        # Verify agent_id pattern from acquire_calls (same ID used for release)
        agent_id = lock_manager.acquire_calls[0].agent_id
        assert "baseline-refresh-" in agent_id
        # Lock should be released (not in locks dict)
        assert "coverage-baseline.lock" not in lock_manager.locks

    def test_injected_lock_manager_release_called_on_error(
        self, tmp_path: Path
    ) -> None:
        """release_lock should be called even when refresh fails."""
        from unittest.mock import MagicMock

        # Create stale baseline
        report = tmp_path / "coverage.xml"
        report.write_text(VALID_COVERAGE_XML_90_PERCENT)
        old_time = 1000000000
        os.utime(report, (old_time, old_time))

        config = YamlCoverageConfig(
            format="xml",
            file="coverage.xml",
            threshold=85.0,
            command="uv run pytest --cov",
        )

        # Mock git: always stale, worktree creation will fail
        def mock_run(args: list[str], **kwargs: object) -> CommandResult:
            if "status" in args:
                return CommandResult(command=args, returncode=0, stdout="", stderr="")
            elif "log" in args:
                return CommandResult(
                    command=args, returncode=0, stdout="2000000000\n", stderr=""
                )
            elif "worktree" in args:
                return CommandResult(
                    command=args, returncode=1, stdout="", stderr="worktree failed"
                )
            return CommandResult(command=args, returncode=0, stdout="", stderr="")

        mock_runner = make_mock_runner(mock_run)

        lock_manager = FakeLockManager()

        service = BaselineCoverageService(
            repo_path=tmp_path,
            env_config=make_mock_env_config(),
            command_runner=mock_runner,
            lock_manager=lock_manager,
            coverage_config=config,
        )

        mock_spec = MagicMock()
        result = service.refresh_if_stale(mock_spec)

        # Should fail (worktree creation failed)
        assert result.success is False

        # Verify lock was acquired and released even on error (finally block)
        assert len(lock_manager.acquire_calls) > 0
        # Verify agent_id pattern
        agent_id = lock_manager.acquire_calls[0].agent_id
        assert "baseline-refresh-" in agent_id
        # Lock should be released (not in locks dict)
        assert "coverage-baseline.lock" not in lock_manager.locks
