"""Unit tests for SpecResultBuilder."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.domain.validation.coverage import CoverageResult, CoverageStatus
from src.domain.validation.spec import CoverageConfig
from src.domain.validation.spec_result_builder import SpecResultBuilder
from src.infra.tools.command_runner import CommandResult


@pytest.mark.unit
class TestCheckCoverage:
    """Tests for _check_coverage method."""

    def test_report_not_found_returns_error(self, tmp_path: Path) -> None:
        """Returns error when coverage report doesn't exist."""
        builder = SpecResultBuilder()
        config = CoverageConfig(enabled=True, min_percent=80.0)

        result = builder._check_coverage(
            config=config,
            cwd=tmp_path,
            log_dir=tmp_path,
            baseline_percent=None,
        )

        assert result.passed is False
        assert result.status == CoverageStatus.ERROR
        assert "not found" in (result.failure_reason or "")

    def test_uses_baseline_when_no_min_percent(self, tmp_path: Path) -> None:
        """Uses baseline_percent when min_percent is None."""
        builder = SpecResultBuilder()
        # Create a coverage.xml with 75% coverage
        coverage_xml = tmp_path / "coverage.xml"
        coverage_xml.write_text(
            '<?xml version="1.0"?>'
            '<coverage line-rate="0.75" lines-valid="100" lines-covered="75"/>'
        )
        config = CoverageConfig(enabled=True, min_percent=None)

        result = builder._check_coverage(
            config=config,
            cwd=tmp_path,
            log_dir=tmp_path,
            baseline_percent=80.0,  # Require 80% but we have 75%
        )

        # Should fail because 75% < 80%
        assert result.passed is False
        assert result.percent == 75.0

    def test_uses_relative_report_path(self, tmp_path: Path) -> None:
        """Resolves relative report path against cwd."""
        builder = SpecResultBuilder()

        # Create coverage.xml in a subdirectory
        subdir = tmp_path / "reports"
        subdir.mkdir()
        coverage_xml = subdir / "coverage.xml"
        coverage_xml.write_text(
            '<?xml version="1.0"?>'
            '<coverage line-rate="0.95" lines-valid="100" lines-covered="95"/>'
        )

        config = CoverageConfig(
            enabled=True,
            min_percent=90.0,
            report_path=Path("reports/coverage.xml"),  # Relative path
        )

        result = builder._check_coverage(
            config=config,
            cwd=tmp_path,
            log_dir=tmp_path,
            baseline_percent=None,
        )

        assert result.passed is True
        assert result.percent == 95.0


@pytest.mark.unit
class TestRunCoverageCommandIfConfigured:
    """Tests for _run_coverage_command_if_configured method."""

    def test_returns_none_when_no_command(self, tmp_path: Path) -> None:
        """Returns None when no coverage command is configured."""
        from src.domain.validation.config import YamlCoverageConfig

        builder = SpecResultBuilder()
        config = YamlCoverageConfig(
            format="xml",
            file="coverage.xml",
            threshold=80.0,
            command=None,
        )
        mock_runner = MagicMock()

        result = builder._run_coverage_command_if_configured(
            coverage_config=config,
            cwd=tmp_path,
            env={},
            command_runner=mock_runner,
        )

        assert result is None
        mock_runner.run.assert_not_called()

    def test_returns_none_on_command_success(self, tmp_path: Path) -> None:
        """Returns None when coverage command succeeds."""
        from src.domain.validation.config import YamlCoverageConfig

        builder = SpecResultBuilder()
        config = YamlCoverageConfig(
            format="xml",
            file="coverage.xml",
            threshold=80.0,
            command="pytest --cov",
        )
        mock_runner = MagicMock()
        mock_runner.run.return_value = CommandResult(
            command=["pytest", "--cov"],
            returncode=0,
            stdout="ok",
            stderr="",
        )

        result = builder._run_coverage_command_if_configured(
            coverage_config=config,
            cwd=tmp_path,
            env={},
            command_runner=mock_runner,
        )

        assert result is None

    def test_returns_failure_on_command_timeout(self, tmp_path: Path) -> None:
        """Returns failure result when coverage command times out."""
        from src.domain.validation.config import YamlCoverageConfig

        builder = SpecResultBuilder()
        config = YamlCoverageConfig(
            format="xml",
            file="coverage.xml",
            threshold=80.0,
            command="pytest --cov",
            timeout=60,
        )
        mock_runner = MagicMock()
        mock_runner.run.return_value = CommandResult(
            command=["pytest", "--cov"],
            returncode=1,
            stdout="",
            stderr="",
            timed_out=True,
        )

        result = builder._run_coverage_command_if_configured(
            coverage_config=config,
            cwd=tmp_path,
            env={},
            command_runner=mock_runner,
        )

        assert result is not None
        assert result.passed is False
        assert "timed out" in (result.failure_reason or "")

    def test_returns_failure_on_command_error(self, tmp_path: Path) -> None:
        """Returns failure result when coverage command fails."""
        from src.domain.validation.config import YamlCoverageConfig

        builder = SpecResultBuilder()
        config = YamlCoverageConfig(
            format="xml",
            file="coverage.xml",
            threshold=80.0,
            command="pytest --cov",
        )
        mock_runner = MagicMock()
        mock_runner.run.return_value = CommandResult(
            command=["pytest", "--cov"],
            returncode=1,
            stdout="",
            stderr="test failed",
        )

        result = builder._run_coverage_command_if_configured(
            coverage_config=config,
            cwd=tmp_path,
            env={},
            command_runner=mock_runner,
        )

        assert result is not None
        assert result.passed is False
        assert "exited 1" in (result.failure_reason or "")


@pytest.mark.unit
class TestBuildFailureResult:
    """Tests for _build_failure_result method."""

    def test_creates_failure_result_with_reason(self, tmp_path: Path) -> None:
        """Creates ValidationResult with passed=False and failure reason."""
        from src.domain.validation.spec import ValidationArtifacts

        builder = SpecResultBuilder()
        artifacts = ValidationArtifacts(log_dir=tmp_path)

        result = builder._build_failure_result(
            steps=[],
            reason="Test failure",
            artifacts=artifacts,
        )

        assert result.passed is False
        assert "Test failure" in result.failure_reasons

    def test_includes_coverage_and_e2e_results(self, tmp_path: Path) -> None:
        """Includes coverage and E2E results in failure."""
        from src.domain.validation.e2e import E2EResult, E2EStatus
        from src.domain.validation.spec import ValidationArtifacts

        builder = SpecResultBuilder()
        artifacts = ValidationArtifacts(log_dir=tmp_path)
        cov = CoverageResult(
            percent=50.0,
            passed=False,
            status=CoverageStatus.FAILED,
            report_path=tmp_path / "coverage.xml",
        )
        e2e = E2EResult(
            passed=False,
            status=E2EStatus.FAILED,
            failure_reason="E2E failed",
            returncode=1,
        )

        result = builder._build_failure_result(
            steps=[],
            reason="Coverage too low",
            artifacts=artifacts,
            coverage_result=cov,
            e2e_result=e2e,
        )

        assert result.passed is False
        assert result.coverage_result == cov
        assert result.e2e_result == e2e
