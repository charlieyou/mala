"""Result building stage for mala validation.

This module provides SpecResultBuilder which handles the post-command-execution
stages of validation:
- Coverage checks (threshold or no-decrease mode)
- E2E execution (run-level only)
- ValidationResult assembly

The builder is called after commands have been executed and produces the final
ValidationResult.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .coverage import (
    CoverageResult,
    CoverageStatus,
    parse_and_check_coverage,
)
from .e2e import E2EConfig as E2ERunnerConfig
from .e2e import E2ERunner, E2EStatus
from .result import ValidationResult

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from .e2e import E2EResult
    from .result import ValidationStepResult
    from .spec import (
        CoverageConfig,
        E2EConfig,
        ValidationArtifacts,
        ValidationContext,
        ValidationSpec,
    )


@dataclass
class ResultBuilderInput:
    """Input for the result builder stage.

    Attributes:
        spec: The validation spec being executed.
        context: The validation context.
        steps: Steps from command execution.
        artifacts: Validation artifacts to update.
        cwd: Working directory where coverage.xml should be.
        log_dir: Directory for logs.
        env: Environment variables for E2E.
        baseline_percent: Baseline coverage for "no decrease" mode.
    """

    spec: ValidationSpec
    context: ValidationContext
    steps: list[ValidationStepResult]
    artifacts: ValidationArtifacts
    cwd: Path
    log_dir: Path
    env: Mapping[str, str]
    baseline_percent: float | None


class SpecResultBuilder:
    """Builds ValidationResult from command execution output.

    This builder handles the post-processing stages after commands have been
    executed:
    1. Coverage check (if enabled)
    2. E2E execution (if enabled and scope is run-level)
    3. Final ValidationResult assembly

    The builder owns the coverage and E2E checking logic, keeping the runner
    focused on workspace setup and command execution.
    """

    def build(self, input: ResultBuilderInput) -> ValidationResult:
        """Build the final ValidationResult.

        Args:
            input: The builder input containing steps, artifacts, and config.

        Returns:
            Complete ValidationResult with coverage/E2E results.
        """
        # Step 1: Check coverage
        cov = self._check_coverage_if_enabled(
            spec=input.spec,
            cwd=input.cwd,
            log_dir=input.log_dir,
            artifacts=input.artifacts,
            baseline_percent=input.baseline_percent,
        )
        if cov is not None and not cov.passed:
            reason = cov.failure_reason or "Coverage check failed"
            return self._build_failure_result(
                steps=input.steps,
                reason=reason,
                artifacts=input.artifacts,
                coverage_result=cov,
            )

        # Step 2: Run E2E
        e2e = self._run_e2e_if_enabled(
            spec=input.spec,
            env=input.env,
            cwd=input.cwd,
            log_dir=input.log_dir,
            artifacts=input.artifacts,
        )
        if e2e is not None and not e2e.passed and e2e.status != E2EStatus.SKIPPED:
            reason = e2e.failure_reason or "E2E failed"
            return self._build_failure_result(
                steps=input.steps,
                reason=reason,
                artifacts=input.artifacts,
                coverage_result=cov,
                e2e_result=e2e,
            )

        # Step 3: Success
        return ValidationResult(
            passed=True,
            steps=input.steps,
            artifacts=input.artifacts,
            coverage_result=cov,
            e2e_result=e2e,
        )

    def _check_coverage_if_enabled(
        self,
        spec: ValidationSpec,
        cwd: Path,
        log_dir: Path,
        artifacts: ValidationArtifacts,
        baseline_percent: float | None,
    ) -> CoverageResult | None:
        """Run coverage check if enabled.

        Args:
            spec: Validation spec with coverage config.
            cwd: Working directory where coverage.xml should be.
            log_dir: Directory for logs.
            artifacts: Artifacts to update with coverage report path.
            baseline_percent: Baseline coverage for "no decrease" mode.

        Returns:
            CoverageResult if coverage is enabled, None otherwise.
        """
        if not spec.coverage.enabled:
            return None

        coverage_result = self._check_coverage(
            spec.coverage, cwd, log_dir, baseline_percent
        )
        if coverage_result.report_path:
            artifacts.coverage_report = coverage_result.report_path

        return coverage_result

    def _check_coverage(
        self,
        config: CoverageConfig,
        cwd: Path,
        log_dir: Path,
        baseline_percent: float | None = None,
    ) -> CoverageResult:
        """Check coverage against threshold.

        Args:
            config: Coverage configuration.
            cwd: Working directory where coverage.xml should be.
            log_dir: Directory for logs.
            baseline_percent: Baseline coverage percentage for "no decrease" mode.
                Used when config.min_percent is None.

        Returns:
            CoverageResult with pass/fail status.
        """
        # Look for coverage.xml in cwd
        # Resolve relative paths against cwd to ensure correct file lookup
        if config.report_path is not None:
            report_path = config.report_path
            if not report_path.is_absolute():
                report_path = cwd / report_path
        else:
            report_path = cwd / "coverage.xml"

        if not report_path.exists():
            # Coverage report not found - this is only an error if coverage was expected
            return CoverageResult(
                percent=None,
                passed=False,
                status=CoverageStatus.ERROR,
                report_path=report_path,
                failure_reason=f"Coverage report not found: {report_path}",
            )

        # Determine threshold: use config.min_percent if set, else baseline_percent
        threshold = (
            config.min_percent if config.min_percent is not None else baseline_percent
        )

        return parse_and_check_coverage(report_path, threshold)

    def _run_e2e_if_enabled(
        self,
        spec: ValidationSpec,
        env: Mapping[str, str],
        cwd: Path,
        log_dir: Path,
        artifacts: ValidationArtifacts,
    ) -> E2EResult | None:
        """Run E2E validation if enabled (only for run-level scope).

        Args:
            spec: Validation spec with E2E config.
            env: Environment variables.
            cwd: Working directory.
            log_dir: Directory for logs.
            artifacts: Artifacts to update with fixture path.

        Returns:
            E2EResult if E2E is enabled and scope is run-level, None otherwise.
        """
        from .spec import ValidationScope

        if not spec.e2e.enabled or spec.scope != ValidationScope.RUN_LEVEL:
            return None

        e2e_result = self._run_e2e(spec.e2e, env, cwd, log_dir)
        if e2e_result.fixture_path:
            artifacts.e2e_fixture_path = e2e_result.fixture_path

        return e2e_result

    def _run_e2e(
        self,
        config: E2EConfig,
        env: Mapping[str, str],
        cwd: Path,
        log_dir: Path,
    ) -> E2EResult:
        """Run E2E validation using the E2ERunner.

        Args:
            config: E2E configuration from spec.
            env: Environment variables.
            cwd: Working directory.
            log_dir: Directory for logs.

        Returns:
            E2EResult with pass/fail status.
        """
        runner_config = E2ERunnerConfig(
            enabled=config.enabled,
            skip_if_no_keys=True,  # Don't fail if API key missing
            keep_fixture=True,  # Keep for debugging
        )
        runner = E2ERunner(runner_config)
        return runner.run(env=dict(env), cwd=cwd)

    def _build_failure_result(
        self,
        steps: list[ValidationStepResult],
        reason: str,
        artifacts: ValidationArtifacts,
        coverage_result: CoverageResult | None = None,
        e2e_result: E2EResult | None = None,
    ) -> ValidationResult:
        """Build a failed ValidationResult.

        Args:
            steps: Command execution steps.
            reason: Failure reason.
            artifacts: Validation artifacts.
            coverage_result: Optional coverage result.
            e2e_result: Optional E2E result.

        Returns:
            ValidationResult with passed=False.
        """
        return ValidationResult(
            passed=False,
            steps=steps,
            failure_reasons=[reason],
            artifacts=artifacts,
            coverage_result=coverage_result,
            e2e_result=e2e_result,
        )
