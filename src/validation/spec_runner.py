"""Spec-based validation runner for mala.

This module provides SpecValidationRunner which runs validation using
ValidationSpec + ValidationContext, the modern API for mala validation.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING

from ..tools.env import get_lock_dir
from .coverage import (
    CoverageResult,
    CoverageStatus,
    parse_and_check_coverage,
)
from .e2e import E2EConfig as E2ERunnerConfig
from .e2e import E2ERunner, E2EStatus
from .spec import ValidationArtifacts
from .spec_executor import (
    ExecutorConfig,
    ExecutorInput,
    SpecCommandExecutor,
)
from .spec_workspace import (
    SetupError,
    cleanup_workspace,
    setup_workspace,
)

if TYPE_CHECKING:
    from pathlib import Path

    from .e2e import E2EResult
    from .result import ValidationStepResult
    from .spec import (
        CoverageConfig,
        E2EConfig,
        ValidationContext,
        ValidationSpec,
    )

from .result import ValidationResult


class CommandFailure(Exception):
    """Raised when a command fails during validation.

    Attributes:
        steps: The steps executed so far (including the failed step).
        reason: Human-readable failure reason.
    """

    def __init__(self, steps: list[ValidationStepResult], reason: str) -> None:
        super().__init__(reason)
        self.steps = steps
        self.reason = reason


class SpecValidationRunner:
    """Runs validation according to a ValidationSpec.

    This runner supports:
    - Scope-aware validation (per-issue vs run-level)
    - Per-command mutex settings
    - Integrated worktree, coverage, and E2E handling
    - Artifact tracking
    - Lint caching to skip redundant lint commands
    """

    def __init__(
        self,
        repo_path: Path,
        step_timeout_seconds: float | None = None,
        enable_lint_cache: bool = True,
    ):
        """Initialize the spec validation runner.

        Args:
            repo_path: Path to the repository to validate.
            step_timeout_seconds: Optional timeout for individual steps.
            enable_lint_cache: Whether to enable lint caching. Set to False
                in tests or when caching is not desired.
        """
        self.repo_path = repo_path.resolve()
        self.step_timeout_seconds = step_timeout_seconds
        self.enable_lint_cache = enable_lint_cache

    async def run_spec(
        self,
        spec: ValidationSpec,
        context: ValidationContext,
        log_dir: Path | None = None,
    ) -> ValidationResult:
        """Run validation according to a ValidationSpec.

        Args:
            spec: What validations to run.
            context: Immutable context for the validation run.
            log_dir: Directory for logs/artifacts. Uses temp dir if None.

        Returns:
            ValidationResult with steps, artifacts, and coverage info.
        """
        return await asyncio.to_thread(self._run_spec_sync, spec, context, log_dir)

    def _run_spec_sync(
        self,
        spec: ValidationSpec,
        context: ValidationContext,
        log_dir: Path | None = None,
    ) -> ValidationResult:
        """Synchronous implementation of run_spec.

        Uses a pipeline pattern: setup_workspace -> run_commands -> check_coverage -> run_e2e -> build_result.
        Delegates workspace/baseline/worktree setup to spec_workspace module.
        """
        # Delegate workspace setup to spec_workspace module
        try:
            workspace = setup_workspace(
                spec=spec,
                context=context,
                log_dir=log_dir,
                step_timeout_seconds=self.step_timeout_seconds,
            )
        except SetupError as e:
            # Return early failure for setup errors
            artifacts = ValidationArtifacts(log_dir=log_dir) if log_dir else None
            return ValidationResult(
                passed=False,
                failure_reasons=[e.reason],
                retriable=e.retriable,
                artifacts=artifacts,
            )

        # Execute pipeline and capture result, ensuring worktree cleanup
        result: ValidationResult | None = None
        try:
            result = self._run_validation_pipeline(
                spec,
                context,
                workspace.validation_cwd,
                workspace.artifacts,
                workspace.log_dir,
                workspace.run_id,
                workspace.baseline_percent,
            )
            return result
        finally:
            # Clean up workspace with correct pass/fail status
            # On exception, result is None so we treat as failed (validation_passed=False)
            validation_passed = result.passed if result is not None else False
            cleanup_workspace(workspace, validation_passed)

    def _run_validation_pipeline(
        self,
        spec: ValidationSpec,
        context: ValidationContext,
        cwd: Path,
        artifacts: ValidationArtifacts,
        log_dir: Path,
        run_id: str,
        baseline_percent: float | None,
    ) -> ValidationResult:
        """Run pipeline: commands -> coverage -> e2e -> result."""
        env = self._build_spec_env(context, run_id)
        expected = [cmd.name for cmd in spec.commands]
        self._write_initial_manifest(log_dir, expected, cwd, run_id, context, spec)

        # Step 1: Run commands
        try:
            steps = self._run_commands(spec, cwd, env, log_dir)
        except CommandFailure as e:
            self._write_completion_manifest(log_dir, expected, e.steps, e.reason)
            return self._build_failure_result(e.steps, e.reason, artifacts)

        # Step 2: Check coverage
        cov = self._check_coverage_if_enabled(
            spec, cwd, log_dir, artifacts, baseline_percent
        )
        if cov is not None and not cov.passed:
            reason = cov.failure_reason or "Coverage check failed"
            self._write_completion_manifest(log_dir, expected, steps, reason)
            return self._build_failure_result(
                steps, reason, artifacts, coverage_result=cov
            )

        # Step 3: Run E2E
        e2e = self._run_e2e_if_enabled(spec, env, cwd, log_dir, artifacts)
        if e2e is not None and not e2e.passed and e2e.status != E2EStatus.SKIPPED:
            reason = e2e.failure_reason or "E2E failed"
            self._write_completion_manifest(log_dir, expected, steps, reason)
            return self._build_failure_result(steps, reason, artifacts, cov, e2e)

        # Step 4: Success
        self._write_completion_manifest(log_dir, expected, steps, None)
        return ValidationResult(
            passed=True,
            steps=steps,
            artifacts=artifacts,
            coverage_result=cov,
            e2e_result=e2e,
        )

    def _build_failure_result(
        self,
        steps: list[ValidationStepResult],
        reason: str,
        artifacts: ValidationArtifacts,
        coverage_result: CoverageResult | None = None,
        e2e_result: E2EResult | None = None,
    ) -> ValidationResult:
        """Build a failed ValidationResult."""
        return ValidationResult(
            passed=False,
            steps=steps,
            failure_reasons=[reason],
            artifacts=artifacts,
            coverage_result=coverage_result,
            e2e_result=e2e_result,
        )

    def _write_initial_manifest(
        self,
        log_dir: Path,
        expected_commands: list[str],
        cwd: Path,
        run_id: str,
        context: ValidationContext,
        spec: ValidationSpec,
    ) -> None:
        """Write initial manifest of expected commands for debugging."""
        manifest_path = log_dir / "validation_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "expected_commands": expected_commands,
                    "cwd": str(cwd),
                    "run_id": run_id,
                    "issue_id": context.issue_id,
                    "scope": spec.scope.value,
                },
                indent=2,
            )
        )

    def _run_commands(
        self,
        spec: ValidationSpec,
        cwd: Path,
        env: dict[str, str],
        log_dir: Path,
    ) -> list[ValidationStepResult]:
        """Execute all commands in the spec.

        Delegates to SpecCommandExecutor for command execution and lint-cache
        handling. The executor encapsulates all execution logic.

        Args:
            spec: Validation spec with commands.
            cwd: Working directory for commands.
            env: Environment variables.
            log_dir: Directory for logs.

        Returns:
            List of step results for all commands.

        Raises:
            CommandFailure: If a command fails (and allow_fail is False).
        """
        # Configure executor
        executor_config = ExecutorConfig(
            enable_lint_cache=self.enable_lint_cache,
            repo_path=self.repo_path,
            step_timeout_seconds=self.step_timeout_seconds,
        )
        executor = SpecCommandExecutor(executor_config)

        # Build executor input
        executor_input = ExecutorInput(
            commands=spec.commands,
            cwd=cwd,
            env=env,
            log_dir=log_dir,
        )

        # Execute commands
        output = executor.execute(executor_input)

        # Raise CommandFailure if execution failed
        if output.failed:
            raise CommandFailure(
                output.steps, output.failure_reason or "Command failed"
            )

        return output.steps

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

    def _run_e2e_if_enabled(
        self,
        spec: ValidationSpec,
        env: dict[str, str],
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

        e2e_result = self._run_spec_e2e(spec.e2e, env, cwd, log_dir)
        if e2e_result.fixture_path:
            artifacts.e2e_fixture_path = e2e_result.fixture_path

        return e2e_result

    def _write_completion_manifest(
        self,
        log_dir: Path,
        expected_commands: list[str],
        steps: list[ValidationStepResult],
        failure_reason: str | None,
    ) -> None:
        """Write completion manifest with expected vs actual commands.

        This helps debug cases where commands are unexpectedly skipped.
        """
        actual_commands = [s.name for s in steps]
        manifest = {
            "expected_commands": expected_commands,
            "actual_commands": actual_commands,
            "commands_executed": len(actual_commands),
            "commands_expected": len(expected_commands),
            "all_executed": expected_commands == actual_commands,
            "missing_commands": [
                c for c in expected_commands if c not in actual_commands
            ],
            "failure_reason": failure_reason,
            "steps": [
                {
                    "name": s.name,
                    "ok": s.ok,
                    "returncode": s.returncode,
                    "duration_seconds": s.duration_seconds,
                }
                for s in steps
            ],
        }
        manifest_path = log_dir / "validation_complete.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

    def _build_spec_env(
        self,
        context: ValidationContext,
        run_id: str,
    ) -> dict[str, str]:
        """Build environment for spec-based validation."""
        return {
            **os.environ,
            "LOCK_DIR": str(get_lock_dir()),
            "AGENT_ID": f"validator-{context.issue_id or run_id}",
        }

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

    def _run_spec_e2e(
        self,
        config: E2EConfig,
        env: dict[str, str],
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
        return runner.run(env=env, cwd=cwd)
