"""Spec-based validation runner for mala.

This module provides SpecValidationRunner which runs validation using
ValidationSpec + ValidationContext, the modern API for mala validation.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from ..tools.env import LOCK_DIR, SCRIPTS_DIR
from .coverage import CoverageResult, CoverageStatus, parse_and_check_coverage
from .deps_sync import DepsSyncState
from .e2e import E2EConfig as E2ERunnerConfig
from .e2e import E2ERunner, E2EStatus
from .helpers import decode_timeout_output, format_step_output, tail
from .spec import CommandKind, ValidationArtifacts
from .worktree import (
    WorktreeConfig,
    WorktreeState,
    create_worktree,
    remove_worktree,
)

if TYPE_CHECKING:
    from .e2e import E2EResult
    from .spec import (
        CoverageConfig,
        E2EConfig,
        ValidationCommand,
        ValidationContext,
        ValidationSpec,
    )
    from .worktree import WorktreeContext

from .result import ValidationResult, ValidationStepResult


class SpecValidationRunner:
    """Runs validation according to a ValidationSpec.

    This runner supports:
    - Scope-aware validation (per-issue vs run-level)
    - Per-command mutex settings
    - Integrated worktree, coverage, and E2E handling
    - Artifact tracking

    This is the modern API that replaces the legacy ValidationConfig approach.
    """

    def __init__(
        self,
        repo_path: Path,
        step_timeout_seconds: float | None = None,
    ):
        """Initialize the spec validation runner.

        Args:
            repo_path: Path to the repository to validate.
            step_timeout_seconds: Optional timeout for individual steps.
        """
        self.repo_path = repo_path.resolve()
        self.step_timeout_seconds = step_timeout_seconds

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
        """Synchronous implementation of run_spec."""
        # Set up log directory
        if log_dir is None:
            log_dir = Path(tempfile.mkdtemp(prefix="mala-validation-logs-"))

        log_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique run ID
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        issue_id = context.issue_id or "run-level"

        # Initialize artifacts
        artifacts = ValidationArtifacts(log_dir=log_dir)

        # Set up worktree if we have a commit to validate
        worktree_ctx: WorktreeContext | None = None
        validation_cwd: Path

        if context.commit_hash:
            worktree_config = WorktreeConfig(
                base_dir=log_dir / "worktrees",
                keep_on_failure=True,  # Keep for debugging
            )
            worktree_ctx = create_worktree(
                repo_path=context.repo_path,
                commit_sha=context.commit_hash,
                config=worktree_config,
                run_id=run_id,
                issue_id=issue_id,
                attempt=1,
            )

            if worktree_ctx.state == WorktreeState.FAILED:
                return ValidationResult(
                    passed=False,
                    failure_reasons=[f"Worktree creation failed: {worktree_ctx.error}"],
                    retriable=False,
                    artifacts=artifacts,
                )

            validation_cwd = worktree_ctx.path
            artifacts.worktree_path = worktree_ctx.path
        else:
            # No commit specified, validate in place
            validation_cwd = context.repo_path

        # Execute commands and capture result, ensuring worktree cleanup on exceptions
        result: ValidationResult | None = None
        try:
            result = self._execute_spec_commands(
                spec=spec,
                context=context,
                cwd=validation_cwd,
                artifacts=artifacts,
                log_dir=log_dir,
                run_id=run_id,
            )
            return result
        finally:
            # Clean up worktree with correct pass/fail status
            # On exception, result is None so we treat as failed (validation_passed=False)
            if worktree_ctx is not None:
                validation_passed = result.passed if result is not None else False
                worktree_ctx = remove_worktree(
                    worktree_ctx, validation_passed=validation_passed
                )
                if worktree_ctx.state == WorktreeState.KEPT:
                    artifacts.worktree_state = "kept"
                elif worktree_ctx.state == WorktreeState.REMOVED:
                    artifacts.worktree_state = "removed"

    def _execute_spec_commands(
        self,
        spec: ValidationSpec,
        context: ValidationContext,
        cwd: Path,
        artifacts: ValidationArtifacts,
        log_dir: Path,
        run_id: str,
    ) -> ValidationResult:
        """Execute all commands in the spec.

        Args:
            spec: Validation spec with commands.
            context: Validation context.
            cwd: Working directory for commands.
            artifacts: Artifacts to update.
            log_dir: Directory for logs.
            run_id: Unique run identifier.

        Returns:
            ValidationResult with all step results.
        """
        # Build environment
        env = self._build_spec_env(context, run_id)

        steps: list[ValidationStepResult] = []
        coverage_result: CoverageResult | None = None

        # Check if we need to run uv sync (for deps optimization)
        deps_state = DepsSyncState(cwd, force=spec.deps.force_sync)
        skip_deps_sync = spec.deps.skip_if_unchanged and not deps_state.needs_sync()

        # Execute each command in order
        for cmd in spec.commands:
            # Skip deps sync if dependencies unchanged
            if cmd.kind == CommandKind.DEPS and skip_deps_sync:
                # Record a skipped step for visibility
                steps.append(
                    ValidationStepResult(
                        name=f"{cmd.name} (skipped - deps unchanged)",
                        command=cmd.command,
                        ok=True,
                        returncode=0,
                        stdout_tail="Skipped: pyproject.toml/uv.lock unchanged",
                        duration_seconds=0.0,
                    )
                )
                continue

            step = self._run_spec_command(cmd, cwd, env, log_dir)
            steps.append(step)

            # Log output to files
            self._write_step_logs(step, log_dir)

            if not step.ok and not cmd.allow_fail:
                reason = f"{cmd.name} failed (exit {step.returncode})"
                details = format_step_output(step.stdout_tail, step.stderr_tail)
                if details:
                    reason = f"{reason}: {details}"
                return ValidationResult(
                    passed=False,
                    steps=steps,
                    failure_reasons=[reason],
                    artifacts=artifacts,
                )

            # Mark deps as synced after successful uv sync
            if cmd.kind == CommandKind.DEPS and step.ok:
                deps_state.mark_synced()

        # Handle coverage if enabled
        if spec.coverage.enabled:
            coverage_result = self._check_coverage(spec.coverage, cwd, log_dir)
            if coverage_result.report_path:
                artifacts.coverage_report = coverage_result.report_path

            # Fail on coverage error or threshold not met
            if not coverage_result.passed:
                return ValidationResult(
                    passed=False,
                    steps=steps,
                    failure_reasons=[
                        coverage_result.failure_reason or "Coverage check failed"
                    ],
                    artifacts=artifacts,
                    coverage_result=coverage_result,
                )

        # Handle E2E if enabled (only for run-level scope)
        from .spec import ValidationScope

        if spec.e2e.enabled and spec.scope == ValidationScope.RUN_LEVEL:
            e2e_result = self._run_spec_e2e(spec.e2e, env, cwd, log_dir)
            if e2e_result.fixture_path:
                artifacts.e2e_fixture_path = e2e_result.fixture_path

            if not e2e_result.passed and e2e_result.status != E2EStatus.SKIPPED:
                return ValidationResult(
                    passed=False,
                    steps=steps,
                    failure_reasons=[e2e_result.failure_reason or "E2E failed"],
                    artifacts=artifacts,
                    coverage_result=coverage_result,
                )

        return ValidationResult(
            passed=True,
            steps=steps,
            artifacts=artifacts,
            coverage_result=coverage_result,
        )

    def _build_spec_env(
        self,
        context: ValidationContext,
        run_id: str,
    ) -> dict[str, str]:
        """Build environment for spec-based validation."""
        return {
            **os.environ,
            "LOCK_DIR": str(LOCK_DIR),
            "AGENT_ID": f"validator-{context.issue_id or run_id}",
        }

    def _run_spec_command(
        self,
        cmd: ValidationCommand,
        cwd: Path,
        env: dict[str, str],
        log_dir: Path,
    ) -> ValidationStepResult:
        """Run a single ValidationCommand.

        Args:
            cmd: The command to run.
            cwd: Working directory.
            env: Environment variables.
            log_dir: Directory for logs.

        Returns:
            ValidationStepResult with execution details.
        """
        # Wrap with mutex if requested
        full_cmd = (
            self._wrap_with_mutex(cmd.command) if cmd.use_test_mutex else cmd.command
        )

        start = time.monotonic()
        try:
            result = subprocess.run(
                full_cmd,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.step_timeout_seconds,
            )
            duration = time.monotonic() - start
            return ValidationStepResult(
                name=cmd.name,
                command=cmd.command,
                ok=result.returncode == 0,
                returncode=result.returncode,
                stdout_tail=tail(result.stdout),
                stderr_tail=tail(result.stderr),
                duration_seconds=duration,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - start
            return ValidationStepResult(
                name=cmd.name,
                command=cmd.command,
                ok=False,
                returncode=124,
                stdout_tail=decode_timeout_output(exc.stdout),
                stderr_tail=decode_timeout_output(exc.stderr),
                duration_seconds=duration,
            )

    def _wrap_with_mutex(self, cmd: list[str]) -> list[str]:
        """Wrap a command with the test mutex script."""
        return [str(SCRIPTS_DIR / "test-mutex.sh"), *cmd]

    def _write_step_logs(self, step: ValidationStepResult, log_dir: Path) -> None:
        """Write step stdout/stderr to log files.

        Args:
            step: The step result to log.
            log_dir: Directory to write logs to.
        """
        # Sanitize step name for filename
        safe_name = step.name.replace(" ", "_").replace("/", "_")

        if step.stdout_tail:
            stdout_path = log_dir / f"{safe_name}.stdout.log"
            stdout_path.write_text(step.stdout_tail)

        if step.stderr_tail:
            stderr_path = log_dir / f"{safe_name}.stderr.log"
            stderr_path.write_text(step.stderr_tail)

    def _check_coverage(
        self,
        config: CoverageConfig,
        cwd: Path,
        log_dir: Path,
    ) -> CoverageResult:
        """Check coverage against threshold.

        Args:
            config: Coverage configuration.
            cwd: Working directory where coverage.xml should be.
            log_dir: Directory for logs.

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

        return parse_and_check_coverage(report_path, config.min_percent)

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
