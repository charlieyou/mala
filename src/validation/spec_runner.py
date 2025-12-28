"""Spec-based validation runner for mala.

This module provides SpecValidationRunner which runs validation using
ValidationSpec + ValidationContext, the modern API for mala validation.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from ..tools.env import SCRIPTS_DIR, get_lock_dir
from ..tools.locking import try_lock, wait_for_lock
from .command_runner import CommandRunner
from .coverage import (
    CoverageResult,
    CoverageStatus,
    get_baseline_coverage,
    is_baseline_stale,
    parse_and_check_coverage,
)
from .e2e import E2EConfig as E2ERunnerConfig
from .e2e import E2ERunner, E2EStatus
from .helpers import format_step_output
from .spec import ValidationArtifacts
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

# Lock file path for baseline refresh coordination
_BASELINE_LOCK_FILE = "coverage-baseline.lock"


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
        """Synchronous implementation of run_spec.

        Uses a pipeline pattern: run_commands -> check_coverage -> run_e2e -> build_result.
        """
        # Set up log directory
        if log_dir is None:
            log_dir = Path(tempfile.mkdtemp(prefix="mala-validation-logs-"))

        log_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique run ID
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        issue_id = context.issue_id or "run-level"

        # Initialize artifacts
        artifacts = ValidationArtifacts(log_dir=log_dir)

        # Check/refresh baseline coverage BEFORE worktree creation
        # This captures baseline from main repo state
        baseline_percent: float | None = None
        if spec.coverage.enabled and spec.coverage.min_percent is None:
            # "No decrease" mode - need to get baseline
            try:
                baseline_percent = self._refresh_baseline(spec, context.repo_path)
            except RuntimeError as e:
                # Baseline refresh failed
                return ValidationResult(
                    passed=False,
                    failure_reasons=[str(e)],
                    retriable=False,
                    artifacts=artifacts,
                )

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

        # Execute pipeline and capture result, ensuring worktree cleanup
        result: ValidationResult | None = None
        try:
            result = self._run_validation_pipeline(
                spec,
                context,
                validation_cwd,
                artifacts,
                log_dir,
                run_id,
                baseline_percent,
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

    def _refresh_baseline(
        self,
        spec: ValidationSpec,
        repo_path: Path,
    ) -> float:
        """Refresh the baseline coverage if stale or missing.

        Uses file locking with double-check pattern to prevent concurrent
        agents from clobbering each other's baseline refresh.

        Args:
            spec: Validation spec with pytest command and coverage config.
            repo_path: Path to the main repository.

        Returns:
            The baseline coverage percentage.

        Raises:
            RuntimeError: If baseline refresh fails (lock timeout, worktree
                creation failure, pytest failure, no coverage generated, etc.).
        """
        # Determine baseline report path
        baseline_path = repo_path / "coverage.xml"

        # Check if baseline is fresh (no refresh needed)
        if not is_baseline_stale(baseline_path, repo_path):
            try:
                baseline = get_baseline_coverage(baseline_path)
                if baseline is not None:
                    return baseline
            except ValueError:
                # Malformed baseline - need to refresh
                pass

        # Baseline is stale or missing - try to acquire lock for refresh
        run_id = f"baseline-{uuid.uuid4().hex[:8]}"
        agent_id = f"baseline-refresh-{run_id}"
        repo_namespace = str(repo_path)

        # Try to acquire lock (non-blocking first)
        if not try_lock(_BASELINE_LOCK_FILE, agent_id, repo_namespace):
            # Another agent is refreshing - wait for them
            if not wait_for_lock(
                _BASELINE_LOCK_FILE,
                agent_id,
                repo_namespace,
                timeout_seconds=300.0,  # 5 min max wait
                poll_interval_ms=1000,
            ):
                raise RuntimeError("Timeout waiting for baseline refresh lock")

        # Lock acquired - double-check if still stale (another agent may have refreshed)
        try:
            if not is_baseline_stale(baseline_path, repo_path):
                try:
                    baseline = get_baseline_coverage(baseline_path)
                    if baseline is not None:
                        return baseline
                except ValueError:
                    pass  # Still need to refresh

            # Still stale - run refresh in temp worktree
            return self._run_baseline_refresh(spec, repo_path, baseline_path)
        finally:
            # Release lock by removing lock file
            from ..tools.locking import _lock_path

            lock_file = _lock_path(_BASELINE_LOCK_FILE, repo_namespace)
            lock_file.unlink(missing_ok=True)

    def _run_baseline_refresh(
        self,
        spec: ValidationSpec,
        repo_path: Path,
        baseline_path: Path,
    ) -> float:
        """Run pytest in temp worktree to refresh baseline coverage.

        Args:
            spec: Validation spec with pytest command.
            repo_path: Path to the main repository.
            baseline_path: Where to write the baseline coverage.xml.

        Returns:
            The new baseline coverage percentage.

        Raises:
            RuntimeError: If worktree creation, uv sync, pytest fails, or no
                coverage report is generated.
        """
        from .spec import CommandKind

        # Find the pytest command from spec
        pytest_commands = spec.commands_by_kind(CommandKind.TEST)
        if not pytest_commands:
            raise RuntimeError("No pytest command in spec for baseline refresh")

        # Create temp worktree at HEAD
        run_id = f"baseline-{uuid.uuid4().hex[:8]}"
        temp_dir = Path(tempfile.mkdtemp(prefix="mala-baseline-"))
        worktree_config = WorktreeConfig(
            base_dir=temp_dir,
            keep_on_failure=False,
        )

        worktree_ctx: WorktreeContext | None = None
        try:
            worktree_ctx = create_worktree(
                repo_path=repo_path,
                commit_sha="HEAD",
                config=worktree_config,
                run_id=run_id,
                issue_id="baseline",
                attempt=1,
            )

            if worktree_ctx.state == WorktreeState.FAILED:
                raise RuntimeError(
                    f"Baseline worktree creation failed: {worktree_ctx.error}"
                )

            worktree_path = worktree_ctx.path

            # Build environment
            env = {
                **os.environ,
                "LOCK_DIR": str(get_lock_dir()),
                "AGENT_ID": f"baseline-{run_id}",
            }

            # Run uv sync first to install dependencies
            runner = CommandRunner(cwd=worktree_path, timeout_seconds=300.0)
            sync_result = runner.run(["uv", "sync", "--all-extras"], env=env)
            if sync_result.returncode != 0:
                raise RuntimeError(
                    f"uv sync failed during baseline refresh: {sync_result.stderr}"
                )

            # Get pytest command and modify for baseline (override threshold to 0)
            pytest_cmd = list(pytest_commands[0].command)

            # Replace any existing --cov-fail-under with 0
            # This ensures we capture baseline even if it's below pyproject.toml threshold
            new_pytest_cmd = []
            skip_next = False
            for arg in pytest_cmd:
                if skip_next:
                    skip_next = False
                    continue
                if arg.startswith("--cov-fail-under="):
                    continue
                if arg == "--cov-fail-under":
                    skip_next = True
                    continue
                new_pytest_cmd.append(arg)
            new_pytest_cmd.append("--cov-fail-under=0")

            # Run pytest
            result = runner.run(new_pytest_cmd, env=env)
            if result.returncode != 0:
                raise RuntimeError(
                    f"pytest failed during baseline refresh (exit {result.returncode})"
                )

            # Check for coverage.xml in worktree
            worktree_coverage = worktree_path / "coverage.xml"
            if not worktree_coverage.exists():
                raise RuntimeError("No coverage.xml generated during baseline refresh")

            # Atomic rename to main repo
            temp_coverage = baseline_path.with_suffix(".xml.tmp")
            shutil.copy2(worktree_coverage, temp_coverage)
            os.rename(temp_coverage, baseline_path)

            # Parse and return the coverage percentage
            try:
                baseline = get_baseline_coverage(baseline_path)
                if baseline is None:
                    raise RuntimeError(
                        "Baseline coverage.xml exists but has no coverage data"
                    )
                return baseline
            except ValueError as e:
                raise RuntimeError(f"Failed to parse baseline coverage: {e}") from e

        finally:
            # Clean up temp worktree
            if worktree_ctx is not None:
                remove_worktree(worktree_ctx, validation_passed=True)
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

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
        steps: list[ValidationStepResult] = []

        for i, cmd in enumerate(spec.commands):
            # Log command start for debugging
            start_marker = log_dir / f"{i:02d}_{cmd.name.replace(' ', '_')}.started"
            start_marker.write_text(f"command: {cmd.command}\ncwd: {cwd}\n")

            try:
                step = self._run_spec_command(cmd, cwd, env, log_dir)
            except Exception as e:
                # Log unexpected exceptions during command execution
                error_path = log_dir / f"{i:02d}_{cmd.name.replace(' ', '_')}.error"
                error_path.write_text(f"Exception: {type(e).__name__}: {e}\n")
                raise

            steps.append(step)

            # Log output to files
            self._write_step_logs(step, log_dir)

            if not step.ok and not cmd.allow_fail:
                reason = f"{cmd.name} failed (exit {step.returncode})"
                details = format_step_output(step.stdout_tail, step.stderr_tail)
                if details:
                    reason = f"{reason}: {details}"
                raise CommandFailure(steps, reason)

        return steps

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

        runner = CommandRunner(cwd=cwd, timeout_seconds=self.step_timeout_seconds)
        result = runner.run(full_cmd, env=env)
        return ValidationStepResult(
            name=cmd.name,
            command=cmd.command,
            ok=result.ok,
            returncode=result.returncode,
            stdout_tail=result.stdout_tail(),
            stderr_tail=result.stderr_tail(),
            duration_seconds=result.duration_seconds,
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
