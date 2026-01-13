"""Spec-based validation runner for mala.

This module provides SpecValidationRunner which runs validation using
ValidationSpec + ValidationContext, the modern API for mala validation.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING

from .lint_cache import LintCache
from .spec import ValidationArtifacts
from .spec_executor import (
    ExecutorConfig,
    ExecutorInput,
    SpecCommandExecutor,
    ValidationInterrupted,
)
from .spec_result_builder import ResultBuilderInput, SpecResultBuilder
from .spec_workspace import (
    SetupError,
    cleanup_workspace,
    setup_workspace,
)
from .validation_gating import (
    should_invalidate_lint_cache,
    should_trigger_validation,
)

if TYPE_CHECKING:
    from pathlib import Path

    from src.core.protocols.events import MalaEventSink
    from src.core.protocols.infra import (
        CommandRunnerPort,
        EnvConfigPort,
        LockManagerPort,
    )

    from .result import ValidationStepResult
    from .spec import (
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
    - Scope-aware validation (per-session vs global)
    - Per-command mutex settings
    - Integrated worktree, coverage, and E2E handling
    - Artifact tracking
    - Lint caching to skip redundant lint commands
    """

    def __init__(
        self,
        repo_path: Path,
        env_config: EnvConfigPort,
        command_runner: CommandRunnerPort,
        lock_manager: LockManagerPort,
        step_timeout_seconds: float | None = None,
        enable_lint_cache: bool = True,
        event_sink: MalaEventSink | None = None,
    ):
        """Initialize the spec validation runner.

        Args:
            repo_path: Path to the repository to validate.
            env_config: Environment configuration for paths.
            command_runner: Command runner for executing commands.
            lock_manager: Lock manager for file locking.
            step_timeout_seconds: Optional timeout for individual steps.
            enable_lint_cache: Whether to enable lint caching. Set to False
                in tests or when caching is not desired.
            event_sink: Event sink for emitting validation step events.
        """
        self.repo_path = repo_path.resolve()
        self.step_timeout_seconds = step_timeout_seconds
        self.enable_lint_cache = enable_lint_cache
        self.env_config = env_config
        self.command_runner = command_runner
        self.lock_manager = lock_manager
        self.event_sink = event_sink

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

        Uses a pipeline pattern:
        1. Check validation gating (skip if no code changes match patterns)
        2. Invalidate caches if config/setup files changed
        3. setup_workspace -> run_commands -> check_coverage -> run_e2e -> build_result

        Delegates workspace/baseline/worktree setup to spec_workspace module.
        """
        # Step 0: Check validation gating based on changed_files and code_patterns
        # Skip validation if no files match code_patterns (unless patterns empty)
        if context.changed_files and not should_trigger_validation(
            context.changed_files, spec
        ):
            # No matching code changes - skip validation (pass without running)
            artifacts = ValidationArtifacts(log_dir=log_dir) if log_dir else None
            return ValidationResult(
                passed=True,
                steps=[],
                failure_reasons=[],
                artifacts=artifacts,
            )

        # Note: setup commands always run fresh (not cached per CACHEABLE_KINDS in
        # spec_executor.py). The should_invalidate_setup_cache() function in
        # validation_gating.py exists for future setup caching support but is not
        # currently wired up.

        # Delegate workspace setup to spec_workspace module
        runner: CommandRunnerPort = self.command_runner

        try:
            # Step 0b: Invalidate lint cache if config_files changed
            # Done after runner is available so cache invalidation works
            if context.changed_files and should_invalidate_lint_cache(
                context.changed_files, spec
            ):
                self._invalidate_lint_cache_for_config_change(runner)

            workspace = setup_workspace(
                spec=spec,
                context=context,
                log_dir=log_dir,
                step_timeout_seconds=self.step_timeout_seconds,
                command_runner=runner,
                env_config=self.env_config,
                lock_manager=self.lock_manager,
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
                runner,
            )
            return result
        finally:
            # Clean up workspace with correct pass/fail status
            # On exception, result is None so we treat as failed (validation_passed=False)
            validation_passed = result.passed if result is not None else False
            cleanup_workspace(workspace, validation_passed, runner)

    def _invalidate_lint_cache_for_config_change(
        self, command_runner: CommandRunnerPort
    ) -> None:
        """Invalidate lint cache when config files change.

        Called when files matching config_files patterns are detected in
        the changed files. This ensures lint/format/typecheck commands
        run fresh when their configuration changes.

        Args:
            command_runner: The command runner to use for cache operations.
        """
        if not self.enable_lint_cache:
            return
        try:
            cache_dir = self.env_config.cache_dir
            cache = LintCache(
                cache_dir=cache_dir,
                repo_path=self.repo_path,
                command_runner=command_runner,
            )
            cache.invalidate_all()
        except Exception:
            # If cache invalidation fails, continue anyway
            # The commands will just run without cache benefit
            pass

    def _run_validation_pipeline(
        self,
        spec: ValidationSpec,
        context: ValidationContext,
        cwd: Path,
        artifacts: ValidationArtifacts,
        log_dir: Path,
        run_id: str,
        baseline_percent: float | None,
        command_runner: CommandRunnerPort,
    ) -> ValidationResult:
        """Run pipeline: commands -> coverage -> e2e -> result."""
        env = self._build_spec_env(context, run_id)
        expected = [cmd.name for cmd in spec.commands]
        self._write_initial_manifest(log_dir, expected, cwd, run_id, context, spec)

        # Step 1: Run commands
        try:
            steps = self._run_commands(spec, cwd, env, log_dir, command_runner)
        except CommandFailure as e:
            self._write_completion_manifest(log_dir, expected, e.steps, e.reason)
            return ValidationResult(
                passed=False,
                steps=e.steps,
                failure_reasons=[e.reason],
                artifacts=artifacts,
            )
        except ValidationInterrupted as e:
            self._write_completion_manifest(
                log_dir, expected, e.steps, "Interrupted by SIGINT"
            )
            raise

        # Step 2: Build result (coverage check, E2E, result assembly)
        builder = SpecResultBuilder()
        builder_input = ResultBuilderInput(
            spec=spec,
            context=context,
            steps=steps,
            artifacts=artifacts,
            cwd=cwd,
            log_dir=log_dir,
            env=env,
            baseline_percent=baseline_percent,
            env_config=self.env_config,
            command_runner=command_runner,
            yaml_coverage_config=spec.yaml_coverage_config,
            event_sink=self.event_sink,
        )
        result = builder.build(builder_input)

        # Write completion manifest
        failure_reason = result.failure_reasons[0] if result.failure_reasons else None
        self._write_completion_manifest(log_dir, expected, steps, failure_reason)

        return result

    def _write_file_flushed(self, path: Path, content: str) -> None:
        """Write content to a file with immediate flush to disk.

        Uses explicit flush() and fsync() to ensure data is persisted
        before returning. This prevents log data loss if mala is interrupted.

        Args:
            path: Path to write to.
            content: Text content to write.
        """
        with open(path, "w") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())

    def _write_initial_manifest(
        self,
        log_dir: Path,
        expected_commands: list[str],
        cwd: Path,
        run_id: str,
        context: ValidationContext,
        spec: ValidationSpec,
    ) -> None:
        """Write initial manifest of expected commands for debugging.

        Uses explicit flush() and fsync() to ensure the manifest is written
        to disk immediately. This provides accurate debugging info if mala
        is interrupted mid-validation.
        """
        manifest_path = log_dir / "validation_manifest.json"
        self._write_file_flushed(
            manifest_path,
            json.dumps(
                {
                    "expected_commands": expected_commands,
                    "cwd": str(cwd),
                    "run_id": run_id,
                    "issue_id": context.issue_id,
                    "scope": spec.scope.value,
                },
                indent=2,
            ),
        )

    def _run_commands(
        self,
        spec: ValidationSpec,
        cwd: Path,
        env: dict[str, str],
        log_dir: Path,
        command_runner: CommandRunnerPort,
    ) -> list[ValidationStepResult]:
        """Execute all commands in the spec.

        Delegates to SpecCommandExecutor for command execution and lint-cache
        handling. The executor encapsulates all execution logic.

        Args:
            spec: Validation spec with commands.
            cwd: Working directory for commands.
            env: Environment variables.
            log_dir: Directory for logs.
            command_runner: Command runner for executing commands.

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
            env_config=self.env_config,
            command_runner=command_runner,
            event_sink=self.event_sink,
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

    def _write_completion_manifest(
        self,
        log_dir: Path,
        expected_commands: list[str],
        steps: list[ValidationStepResult],
        failure_reason: str | None,
    ) -> None:
        """Write completion manifest with expected vs actual commands.

        Uses explicit flush() and fsync() to ensure the manifest is written
        to disk immediately. This helps debug cases where commands are
        unexpectedly skipped and prevents data loss if mala is interrupted.
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
        self._write_file_flushed(manifest_path, json.dumps(manifest, indent=2))

    def _build_spec_env(
        self,
        context: ValidationContext,
        run_id: str,
    ) -> dict[str, str]:
        """Build environment for spec-based validation."""
        lock_dir = str(self.env_config.lock_dir)
        return {
            **os.environ,
            "LOCK_DIR": lock_dir,
            "AGENT_ID": f"validator-{context.issue_id or run_id}",
        }
