"""Post-commit validation runner for mala.

DEPRECATION NOTICE:
    ValidationRunner is a legacy facade for backwards compatibility only.
    New code should use SpecValidationRunner directly with ValidationSpec.

This module provides:
- ValidationRunner: DEPRECATED facade (wraps both legacy and spec runners)
- LegacyValidationRunner: Legacy API using ValidationConfig (re-exported)
- SpecValidationRunner: Modern API using ValidationSpec (re-exported)

Migration guide:
    # Old (deprecated):
    runner = ValidationRunner(repo_path, ValidationConfig(...))
    result = await runner.run_spec(spec, context)

    # New (recommended):
    runner = SpecValidationRunner(repo_path)
    result = await runner.run_spec(spec, context)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Re-export result types
from .result import ValidationResult, ValidationStepResult

# Re-export legacy config
from .legacy_runner import LegacyValidationRunner, ValidationConfig

# Re-export spec runner
from .spec_runner import SpecValidationRunner

if TYPE_CHECKING:
    from pathlib import Path

    from .e2e import E2EResult
    from .spec import E2EConfig, ValidationContext, ValidationSpec

# Silence unused import warnings for re-exports
__all__ = [
    "LegacyValidationRunner",
    "SpecValidationRunner",
    "ValidationConfig",
    "ValidationResult",
    "ValidationRunner",
    "ValidationStepResult",
]


class ValidationRunner:
    """DEPRECATED: Facade for post-commit validation runners.

    .. deprecated::
        Use SpecValidationRunner directly for spec-based validation.
        This class exists only for backwards compatibility with code
        that uses the legacy validate_commit() API.

    This class delegates to:
    - LegacyValidationRunner for validate_commit() using ValidationConfig
    - SpecValidationRunner for run_spec() using ValidationSpec + ValidationContext

    For new code, use SpecValidationRunner directly:

        # Recommended approach:
        runner = SpecValidationRunner(repo_path)
        result = await runner.run_spec(spec, context)

    The validate_commit() method is the only remaining use case for this class.
    If you're only using run_spec(), switch to SpecValidationRunner.
    """

    def __init__(self, repo_path: Path, config: ValidationConfig):
        """Initialize the validation runner facade.

        .. deprecated::
            For spec-based validation, use SpecValidationRunner directly.
            This constructor is only needed for legacy validate_commit() usage.

        Args:
            repo_path: Path to the repository to validate.
            config: Legacy configuration for validation.
        """
        self.repo_path = repo_path.resolve()
        self.config = config
        self._legacy_runner = LegacyValidationRunner(repo_path, config)
        self._spec_runner = SpecValidationRunner(
            repo_path, step_timeout_seconds=config.step_timeout_seconds
        )

    async def validate_commit(self, commit_sha: str, label: str) -> ValidationResult:
        """Validate a commit in a clean worktree (legacy API).

        Delegates to LegacyValidationRunner.

        Args:
            commit_sha: Git commit hash to validate.
            label: Label for the validation run.

        Returns:
            ValidationResult with steps and pass/fail status.
        """
        return await self._legacy_runner.validate_commit(commit_sha, label)

    async def run_spec(
        self,
        spec: ValidationSpec,
        context: ValidationContext,
        log_dir: Path | None = None,
    ) -> ValidationResult:
        """Run validation according to a ValidationSpec.

        Delegates to SpecValidationRunner.

        Args:
            spec: What validations to run.
            context: Immutable context for the validation run.
            log_dir: Directory for logs/artifacts. Uses temp dir if None.

        Returns:
            ValidationResult with steps, artifacts, and coverage info.
        """
        return await self._spec_runner.run_spec(spec, context, log_dir)

    # Expose internal methods for backwards compatibility with tests
    def _validate_commit_sync(self, commit_sha: str, label: str) -> ValidationResult:
        """Sync wrapper for validate_commit (for testing)."""
        return self._legacy_runner._validate_commit_sync(commit_sha, label)

    def _run_spec_sync(
        self,
        spec: ValidationSpec,
        context: ValidationContext,
        log_dir: Path | None = None,
    ) -> ValidationResult:
        """Sync wrapper for run_spec (for testing)."""
        return self._spec_runner._run_spec_sync(spec, context, log_dir)

    def _build_validation_commands(self) -> list[tuple[str, list[str]]]:
        """Build validation commands (for testing)."""
        return self._legacy_runner._build_validation_commands()

    def _run_command(
        self, name: str, cmd: list[str], cwd: Path, env: dict[str, str]
    ) -> ValidationStepResult:
        """Run a single command (for testing)."""
        from .command_runner import CommandRunner

        runner = CommandRunner(
            cwd=cwd, timeout_seconds=self.config.step_timeout_seconds
        )
        return self._legacy_runner._run_command(name, cmd, runner, env)

    def _run_validation_steps(
        self, cwd: Path, label: str, include_e2e: bool
    ) -> ValidationResult:
        """Run validation steps (for testing)."""
        return self._legacy_runner._run_validation_steps(cwd, label, include_e2e)

    def _run_e2e(self, label: str, env: dict[str, str]) -> ValidationResult:
        """Run E2E validation (for testing)."""
        return self._legacy_runner._run_e2e(label, env)

    def _wrap_with_mutex(self, cmd: list[str]) -> list[str]:
        """Wrap command with mutex (for testing)."""
        return self._legacy_runner._wrap_with_mutex(cmd)

    def _build_spec_env(
        self, context: ValidationContext, run_id: str
    ) -> dict[str, str]:
        """Build environment for spec validation (for testing)."""
        return self._spec_runner._build_spec_env(context, run_id)

    def _run_spec_e2e(
        self,
        config: E2EConfig,
        env: dict[str, str],
        cwd: Path,
        log_dir: Path,
    ) -> E2EResult:
        """Run spec E2E (for testing)."""
        return self._spec_runner._run_spec_e2e(config, env, cwd, log_dir)
