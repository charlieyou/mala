"""Command executor for spec-based validation.

This module provides SpecCommandExecutor which encapsulates command execution
and lint-cache handling. It is separated from SpecValidationRunner to:
- Isolate execution logic from orchestration
- Enable unit testing with fake CommandRunners
- Keep lint-cache skipping behavior in one place

The executor receives commands and environment, executes them with lint-cache
awareness, and returns structured results without knowledge of the overall
validation pipeline.
"""

from __future__ import annotations

import os
import shlex
import signal
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .helpers import format_step_output
from .lint_cache import LintCache
from .result import ValidationStepResult
from .spec import CommandKind

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from src.core.protocols.events import MalaEventSink
    from src.core.protocols.infra import CommandRunnerPort, EnvConfigPort

    from .spec import ValidationCommand


@dataclass(frozen=True, kw_only=True)
class ExecutorConfig:
    """Configuration for SpecCommandExecutor.

    Attributes:
        enable_lint_cache: Whether to enable lint caching for cacheable commands.
        repo_path: Path to the main repo (for lint cache storage).
        step_timeout_seconds: Optional timeout for individual command execution.
        env_config: Environment configuration for paths (scripts, cache, etc.).
        command_runner: Command runner for executing commands.
        event_sink: Event sink for emitting validation step events (optional).
    """

    enable_lint_cache: bool = True
    repo_path: Path | None = None
    step_timeout_seconds: float | None = None
    env_config: EnvConfigPort
    command_runner: CommandRunnerPort
    event_sink: MalaEventSink | None = None


@dataclass
class ExecutorInput:
    """Input to SpecCommandExecutor for a single execution batch.

    Attributes:
        commands: List of commands to execute.
        cwd: Working directory for command execution.
        env: Environment variables for commands.
        log_dir: Directory for writing step logs.
    """

    commands: list[ValidationCommand]
    cwd: Path
    env: Mapping[str, str]
    log_dir: Path


@dataclass
class ExecutorOutput:
    """Output from SpecCommandExecutor after execution.

    Attributes:
        steps: List of step results for all commands (including skipped).
        failed: Whether a command failed (and allow_fail was False).
        failure_reason: Human-readable failure reason if failed.
    """

    steps: list[ValidationStepResult] = field(default_factory=list)
    failed: bool = False
    failure_reason: str | None = None


class ValidationInterrupted(Exception):
    """Raised when a validation command is interrupted by SIGINT."""

    def __init__(
        self, step: ValidationStepResult, steps: list[ValidationStepResult]
    ) -> None:
        self.step = step
        self.steps = steps
        super().__init__(f"Validation interrupted (exit {step.returncode})")


class SpecCommandExecutor:
    """Executes validation commands with lint-cache awareness.

    This executor handles:
    - Lint cache checking for FORMAT/LINT/TYPECHECK commands
    - Command execution via CommandRunner
    - Test mutex wrapping when requested
    - Step logging to files
    - Failure detection and early exit

    The executor is stateless per execution batch - it does not retain
    state between execute() calls. The lint cache itself persists to disk.

    Example:
        executor = SpecCommandExecutor(config)
        output = executor.execute(input)
        if output.failed:
            print(f"Failed: {output.failure_reason}")
    """

    # Command kinds eligible for lint caching
    CACHEABLE_KINDS = frozenset(
        {CommandKind.LINT, CommandKind.FORMAT, CommandKind.TYPECHECK}
    )

    def __init__(self, config: ExecutorConfig) -> None:
        """Initialize the command executor.

        Args:
            config: Executor configuration.
        """
        self.config = config

    def execute(self, input: ExecutorInput) -> ExecutorOutput:
        """Execute all commands in the input.

        Iterates through commands, checking lint cache for cacheable commands,
        executing non-cached commands, and stopping on first failure (unless
        allow_fail is set).

        Args:
            input: ExecutorInput with commands and context.

        Returns:
            ExecutorOutput with step results and failure info.
        """
        output = ExecutorOutput()

        # Initialize lint cache if enabled
        lint_cache = self._create_lint_cache(input.cwd)

        for i, cmd in enumerate(input.commands):
            # Check if this command can be skipped via cache
            if self._should_skip_cached(cmd, lint_cache):
                step = self._create_skipped_step(cmd)
                output.steps.append(step)
                continue

            # Write start marker for debugging
            self._write_start_marker(input.log_dir, i, cmd, input.cwd)

            # Emit event sink notification for command start
            if self.config.event_sink is not None:
                self.config.event_sink.on_validation_step_running(cmd.name)

            # Execute the command
            step = self._run_command(cmd, input.cwd, input.env)
            output.steps.append(step)

            # Write step logs to files
            self._write_step_logs(step, input.log_dir)

            if self._is_sigint_returncode(step.returncode):
                self._log_failure(cmd, step)
                raise ValidationInterrupted(step, list(output.steps))

            # Log result to terminal and update cache
            if step.ok:
                self._log_success(cmd, step, lint_cache)
            else:
                self._log_failure(cmd, step)
                if not cmd.allow_fail:
                    output.failed = True
                    output.failure_reason = self._format_failure_reason(cmd, step)
                    break

        return output

    def _is_sigint_returncode(self, returncode: int) -> bool:
        """Return True when a command exited due to SIGINT."""
        return returncode in (-signal.SIGINT, 128 + signal.SIGINT)

    def _create_lint_cache(self, cwd: Path) -> LintCache | None:
        """Create lint cache if enabled and paths available.

        Args:
            cwd: Working directory (passed to LintCache for git operations).

        Returns:
            LintCache if enabled and paths available, None otherwise.

        Note:
            The lint cache is created fresh for each execution batch to ensure
            correct git state detection, since the same batch may be
            validated in different worktrees.
        """
        if not self.config.enable_lint_cache:
            return None
        if self.config.repo_path is None:
            return None
        cache_dir = self.config.env_config.cache_dir
        return LintCache(
            cache_dir=cache_dir,
            repo_path=self.config.repo_path,
            command_runner=self.config.command_runner,
            git_cwd=cwd,
        )

    def _should_skip_cached(
        self,
        cmd: ValidationCommand,
        lint_cache: LintCache | None,
    ) -> bool:
        """Check if a command should be skipped due to cache hit.

        Args:
            cmd: The command to check.
            lint_cache: The lint cache (may be None if disabled).

        Returns:
            True if the command can be skipped.
        """
        if lint_cache is None:
            return False
        if cmd.kind not in self.CACHEABLE_KINDS:
            return False
        return lint_cache.should_skip(cmd.name)

    def _create_skipped_step(self, cmd: ValidationCommand) -> ValidationStepResult:
        """Create a synthetic step result for a skipped command.

        Args:
            cmd: The skipped command.

        Returns:
            ValidationStepResult indicating the command was skipped.
        """
        if self.config.event_sink is not None:
            self.config.event_sink.on_validation_step_skipped(
                cmd.name, "no changes since last check"
            )
        return ValidationStepResult(
            name=cmd.name,
            command=cmd.command,
            ok=True,
            returncode=0,
            stdout_tail="Skipped: no changes since last check",
            stderr_tail="",
            duration_seconds=0.0,
        )

    def _write_start_marker(
        self,
        log_dir: Path,
        index: int,
        cmd: ValidationCommand,
        cwd: Path,
    ) -> None:
        """Write a start marker file for debugging.

        Uses explicit flush() and fsync() to ensure the marker is written
        to disk immediately. This provides accurate debugging info if mala
        is interrupted mid-execution.

        Args:
            log_dir: Directory for logs.
            index: Command index in the batch.
            cmd: The command about to run.
            cwd: Working directory.
        """
        start_marker = log_dir / f"{index:02d}_{cmd.name.replace(' ', '_')}.started"
        self._write_file_flushed(start_marker, f"command: {cmd.command}\ncwd: {cwd}\n")

    def _run_command(
        self,
        cmd: ValidationCommand,
        cwd: Path,
        env: Mapping[str, str],
    ) -> ValidationStepResult:
        """Execute a single command.

        Args:
            cmd: The command to run.
            cwd: Working directory.
            env: Environment variables.

        Returns:
            ValidationStepResult with execution details.
        """
        # Use command's timeout if specified, else fall back to config timeout
        timeout = cmd.timeout or self.config.step_timeout_seconds

        # Use injected runner (required)
        runner = self.config.command_runner

        if cmd.shell:
            # Shell mode: pass command string directly with shell=True
            # For mutex wrapping in shell mode, prepend the script path
            if cmd.use_test_mutex:
                scripts_dir = self.config.env_config.scripts_dir
                full_cmd = f"{scripts_dir / 'test-mutex.sh'} {cmd.command}"
            else:
                full_cmd = cmd.command
            result = runner.run(full_cmd, env=env, shell=True, cwd=cwd, timeout=timeout)
        else:
            # Non-shell mode: split command and run as list (legacy behavior)
            cmd_list = shlex.split(cmd.command)
            full_cmd = (
                self._wrap_with_mutex(cmd_list) if cmd.use_test_mutex else cmd_list
            )
            result = runner.run(full_cmd, env=env, cwd=cwd, timeout=timeout)

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
        """Wrap a command with the test mutex script.

        Args:
            cmd: The command to wrap as a list.

        Returns:
            Command prefixed with test-mutex.sh.
        """
        scripts_dir = self.config.env_config.scripts_dir
        return [str(scripts_dir / "test-mutex.sh"), *cmd]

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

    def _write_step_logs(self, step: ValidationStepResult, log_dir: Path) -> None:
        """Write step stdout/stderr to log files.

        Uses explicit flush() and fsync() to ensure logs are written to disk
        immediately. This prevents log data loss if mala is interrupted.

        Args:
            step: The step result to log.
            log_dir: Directory to write logs to.
        """
        # Sanitize step name for filename
        safe_name = step.name.replace(" ", "_").replace("/", "_")

        if step.stdout_tail:
            stdout_path = log_dir / f"{safe_name}.stdout.log"
            self._write_file_flushed(stdout_path, step.stdout_tail)

        if step.stderr_tail:
            stderr_path = log_dir / f"{safe_name}.stderr.log"
            self._write_file_flushed(stderr_path, step.stderr_tail)

    def _log_success(
        self,
        cmd: ValidationCommand,
        step: ValidationStepResult,
        lint_cache: LintCache | None,
    ) -> None:
        """Log successful step and update lint cache.

        Args:
            cmd: The command that succeeded.
            step: The step result.
            lint_cache: The lint cache to update (may be None).
        """
        # Emit event sink notification
        if self.config.event_sink is not None:
            self.config.event_sink.on_validation_step_passed(
                cmd.name, step.duration_seconds
            )

        # Mark command as passed in cache for cacheable commands
        if lint_cache is not None and cmd.kind in self.CACHEABLE_KINDS:
            lint_cache.mark_passed(cmd.name)

    def _log_failure(
        self,
        cmd: ValidationCommand,
        step: ValidationStepResult,
    ) -> None:
        """Log failed step.

        Args:
            cmd: The command that failed.
            step: The step result.
        """
        # Emit event sink notification
        if self.config.event_sink is not None:
            self.config.event_sink.on_validation_step_failed(cmd.name, step.returncode)

    def _format_failure_reason(
        self,
        cmd: ValidationCommand,
        step: ValidationStepResult,
    ) -> str:
        """Format a human-readable failure reason.

        Args:
            cmd: The command that failed.
            step: The step result.

        Returns:
            Formatted failure reason string.
        """
        reason = f"{cmd.name} failed (exit {step.returncode})"
        details = format_step_output(step.stdout_tail, step.stderr_tail)
        if details:
            reason = f"{reason}: {details}"
        return reason
