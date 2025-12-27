"""Standardized subprocess execution with timeout and process-group termination.

This module provides CommandRunner for consistent subprocess handling across:
- quality_gate.py
- validation/legacy_runner.py
- validation/spec_runner.py
- validation/e2e.py

Key features:
- Unified timeout handling with process-group termination
- Both sync and async execution
- Standardized output capture and tailing
- Consistent error handling
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .helpers import tail

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

# Exit code used for timeout (matches GNU coreutils timeout command)
TIMEOUT_EXIT_CODE = 124

# Default grace period before SIGKILL after SIGTERM (seconds)
DEFAULT_KILL_GRACE_SECONDS = 2.0


@dataclass
class CommandResult:
    """Result of a command execution.

    Provides structured access to command output with helper methods
    for truncating long output.

    Attributes:
        command: The command that was executed.
        returncode: Exit code of the command.
        stdout: Full stdout output.
        stderr: Full stderr output.
        duration_seconds: How long the command took.
        timed_out: Whether the command was killed due to timeout.
    """

    command: list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        """Whether the command succeeded (returncode == 0)."""
        return self.returncode == 0

    def stdout_tail(self, max_chars: int = 800, max_lines: int = 20) -> str:
        """Get truncated stdout.

        Args:
            max_chars: Maximum number of characters.
            max_lines: Maximum number of lines.

        Returns:
            Truncated stdout string.
        """
        return tail(self.stdout, max_chars=max_chars, max_lines=max_lines)

    def stderr_tail(self, max_chars: int = 800, max_lines: int = 20) -> str:
        """Get truncated stderr.

        Args:
            max_chars: Maximum number of characters.
            max_lines: Maximum number of lines.

        Returns:
            Truncated stderr string.
        """
        return tail(self.stderr, max_chars=max_chars, max_lines=max_lines)


class CommandRunner:
    """Runs commands with standardized timeout and process-group handling.

    This class provides both sync and async execution methods with:
    - Configurable timeout
    - Process-group termination (kills child processes on timeout)
    - Consistent output capture
    - Optional environment variable merging

    Example:
        runner = CommandRunner(cwd=Path("/my/repo"), timeout_seconds=60.0)
        result = runner.run(["pytest", "-v"])
        if not result.ok:
            print(f"Failed: {result.stderr_tail()}")
    """

    def __init__(
        self,
        cwd: Path,
        timeout_seconds: float | None = None,
        kill_grace_seconds: float = DEFAULT_KILL_GRACE_SECONDS,
    ):
        """Initialize CommandRunner.

        Args:
            cwd: Working directory for commands.
            timeout_seconds: Default timeout for commands. None means no timeout.
            kill_grace_seconds: Grace period after SIGTERM before SIGKILL.
        """
        self.cwd = cwd
        self.timeout_seconds = timeout_seconds
        self.kill_grace_seconds = kill_grace_seconds

    def run(
        self,
        cmd: list[str],
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        """Run a command synchronously.

        Args:
            cmd: Command and arguments to run.
            env: Environment variables (merged with os.environ).
            timeout: Override default timeout for this command.

        Returns:
            CommandResult with execution details.
        """
        effective_timeout = timeout if timeout is not None else self.timeout_seconds
        merged_env = self._merge_env(env)

        # Use process group on Unix for proper child termination
        use_process_group = sys.platform != "win32"

        start = time.monotonic()
        try:
            result = subprocess.run(
                cmd,
                cwd=self.cwd,
                env=merged_env,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                start_new_session=use_process_group,
            )
            duration = time.monotonic() - start
            return CommandResult(
                command=cmd,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                duration_seconds=duration,
                timed_out=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - start
            # subprocess.run handles termination, but we need to decode output
            stdout = self._decode_output(exc.stdout)
            stderr = self._decode_output(exc.stderr)
            return CommandResult(
                command=cmd,
                returncode=TIMEOUT_EXIT_CODE,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
                timed_out=True,
            )

    async def run_async(
        self,
        cmd: list[str],
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        """Run a command asynchronously.

        Uses asyncio subprocess for proper async timeout handling with
        process-group termination.

        Args:
            cmd: Command and arguments to run.
            env: Environment variables (merged with os.environ).
            timeout: Override default timeout for this command.

        Returns:
            CommandResult with execution details.
        """
        effective_timeout = timeout if timeout is not None else self.timeout_seconds
        merged_env = self._merge_env(env)

        # Use process group on Unix for proper child termination
        use_process_group = sys.platform != "win32"

        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
                env=merged_env,
                start_new_session=use_process_group,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=effective_timeout,
                )
                duration = time.monotonic() - start
                return CommandResult(
                    command=cmd,
                    returncode=proc.returncode or 0,
                    stdout=stdout_bytes.decode() if stdout_bytes else "",
                    stderr=stderr_bytes.decode() if stderr_bytes else "",
                    duration_seconds=duration,
                    timed_out=False,
                )
            except TimeoutError:
                duration = time.monotonic() - start
                await self._terminate_process(proc, use_process_group)
                return CommandResult(
                    command=cmd,
                    returncode=TIMEOUT_EXIT_CODE,
                    stdout="",
                    stderr="",
                    duration_seconds=duration,
                    timed_out=True,
                )
        except Exception as e:
            duration = time.monotonic() - start
            return CommandResult(
                command=cmd,
                returncode=1,
                stdout="",
                stderr=str(e),
                duration_seconds=duration,
                timed_out=False,
            )

    async def _terminate_process(
        self,
        proc: asyncio.subprocess.Process,
        use_process_group: bool,
    ) -> None:
        """Terminate a process and all its children.

        Sends SIGTERM to the process group, waits for grace period,
        then sends SIGKILL if still running.

        Args:
            proc: The process to terminate.
            use_process_group: Whether to kill the entire process group.
        """
        if proc.returncode is not None:
            return

        pgid = proc.pid if use_process_group else None

        try:
            # Send SIGTERM to process group
            if pgid is not None:
                try:
                    os.killpg(pgid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
            else:
                proc.terminate()

            # Wait for grace period
            try:
                await asyncio.wait_for(proc.wait(), timeout=self.kill_grace_seconds)
            except TimeoutError:
                pass

            # Force kill if still running
            if pgid is not None:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            elif proc.returncode is None:
                proc.kill()

            # Ensure process is fully reaped
            if proc.returncode is None:
                await proc.wait()
        except ProcessLookupError:
            pass

    def _merge_env(self, env: Mapping[str, str] | None) -> dict[str, str]:
        """Merge provided env with os.environ.

        Args:
            env: Environment variables to merge.

        Returns:
            Merged environment dictionary.
        """
        if env is None:
            return dict(os.environ)
        return {**os.environ, **env}

    def _decode_output(self, data: bytes | str | None) -> str:
        """Decode subprocess output which may be bytes, str, or None.

        Args:
            data: Output data from subprocess.

        Returns:
            Decoded string.
        """
        if data is None:
            return ""
        if isinstance(data, str):
            return data
        return data.decode()


def run_command(
    cmd: list[str],
    cwd: Path,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float | None = None,
) -> CommandResult:
    """Convenience function for running a single command.

    Args:
        cmd: Command and arguments to run.
        cwd: Working directory.
        env: Environment variables (merged with os.environ).
        timeout_seconds: Timeout for the command.

    Returns:
        CommandResult with execution details.
    """
    runner = CommandRunner(cwd=cwd, timeout_seconds=timeout_seconds)
    return runner.run(cmd, env=env)


async def run_command_async(
    cmd: list[str],
    cwd: Path,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float | None = None,
) -> CommandResult:
    """Convenience function for running a single command asynchronously.

    Args:
        cmd: Command and arguments to run.
        cwd: Working directory.
        env: Environment variables (merged with os.environ).
        timeout_seconds: Timeout for the command.

    Returns:
        CommandResult with execution details.
    """
    runner = CommandRunner(cwd=cwd, timeout_seconds=timeout_seconds)
    return await runner.run_async(cmd, env=env)
