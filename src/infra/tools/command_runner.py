"""Standardized subprocess execution with timeout and process-group termination.

This module provides CommandRunner for consistent subprocess handling across:
- src/beads_client.py
- evidence_check.py
- validation/legacy_runner.py
- validation/spec_runner.py
- validation/e2e.py
- validation/worktree.py

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

from src.core.protocols.infra import CommandRunnerPort

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

# Exit code used for timeout (matches GNU coreutils timeout command)
TIMEOUT_EXIT_CODE = 124

# Default grace period before SIGKILL after SIGTERM (seconds)
DEFAULT_KILL_GRACE_SECONDS = 2.0

# Track process groups to receive forwarded SIGINTs during long-running commands.
_SIGINT_FORWARD_PGIDS: set[int] = set()


def _register_sigint_pgid(pgid: int) -> None:
    _SIGINT_FORWARD_PGIDS.add(pgid)


def _unregister_sigint_pgid(pgid: int) -> None:
    _SIGINT_FORWARD_PGIDS.discard(pgid)


def _forward_sigint_to_process_groups() -> None:
    if sys.platform == "win32":
        return
    for pgid in list(_SIGINT_FORWARD_PGIDS):
        try:
            os.killpg(pgid, signal.SIGINT)
        except (ProcessLookupError, PermissionError):
            pass


def _tail(text: str, max_chars: int = 800, max_lines: int = 20) -> str:
    """Truncate text to last N lines and M characters.

    Args:
        text: The text to truncate.
        max_chars: Maximum number of characters to keep.
        max_lines: Maximum number of lines to keep.

    Returns:
        The truncated text.
    """
    if not text:
        return ""
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    clipped = "\n".join(lines)
    if len(clipped) > max_chars:
        return clipped[-max_chars:]
    return clipped


@dataclass
class CommandResult:
    """Result of a command execution.

    Provides structured access to command output with helper methods
    for truncating long output.

    Attributes:
        command: The command that was executed (list or shell string).
        returncode: Exit code of the command.
        stdout: Full stdout output.
        stderr: Full stderr output.
        duration_seconds: How long the command took.
        timed_out: Whether the command was killed due to timeout.
    """

    command: list[str] | str
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
        return _tail(self.stdout, max_chars=max_chars, max_lines=max_lines)

    def stderr_tail(self, max_chars: int = 800, max_lines: int = 20) -> str:
        """Get truncated stderr.

        Args:
            max_chars: Maximum number of characters.
            max_lines: Maximum number of lines.

        Returns:
            Truncated stderr string.
        """
        return _tail(self.stderr, max_chars=max_chars, max_lines=max_lines)


class CommandRunner(CommandRunnerPort):
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

    @staticmethod
    def forward_sigint() -> None:
        """Forward SIGINT to active process groups started by CommandRunner."""
        _forward_sigint_to_process_groups()

    @staticmethod
    def register_sigint_pgid(pgid: int) -> None:
        """Register an external process group for SIGINT forwarding."""
        _register_sigint_pgid(pgid)

    @staticmethod
    def unregister_sigint_pgid(pgid: int) -> None:
        """Unregister an external process group from SIGINT forwarding."""
        _unregister_sigint_pgid(pgid)

    @staticmethod
    def kill_active_process_groups() -> None:
        """Send SIGKILL to all tracked process groups.

        Safe to call multiple times - removes attempted pgids from set.
        No-op on Windows. Silently handles ProcessLookupError/PermissionError.
        """
        if sys.platform == "win32":
            return
        pgids = _SIGINT_FORWARD_PGIDS.copy()
        _SIGINT_FORWARD_PGIDS.difference_update(pgids)
        for pgid in pgids:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass  # Process already dead or permission issue - expected during shutdown

    def run(
        self,
        cmd: list[str] | str,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        use_process_group: bool | None = None,
        shell: bool = False,
        cwd: Path | None = None,
    ) -> CommandResult:
        """Run a command synchronously.

        Uses Popen with process-group termination on timeout. When timeout
        occurs, sends SIGTERM to the process group, waits kill_grace_seconds,
        then sends SIGKILL if still running.

        Args:
            cmd: Command to run. Can be a list of strings (for non-shell mode)
                or a shell string when shell=True.
            env: Environment variables (merged with os.environ).
            timeout: Override default timeout for this command.
            use_process_group: Whether to use process group for termination.
                If None (default), uses process group on Unix, disabled on Windows.
                When True, creates a new session and kills the entire process group
                on timeout using os.killpg(). When False, only terminates the
                main process. Note: On Windows, process groups are not supported
                and this parameter is ignored (always treated as False).
            shell: If True, run command through shell (cmd should be a string).
                Defaults to False for backwards compatibility.
            cwd: Override working directory for this command. If None, uses self.cwd.

        Returns:
            CommandResult with execution details.
        """
        effective_timeout = timeout if timeout is not None else self.timeout_seconds
        effective_cwd = cwd if cwd is not None else self.cwd
        merged_env = self._merge_env(env)

        # Use process group on Unix for proper child termination (default behavior)
        # On Windows, os.killpg is not available, so always disable process groups
        effective_use_process_group = (
            use_process_group
            if use_process_group is not None
            else sys.platform != "win32"
        ) and sys.platform != "win32"

        start = time.monotonic()
        proc = subprocess.Popen(
            cmd,
            cwd=effective_cwd,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=effective_use_process_group,
            shell=shell,
        )
        pgid = proc.pid if effective_use_process_group else None
        if pgid is not None:
            _register_sigint_pgid(pgid)

        try:
            try:
                stdout, stderr = proc.communicate(timeout=effective_timeout)
                duration = time.monotonic() - start
                return CommandResult(
                    command=cmd,
                    returncode=proc.returncode,
                    stdout=stdout or "",
                    stderr=stderr or "",
                    duration_seconds=duration,
                    timed_out=False,
                )
            except subprocess.TimeoutExpired:
                # Terminate the process group properly
                duration = time.monotonic() - start
                stdout, stderr = self._terminate_process_sync(
                    proc, effective_use_process_group
                )
                return CommandResult(
                    command=cmd,
                    returncode=TIMEOUT_EXIT_CODE,
                    stdout=stdout,
                    stderr=stderr,
                    duration_seconds=duration,
                    timed_out=True,
                )
        finally:
            if pgid is not None:
                _unregister_sigint_pgid(pgid)

    def _terminate_process_sync(
        self,
        proc: subprocess.Popen[str],
        use_process_group: bool,
    ) -> tuple[str, str]:
        """Terminate a process and all its children synchronously.

        Sends SIGTERM to the process group, waits kill_grace_seconds,
        then sends SIGKILL if still running.

        Args:
            proc: The Popen process to terminate.
            use_process_group: Whether to kill the entire process group.

        Returns:
            Tuple of (stdout, stderr) captured before termination.
        """
        pgid = proc.pid if use_process_group else None

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
            stdout, stderr = proc.communicate(timeout=self.kill_grace_seconds)
            return stdout or "", stderr or ""
        except subprocess.TimeoutExpired:
            pass

        # Force kill if still running
        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        else:
            proc.kill()

        # Wait for process to exit and capture any remaining output
        # Use a bounded timeout to avoid hanging if children hold pipes open
        # (can happen when use_process_group=False and children survive)
        try:
            stdout, stderr = proc.communicate(timeout=self.kill_grace_seconds)
            return stdout or "", stderr or ""
        except subprocess.TimeoutExpired:
            # Children are holding pipes open; close them and give up on output
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()
            proc.wait()
            return "", ""

    async def run_async(
        self,
        cmd: list[str] | str,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        use_process_group: bool | None = None,
        shell: bool = False,
        cwd: Path | None = None,
    ) -> CommandResult:
        """Run a command asynchronously.

        Uses asyncio subprocess for proper async timeout handling with
        process-group termination. On timeout, sends SIGTERM to the process
        group, waits kill_grace_seconds, then sends SIGKILL if still running.

        Args:
            cmd: Command to run. Can be a list of strings (for non-shell mode)
                or a shell string when shell=True.
            env: Environment variables (merged with os.environ).
            timeout: Override default timeout for this command.
            use_process_group: Whether to use process group for termination.
                If None (default), uses process group on Unix, disabled on Windows.
                When True, creates a new session (os.setsid) and kills the entire
                process group on timeout using os.killpg(). When False, only
                terminates the main process. Note: On Windows, process groups
                are not supported and this parameter is ignored (always treated
                as False).
            shell: If True, run command through shell (cmd should be a string).
                Defaults to False for backwards compatibility.
            cwd: Override working directory for this command. If None, uses self.cwd.

        Returns:
            CommandResult with execution details.
        """
        effective_timeout = timeout if timeout is not None else self.timeout_seconds
        effective_cwd = cwd if cwd is not None else self.cwd
        merged_env = self._merge_env(env)

        # Use process group on Unix for proper child termination (default behavior)
        # On Windows, os.killpg is not available, so always disable process groups
        effective_use_process_group = (
            use_process_group
            if use_process_group is not None
            else sys.platform != "win32"
        ) and sys.platform != "win32"

        start = time.monotonic()
        try:
            if shell:
                # For shell mode, use create_subprocess_shell with command as string
                if isinstance(cmd, list):
                    cmd = " ".join(cmd)
                proc = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=effective_cwd,
                    env=merged_env,
                    start_new_session=effective_use_process_group,
                )
            else:
                # For non-shell mode, use create_subprocess_exec
                if isinstance(cmd, str):
                    raise ValueError("cmd must be a list when shell=False")
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=effective_cwd,
                    env=merged_env,
                    start_new_session=effective_use_process_group,
                )

            pgid = proc.pid if effective_use_process_group else None
            if pgid is not None:
                _register_sigint_pgid(pgid)

            try:
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
                    await self._terminate_process(proc, effective_use_process_group)
                    return CommandResult(
                        command=cmd,
                        returncode=TIMEOUT_EXIT_CODE,
                        stdout="",
                        stderr="",
                        duration_seconds=duration,
                        timed_out=True,
                    )
            finally:
                if pgid is not None:
                    _unregister_sigint_pgid(pgid)
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
    cmd: list[str] | str,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float | None = None,
    shell: bool = False,
) -> CommandResult:
    """Convenience function for running a single command.

    Args:
        cmd: Command to run. Can be a list of strings (for non-shell mode)
            or a shell string when shell=True.
        cwd: Working directory.
        env: Environment variables (merged with os.environ).
        timeout_seconds: Timeout for the command.
        shell: If True, run command through shell (cmd should be a string).
            Defaults to False for backwards compatibility.

    Returns:
        CommandResult with execution details.
    """
    runner = CommandRunner(cwd=cwd, timeout_seconds=timeout_seconds)
    return runner.run(cmd, env=env, shell=shell)


async def run_command_async(
    cmd: list[str] | str,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float | None = None,
    shell: bool = False,
) -> CommandResult:
    """Convenience function for running a single command asynchronously.

    Args:
        cmd: Command to run. Can be a list of strings (for non-shell mode)
            or a shell string when shell=True.
        cwd: Working directory.
        env: Environment variables (merged with os.environ).
        timeout_seconds: Timeout for the command.
        shell: If True, run command through shell (cmd should be a string).
            Defaults to False for backwards compatibility.

    Returns:
        CommandResult with execution details.
    """
    runner = CommandRunner(cwd=cwd, timeout_seconds=timeout_seconds)
    return await runner.run_async(cmd, env=env, shell=shell)
