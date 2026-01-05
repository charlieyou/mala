"""Fake command runner for testing.

Provides FakeCommandRunner implementing CommandRunnerPort with fail-closed semantics:
unregistered commands raise errors by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

from src.infra.tools.command_runner import CommandResult


class UnregisteredCommandError(Exception):
    """Raised when FakeCommandRunner receives an unregistered command.

    Fail-closed design: tests must explicitly register expected commands,
    preventing silent passes when wrong commands are executed.
    """

    def __init__(self, cmd: tuple[str, ...]) -> None:
        self.cmd = cmd
        super().__init__(f"Unregistered command: {cmd}")


@dataclass
class FakeCommandRunner:
    """In-memory command runner for testing.

    Implements CommandRunnerPort with fail-closed semantics: commands must be
    registered before execution, preventing tests from silently passing when
    unexpected commands are run.

    Attributes:
        responses: Map of command tuples to their results.
        allow_unregistered: If True, return success for unregistered commands.
        calls: Observable list of (cmd_tuple, kwargs) for assertions.
    """

    responses: dict[tuple[str, ...], CommandResult] = field(default_factory=dict)
    allow_unregistered: bool = False
    calls: list[tuple[tuple[str, ...], dict[str, Any]]] = field(default_factory=list)

    def _normalize_cmd(self, cmd: list[str] | str | tuple[str, ...]) -> tuple[str, ...]:
        """Convert command to normalized tuple form."""
        if isinstance(cmd, str):
            return (cmd,)
        return tuple(cmd)

    def _execute(
        self,
        cmd: list[str] | str,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        use_process_group: bool | None = None,
        shell: bool = False,
        cwd: Path | None = None,
    ) -> CommandResult:
        """Core execution logic shared by run() and run_async()."""
        cmd_tuple = self._normalize_cmd(cmd)
        kwargs: dict[str, Any] = {
            "env": env,
            "timeout": timeout,
            "use_process_group": use_process_group,
            "shell": shell,
            "cwd": cwd,
        }
        self.calls.append((cmd_tuple, kwargs))

        if cmd_tuple in self.responses:
            return self.responses[cmd_tuple]

        if self.allow_unregistered:
            # Return success result for unregistered commands
            # Preserve original cmd type (str or list) like real CommandRunner
            return CommandResult(
                command=cmd if isinstance(cmd, str) else list(cmd),
                returncode=0,
                stdout="",
                stderr="",
            )

        raise UnregisteredCommandError(cmd_tuple)

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

        Args:
            cmd: Command to run. Can be a list of strings or a shell string.
            env: Environment variables to set (merged with os.environ).
            timeout: Timeout for command execution in seconds.
            use_process_group: Whether to use process group for termination.
            shell: If True, run command through shell.
            cwd: Override working directory for this command.

        Returns:
            CommandResult with execution details.

        Raises:
            UnregisteredCommandError: If command not registered and allow_unregistered=False.
        """
        return self._execute(cmd, env, timeout, use_process_group, shell, cwd)

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

        Args:
            cmd: Command to run. Can be a list of strings or a shell string.
            env: Environment variables to set (merged with os.environ).
            timeout: Timeout for command execution in seconds.
            use_process_group: Whether to use process group for termination.
            shell: If True, run command through shell.
            cwd: Override working directory for this command.

        Returns:
            CommandResult with execution details.

        Raises:
            UnregisteredCommandError: If command not registered and allow_unregistered=False.
        """
        return self._execute(cmd, env, timeout, use_process_group, shell, cwd)

    def has_call_with_prefix(self, prefix: list[str] | str | tuple[str, ...]) -> bool:
        """Check if any call starts with the given prefix.

        Args:
            prefix: Command parts to match at start. Can be list, str, or tuple.

        Returns:
            True if any recorded call starts with prefix.
        """
        prefix_tuple = self._normalize_cmd(prefix)
        return any(
            cmd_tuple[: len(prefix_tuple)] == prefix_tuple
            for cmd_tuple, _ in self.calls
        )

    def get_calls_with_prefix(
        self, prefix: list[str] | str | tuple[str, ...]
    ) -> list[tuple[tuple[str, ...], dict[str, Any]]]:
        """Get all calls that start with the given prefix.

        Args:
            prefix: Command parts to match at start. Can be list, str, or tuple.

        Returns:
            List of (cmd_tuple, kwargs) for matching calls.
        """
        prefix_tuple = self._normalize_cmd(prefix)
        return [
            (cmd_tuple, kwargs)
            for cmd_tuple, kwargs in self.calls
            if cmd_tuple[: len(prefix_tuple)] == prefix_tuple
        ]

    def has_call_containing(self, substring: str) -> bool:
        """Check if any call contains the given substring (useful for shell commands).

        Args:
            substring: Text to search for within any command part.

        Returns:
            True if any recorded call contains the substring.
        """
        return any(
            substring in part for cmd_tuple, _ in self.calls for part in cmd_tuple
        )

    def get_calls_containing(self, substring: str) -> list[tuple[str, ...]]:
        """Get all calls that contain the given substring.

        Args:
            substring: Text to search for within any command part.

        Returns:
            List of cmd_tuples that contain the substring.
        """
        return [
            cmd_tuple
            for cmd_tuple, _ in self.calls
            if any(substring in part for part in cmd_tuple)
        ]
