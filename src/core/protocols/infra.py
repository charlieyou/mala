"""Infrastructure protocols for command execution, environment, and locking.

This module defines protocols for abstracting infrastructure concerns,
enabling dependency injection and testability for command runners,
environment configuration, lock managers, and logging.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


@runtime_checkable
class CommandResultProtocol(Protocol):
    """Protocol for command execution results.

    Matches the interface of src.infra.tools.command_runner.CommandResult
    for structural typing without import-time dependencies.
    """

    command: list[str] | str
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_seconds: float

    def stdout_tail(self, max_chars: int = 800, max_lines: int = 20) -> str:
        """Get truncated stdout.

        Args:
            max_chars: Maximum number of characters.
            max_lines: Maximum number of lines.

        Returns:
            Truncated stdout string.
        """
        ...

    def stderr_tail(self, max_chars: int = 800, max_lines: int = 20) -> str:
        """Get truncated stderr.

        Args:
            max_chars: Maximum number of characters.
            max_lines: Maximum number of lines.

        Returns:
            Truncated stderr string.
        """
        ...


@runtime_checkable
class CommandRunnerPort(Protocol):
    """Protocol for abstracting command execution.

    Enables dependency injection of command runners into domain modules,
    allowing the core layer to define the interface without depending on
    the infrastructure implementation.

    The canonical implementation is CommandRunner in
    src/infra/tools/command_runner.py.
    """

    def run(
        self,
        cmd: list[str] | str,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        use_process_group: bool | None = None,
        shell: bool = False,
        cwd: Path | None = None,
    ) -> CommandResultProtocol:
        """Run a command synchronously.

        Args:
            cmd: Command to run. Can be a list of strings or a shell string.
            env: Environment variables to set (merged with os.environ).
            timeout: Timeout for command execution in seconds.
            use_process_group: Whether to use process group for termination.
            shell: If True, run command through shell.
            cwd: Override working directory for this command.

        Returns:
            CommandResultProtocol with execution details.
        """
        ...

    async def run_async(
        self,
        cmd: list[str] | str,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        use_process_group: bool | None = None,
        shell: bool = False,
        cwd: Path | None = None,
    ) -> CommandResultProtocol:
        """Run a command asynchronously.

        Args:
            cmd: Command to run. Can be a list of strings or a shell string.
            env: Environment variables to set (merged with os.environ).
            timeout: Timeout for command execution in seconds.
            use_process_group: Whether to use process group for termination.
            shell: If True, run command through shell.
            cwd: Override working directory for this command.

        Returns:
            CommandResultProtocol with execution details.
        """
        ...


@runtime_checkable
class EnvConfigPort(Protocol):
    """Protocol for abstracting environment configuration.

    Enables dependency injection of environment config into domain modules,
    allowing the core layer to define the interface without depending on
    the infrastructure implementation.

    The canonical implementation is in src/infra/tools/env.py.
    """

    @property
    def scripts_dir(self) -> Path:
        """Path to the scripts directory (e.g., test-mutex.sh)."""
        ...

    @property
    def cache_dir(self) -> Path:
        """Path to the mala cache directory."""
        ...

    @property
    def lock_dir(self) -> Path:
        """Path to the lock directory for multi-agent coordination."""
        ...

    def find_cerberus_bin_path(self) -> Path | None:
        """Find the cerberus plugin bin directory.

        Returns:
            Path to cerberus bin directory, or None if not found.
        """
        ...


@runtime_checkable
class LockManagerPort(Protocol):
    """Protocol for abstracting file-based locking operations.

    Enables dependency injection of lock managers into domain modules,
    allowing the core layer to define the interface without depending on
    the infrastructure implementation.

    The canonical implementation is in src/infra/tools/locking.py.
    """

    def lock_path(self, filepath: str, repo_namespace: str | None = None) -> Path:
        """Get the lock file path for a given filepath.

        Args:
            filepath: Path to the file to lock.
            repo_namespace: Optional repo namespace for cross-repo disambiguation.

        Returns:
            Path to the lock file.
        """
        ...

    def try_lock(
        self, filepath: str, agent_id: str, repo_namespace: str | None = None
    ) -> bool:
        """Try to acquire a lock without blocking.

        Args:
            filepath: Path to the file to lock.
            agent_id: Identifier of the agent requesting the lock.
            repo_namespace: Optional repo namespace for cross-repo disambiguation.

        Returns:
            True if lock was acquired, False if already held by another agent.
        """
        ...

    def wait_for_lock(
        self,
        filepath: str,
        agent_id: str,
        repo_namespace: str | None = None,
        timeout_seconds: float = 30.0,
        poll_interval_ms: int = 100,
    ) -> bool:
        """Wait for and acquire a lock on a file.

        Args:
            filepath: Path to the file to lock.
            agent_id: Identifier of the agent requesting the lock.
            repo_namespace: Optional repo namespace for cross-repo disambiguation.
            timeout_seconds: Maximum time to wait for the lock in seconds.
            poll_interval_ms: Polling interval in milliseconds.

        Returns:
            True if lock was acquired, False if timeout.
        """
        ...

    def release_lock(
        self, filepath: str, agent_id: str, repo_namespace: str | None = None
    ) -> bool:
        """Release a lock on a file.

        Only releases the lock if it is held by the specified agent_id.
        This prevents accidental or malicious release of locks held by
        other agents.

        Args:
            filepath: Path to the file to unlock.
            agent_id: Identifier of the agent releasing the lock.
            repo_namespace: Optional repo namespace for cross-repo disambiguation.

        Returns:
            True if lock was released, False if lock was not held by agent_id.
        """
        ...


@runtime_checkable
class LoggerPort(Protocol):
    """Protocol for console/terminal logging with colored output.

    Enables dependency injection of loggers into domain modules,
    allowing the core layer to define the interface without depending on
    the infrastructure implementation.

    The canonical implementation is in src/infra/io/log_output/console.py.
    """

    def log(
        self,
        message: str,
        *,
        level: str = "info",
        color: str | None = None,
    ) -> None:
        """Log a message to the console.

        Args:
            message: The message to log.
            level: Log level (e.g., "info", "debug", "error").
            color: Optional color name (e.g., "cyan", "green", "red").
        """
        ...
