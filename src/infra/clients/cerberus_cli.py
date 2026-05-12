"""Cerberus CLI subprocess management.

This module provides CerberusCLI for managing subprocess interactions
with the cerberus v2 CLI binary. It handles:
- Binary validation (exists and is executable)
- Environment variable merging for the v2 env contract
- Subprocess spawn/wait/resolve with timeout handling
- Exit code interpretation

This is a low-level component extracted from DefaultReviewer to enable
independent testing of subprocess logic.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.core.protocols.infra import CommandRunnerPort


@dataclass
class SpawnResult:
    """Result of spawning a code review.

    Attributes:
        success: Whether spawn succeeded.
        timed_out: Whether spawn timed out.
        error_detail: Error message if spawn failed.
        already_active: Whether failure was due to an existing active gate.
    """

    success: bool
    timed_out: bool = False
    error_detail: str = ""
    already_active: bool = False


@dataclass
class WaitResult:
    """Result of waiting for review completion.

    Attributes:
        returncode: Exit code from wait command.
        stdout: Standard output (JSON).
        stderr: Standard error.
        timed_out: Whether wait timed out.
    """

    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False

    @property
    def session_dir(self) -> Path | None:
        """Legacy adapter-only review log path."""
        return self.__dict__.get("_session_dir")

    @session_dir.setter
    def session_dir(self, value: Path | None) -> None:
        self.__dict__["_session_dir"] = value


@dataclass
class ResolveResult:
    """Result of resolving (clearing) an active gate.

    Attributes:
        success: Whether resolve succeeded.
        error_detail: Error message if resolve failed.
    """

    success: bool
    error_detail: str = ""


@dataclass
class CerberusCLI:
    """Low-level CLI subprocess management for cerberus.

    Handles subprocess spawn/wait/resolve, binary validation, env merging,
    and timeout handling. JSON parsing and result mapping belong to the
    adapter/parser layers.

    Attributes:
        repo_path: Working directory for subprocess execution.
        env: Additional environment variables to merge with os.environ.
    """

    repo_path: Path
    env: dict[str, str] = field(default_factory=dict)

    def _cerberus_bin(self) -> str:
        """Get the cerberus binary name."""
        return "cerberus"

    def validate_binary(self) -> str | None:
        """Validate that the cerberus binary and root prompts are available.

        Uses the merged env's PATH (respecting self.env) when checking for
        the binary to avoid false negatives when callers inject PATH via cerberus_env.

        Returns:
            None if the binary is valid, or an error message if not.
        """
        effective_path = self._effective_path()
        if shutil.which("cerberus", path=effective_path) is None:
            return "cerberus binary not found in PATH"

        root = self._resolve_cerberus_root(effective_path)
        if root is None or not self._has_reviewer_prompts(root):
            root_label = str(root) if root is not None else "<unresolved>"
            return f"cerberus root {root_label} missing prompts/reviewers/"
        return None

    def build_env(
        self,
        *,
        run_key: str,
        state_root: Path,
        project_key: str | None = None,
        claude_session_id: str | None = None,
    ) -> dict[str, str]:
        """Build environment dict for the cerberus v2 env contract.

        Args:
            run_key: Cerberus run key owned by mala.
            state_root: Cerberus state root owned by mala.
            project_key: Project key default; user env may override it.
            claude_session_id: Optional Claude session ID to preserve.

        Returns:
            Merged environment dict with v2 cerberus keys set.
        """
        merged = dict(self.env)
        merged.setdefault("CERBERUS_HOST", "generic")
        if project_key is not None:
            merged.setdefault("CERBERUS_PROJECT_KEY", project_key)
        else:
            merged.setdefault("CERBERUS_PROJECT_KEY", "")

        root = self._resolve_cerberus_root(self._effective_path())
        if root is not None and not merged.get("CERBERUS_ROOT"):
            merged["CERBERUS_ROOT"] = str(root)

        # Mala-owned keys must address the exact state directory that wait reads.
        merged["CERBERUS_RUN_KEY"] = run_key
        merged["CERBERUS_STATE_ROOT"] = str(state_root)

        if claude_session_id:
            merged["CLAUDE_SESSION_ID"] = claude_session_id
        else:
            merged.pop("CLAUDE_SESSION_ID", None)
        return merged

    def _effective_path(self) -> str:
        """Return the PATH Cerberus subprocesses will use."""
        if "PATH" in self.env:
            return self.env["PATH"]
        return os.environ.get("PATH", "")

    def _resolve_cerberus_root(self, effective_path: str) -> Path | None:
        """Resolve CERBERUS_ROOT from user env, process env, or the binary path."""
        if configured_root := self.env.get("CERBERUS_ROOT"):
            return Path(configured_root)
        if process_root := os.environ.get("CERBERUS_ROOT"):
            return Path(process_root)
        if binary := shutil.which("cerberus", path=effective_path):
            return Path(binary).resolve().parent.parent
        return None

    @staticmethod
    def _has_reviewer_prompts(root: Path) -> bool:
        """Return True when cerberus root contains reviewer prompts."""
        return (root / "prompts" / "reviewers").is_dir()

    async def spawn_code_review(
        self,
        runner: CommandRunnerPort,
        env: dict[str, str],
        timeout: int,
        *,
        commit_shas: Sequence[str],
        spawn_args: tuple[str, ...] = (),
        context_file: Path | None = None,
    ) -> SpawnResult:
        """Spawn a code review subprocess.

        Args:
            runner: CommandRunnerPort instance for subprocess execution.
            env: Environment variables for the command.
            timeout: Timeout in seconds.
            commit_shas: List of commit SHAs for commit-based review.
            spawn_args: Additional arguments for spawn-code-review.
            context_file: Optional context file path.

        Returns:
            SpawnResult with success/failure status and error details.
        """
        spawn_cmd = [self._cerberus_bin(), "spawn-code-review"]
        # Always exclude .beads/ directory (auto-generated issue tracker files)
        spawn_cmd.extend(["--exclude", ".beads/"])
        if context_file is not None:
            spawn_cmd.extend(["--context-file", str(context_file)])
        if spawn_args:
            spawn_cmd.extend(spawn_args)
        spawn_cmd.append("--commit")
        spawn_cmd.extend(commit_shas)

        result = await runner.run_async(spawn_cmd, env=env, timeout=timeout)
        if result.timed_out:
            return SpawnResult(
                success=False, timed_out=True, error_detail="spawn timeout"
            )

        if result.returncode != 0:
            stderr = result.stderr_tail()
            stdout = result.stdout_tail()
            detail = stderr or stdout or "spawn failed"
            combined = f"{stderr or ''} {stdout or ''}".lower()
            already_active = "already active" in combined
            return SpawnResult(
                success=False,
                timed_out=False,
                error_detail=detail,
                already_active=already_active,
            )

        return SpawnResult(success=True)

    async def spawn_epic_verify(
        self,
        runner: CommandRunnerPort,
        env: dict[str, str],
        timeout: int,
        *,
        epic_path: Path,
        spawn_args: tuple[str, ...] = (),
    ) -> SpawnResult:
        """Spawn an epic verification subprocess.

        Args:
            runner: CommandRunnerPort instance for subprocess execution.
            env: Environment variables for the command.
            timeout: Timeout in seconds.
            epic_path: Path to the epic markdown file.
            spawn_args: Additional arguments for spawn-epic-verify.

        Returns:
            SpawnResult with success/failure status and error details.
        """
        spawn_cmd = [self._cerberus_bin(), "spawn-epic-verify"]
        if spawn_args:
            spawn_cmd.extend(spawn_args)
        spawn_cmd.append(str(epic_path))

        result = await runner.run_async(spawn_cmd, env=env, timeout=timeout)
        if result.timed_out:
            return SpawnResult(
                success=False, timed_out=True, error_detail="spawn timeout"
            )

        if result.returncode != 0:
            stderr = result.stderr_tail()
            stdout = result.stdout_tail()
            detail = stderr or stdout or "spawn failed"
            combined = f"{stderr or ''} {stdout or ''}".lower()
            already_active = "already active" in combined
            return SpawnResult(
                success=False,
                timed_out=False,
                error_detail=detail,
                already_active=already_active,
            )

        return SpawnResult(success=True)

    async def wait_for_review(
        self,
        run_key: str,
        runner: CommandRunnerPort,
        env: dict[str, str],
        cli_timeout: int,
        wait_args: tuple[str, ...] = (),
        user_timeout: int | None = None,
    ) -> WaitResult:
        """Wait for review completion.

        Args:
            run_key: Cerberus run key to wait for.
            runner: CommandRunnerPort instance for subprocess execution.
            env: Environment variables for the command.
            cli_timeout: Timeout value to pass to CLI (if user_timeout is None).
            wait_args: Additional arguments for wait command.
            user_timeout: User-specified timeout (if already in wait_args).

        Returns:
            WaitResult with returncode, stdout, stderr, and timeout status.
        """
        wait_cmd = [
            self._cerberus_bin(),
            "wait",
            "--json",
            "--finalize",
            "--session-key",
            run_key,
        ]
        # Only add --timeout if not already specified in wait_args
        if user_timeout is None:
            wait_cmd.extend(["--timeout", str(cli_timeout)])
        if wait_args:
            wait_cmd.extend(wait_args)

        # Use CLI timeout + grace period for subprocess timeout
        effective_timeout = (
            user_timeout if user_timeout is not None else cli_timeout
        ) + 30
        result = await runner.run_async(wait_cmd, env=env, timeout=effective_timeout)

        return WaitResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            timed_out=result.timed_out,
        )

    async def resolve_gate(
        self,
        runner: CommandRunnerPort,
        env: dict[str, str],
        reason: str = "mala: auto-clearing stale gate for retry",
    ) -> ResolveResult:
        """Resolve (clear) an active gate.

        Args:
            runner: CommandRunnerPort instance for subprocess execution.
            env: Environment variables for the command.
            reason: Reason message for the resolve command.

        Returns:
            ResolveResult with success/failure status and error details.
        """
        resolve_cmd = [
            self._cerberus_bin(),
            "resolve",
            "--reason",
            reason,
        ]
        try:
            result = await runner.run_async(resolve_cmd, env=env, timeout=30)
            if result.returncode == 0:
                return ResolveResult(success=True)
            stderr = result.stderr.strip() if result.stderr else ""
            stdout = result.stdout.strip() if result.stdout else ""
            return ResolveResult(
                success=False, error_detail=stderr or stdout or "resolve failed"
            )
        except Exception as e:
            return ResolveResult(success=False, error_detail=str(e))

    @staticmethod
    def extract_wait_timeout(args: tuple[str, ...]) -> int | None:
        """Extract --timeout value from wait args if provided.

        Parses args to detect if a --timeout flag is already specified.
        Supports both '--timeout VALUE' and '--timeout=VALUE' formats.

        Args:
            args: Tuple of command-line arguments to search.

        Returns:
            The timeout value as int if found and valid, None otherwise.
        """
        i = 0
        while i < len(args):
            arg = args[i]
            if arg.startswith("--timeout="):
                value = arg.split("=", 1)[1]
                if value.isdigit():
                    return int(value)
                return None
            if arg == "--timeout" and i + 1 < len(args):
                value = args[i + 1]
                if value.isdigit():
                    return int(value)
                return None
            i += 1
        return None
